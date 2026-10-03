"""`LoopTraceCollector` implementation: wires `src.model.loop_trace`'s raw
tensors into the JSONL record functions in `collector.py`.

This is the thin adapter layer: `collector.py`'s functions
take plain tensors and know nothing about `src.model.loop_trace` or any
trainer; this module is the only place that imports `TracePoint` /
`TraceMeta` / `LossComponents` and translates them into calls on those pure
functions. If the training backend changes (e.g. a future Megatron port), only
this file and `loop_trace.py` should need to change --
`collector.py`'s tested record-shaping logic does not.

Tensor lifetime: every function in `collector.py` this module calls reduces
its inputs to Python floats/lists before returning (`.item()`/`.tolist()`),
so no tensor reference outlives `on_trace_step`'s or `on_probe_step`'s call
-- satisfying `loop_trace.py`'s "collector must not retain tensors" contract
without this module needing any extra bookkeeping of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.model.loop_trace import (
    LOOP_AXIS_PROVENANCE,
    ROUTER_LOGITS_ARE_PRESOFTMAX,
    LossComponents,
    TraceMeta,
    TraceSink,
)
from src.metrics.collector import (
    JsonlWriter,
    loss_record,
    overlap_record,
    repeat_stratified_loss_records,
    residual_norm_records,
    router_layer_loop_record,
    state_cosine_records,
)


def _distributed_scope() -> dict[str, object]:
    """Which rank's tokens the router numbers cover.

    Always "rank_local": these are computed inside one process's forward, over
    that rank's shard of the batch. Cross-rank aggregation does not exist yet
    (infra outline 2.4.1, second half), and labelling every row keeps "the rows
    appeared" from being mistaken for "the gap is closed".
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return {
                "aggregation_scope": "rank_local",
                "rank": dist.get_rank(),
                "world_size": dist.get_world_size(),
            }
    except Exception:  # noqa: BLE001 - provenance must never break collection
        pass
    return {"aggregation_scope": "rank_local", "rank": 0, "world_size": 1}


class JsonlMetricsCollector:
    """Implements `loop_trace.LoopTraceCollector` by writing every record to
    `train_metrics.jsonl` (see `collector.py`'s module docstring for the file
    layout / `kind`-routing convention).

    `cos_layer_idxs`: forwarded to `state_cosine_records` (logged every N
    steps) to control JSONL volume -- `None`
    (default) logs every layer present in a trace step; pass a smaller list
    (e.g. `[num_layers_in_stack - 1]`, last layer only) to cut volume. This
    is a caller-supplied choice, not a default baked in here (no default
    value silently narrows what gets logged -- callers who want "every
    layer" must not have to guess that `None` means that).

    Probe-tier args (`probe_metrics_path`, `probe_doc_buckets`,
    `probe_n_buckets`, `probe_input_ids_sha8`) are all STATIC for the whole
    training run -- the fixed probe corpus never changes during training
    (changing it would break comparability across checkpoints),
    so they belong at construction time, not threaded through every
    `on_probe_step` call (`run_probe_forward`'s signature has no room for
    them either -- it only knows
    `model`/`input_ids`/`collector`). `on_probe_step` raises if called
    without these configured -- there is no meaningful default for "which
    documents are in which repetition-rate bucket" (semantic parameters have
    no defaults).
    """

    def __init__(
        self,
        train_metrics_path: str | Path,
        *,
        cos_layer_idxs: list[int] | None = None,
        probe_metrics_path: str | Path | None = None,
        probe_doc_buckets: list[int] | None = None,
        probe_n_buckets: int | None = None,
        probe_input_ids_sha8: str | None = None,
    ) -> None:
        self._writer = JsonlWriter(train_metrics_path)
        self._cos_layer_idxs = cos_layer_idxs

        self._probe_writer = JsonlWriter(probe_metrics_path) if probe_metrics_path is not None else None
        self._probe_doc_buckets = probe_doc_buckets
        self._probe_n_buckets = probe_n_buckets
        self._probe_input_ids_sha8 = probe_input_ids_sha8

    def close(self) -> None:
        self._writer.close()
        if self._probe_writer is not None:
            self._probe_writer.close()

    def __enter__(self) -> "JsonlMetricsCollector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- LoopTraceCollector protocol -----------------------------------

    def on_metrics_step(
        self,
        step: int,
        loss: LossComponents,
        grad_norm: float | None = None,
        lr: float | None = None,
    ) -> dict[str, float]:
        """Every step. One `kind="loss"` JSONL row.

        `grad_norm` is optional and sparse -- only the steps torchtitan itself
        logs carry one. `lr` is the rate actually in force, read from the live
        optimizer by the caller rather than recomputed. See `loss_record`.
        """
        rec = loss_record(
            step=step,
            task_loss=loss.task_loss,
            lb_loss=loss.lb_loss,
            lb_loss_factor=loss.lb_loss_factor,
            z_loss=loss.z_loss,
            z_loss_factor=loss.z_loss_factor,
            total_loss=loss.total_loss,
            grad_norm=grad_norm,
            lr=lr,
        )
        self._writer.write(rec)
        return {k: v for k, v in rec.items() if k not in ("kind", "step")}

    def on_trace_step(self, step: int, sink: TraceSink, meta: TraceMeta) -> dict[str, float]:
        """Every N steps. Writes one `kind="router"` row
        per (loop_step, layer_idx), plus `kind="residual"` and
        `kind="state_cosine"` rows derived from the same trace's residual
        tensors.

        ⚠️ Known gap: under FSDP2 in production,
        the training loop is not currently calling this with a populated
        `sink` -- see `collector.py`'s module docstring for the full note and
        the offline-recompute fallback. This method's own logic and tests are
        unaffected; the gap is upstream, in what the trainer passes in.

        Sanity checks (raise, do not silently proceed): the trace's
        declared semantic markers must match this module's assumptions.
        These are constants on both sides today, so a mismatch here would
        mean one side changed without the other -- exactly the "looked
        entirely reasonable but was all wrong" failure this project's
        discipline exists to catch, not a hypothetical.
        """
        if meta.router_logits_are_presoftmax is not ROUTER_LOGITS_ARE_PRESOFTMAX:
            raise ValueError(
                f"TraceMeta.router_logits_are_presoftmax={meta.router_logits_are_presoftmax!r} "
                f"disagrees with this module's assumption ({ROUTER_LOGITS_ARE_PRESOFTMAX!r}); "
                "router_layer_loop_record calls moe.topk_margin on raw logits and would "
                "silently compute a wrong-scale margin if this trace's logits are not "
                "actually pre-softmax."
            )
        if meta.loop_axis_provenance != LOOP_AXIS_PROVENANCE:
            raise ValueError(
                f"TraceMeta.loop_axis_provenance={meta.loop_axis_provenance!r} != "
                f"{LOOP_AXIS_PROVENANCE!r} -- this project's discipline requires the loop "
                "axis to come from an explicit call counter, never a reshape; a trace "
                "declaring a different provenance must not be silently accepted."
            )

        residual_by_loop_layer = {}
        entropy_ratios: list[float] = []
        maxvio_masses: list[float] = []

        # Keyed and ORDERED by `unrolled_pos`, never by `(loop_step, layer_idx)`:
        # head, the loop's first layer and tail all carry (0, 0), so grouping or
        # sorting on that pair silently merges three different depths into one.
        # Sorting also makes the rows come out in true depth order, so a reader
        # does not have to reconstruct it.
        for unrolled_pos, point in sorted(sink.items()):
            loop_step, layer_idx = point.loop_step, point.layer_idx
            rec = router_layer_loop_record(
                step=step,
                layer_idx=layer_idx,
                loop_step=loop_step,
                router_logits=point.router_logits,
                topk_idx=point.topk_idx,
                topk_weights=point.topk_weights,
                # THIS layer's E and k, not the model's. `meta.num_experts` is
                # the loop block's E; head and tail keep their own count (8 by
                # default, independent of the loop). Using the meta value builds
                # a per-expert vector of the wrong length for those two rows:
                #
                #   loop E < head/tail E (S13: 4 vs 8) -- `scatter_add_` gets an
                #     index past the end and the run CRASHES at the first
                #     diagnostic step;
                #   loop E > head/tail E (S2: 16 vs 8) -- no crash, and that is
                #     the worse case: the vector is padded with phantom zero
                #     experts, which halves the mean and inflates MaxVio, counts
                #     the padding as dead experts, and divides entropy by
                #     log(16) when the layer only has 8 slots.
                #
                # `TracePoint` carries both per row, read off the router tensor's
                # own shape, so this is the value the numbers were computed from.
                n_experts=point.num_experts,
                top_k=point.top_k,
                **_distributed_scope(),
            )
            # Where in the unrolled network this row came from. `block` is what
            # tells a reader whether `loop_step`/`layer_idx` mean anything at all:
            # a head or tail row reports 0/0 for both, which is indistinguishable
            # from the loop's first layer without it.
            rec["block"] = point.block
            rec["unrolled_pos"] = unrolled_pos
            self._writer.write(rec)
            entropy_ratios.append(rec["entropy_ratio_vs_uniform"])
            maxvio_masses.append(rec["maxvio_mass"])
            if point.block == "loop":
                # `residual_norm_records` reduces over a (loop_step x layer_idx)
                # GRID -- "within a loop step" and "across the step boundary" are
                # defined by those two axes. Head and tail sit outside that grid
                # entirely (they run once), and feeding them in under (0, 0) would
                # overwrite the loop's own first layer and silently change both
                # reductions. They are excluded here rather than remapped: there
                # is no honest (loop_step, layer_idx) for a layer that is not in
                # the loop.
                residual_by_loop_layer[(loop_step, layer_idx)] = point.residual

        norm_recs = residual_norm_records(step=step, residual_by_loop_layer=residual_by_loop_layer)
        for rec in norm_recs:
            self._writer.write(rec)

        cos_recs = state_cosine_records(
            step=step,
            residual_by_loop_layer=residual_by_loop_layer,
            layer_idxs=self._cos_layer_idxs,
        )
        for rec in cos_recs:
            self._writer.write(rec)

        # Flat scalar summary for the caller (e.g. a live dashboard) -- the
        # full per-(layer, loop_step) detail is already on disk; this is a
        # cheap headline aggregate, not a substitute for the JSONL rows.
        summary: dict[str, float] = {}
        if entropy_ratios:
            summary["router/mean_entropy_ratio_vs_uniform"] = sum(entropy_ratios) / len(entropy_ratios)
        if maxvio_masses:
            summary["router/mean_maxvio_mass"] = sum(maxvio_masses) / len(maxvio_masses)
        within = [r["norm_mean"] for r in norm_recs if r["residual_kind"] == "within_step"]
        boundary = [r["norm_mean"] for r in norm_recs if r["residual_kind"] == "step_boundary"]
        if within:
            summary["residual/mean_within_step_norm"] = sum(within) / len(within)
        if boundary:
            summary["residual/mean_step_boundary_norm"] = sum(boundary) / len(boundary)
        return summary

    def on_probe_step(
        self, step: int, sink: TraceSink, meta: TraceMeta, per_token_loss: Tensor
    ) -> dict[str, float]:
        """Every M steps, from a `no_grad` forward on the fixed probe corpus
        ("cheap tier" -- see `run_probe_forward`'s contract in
        `loop_trace.py`). Writes to
        `probe_metrics.jsonl` (a separate file from the free tier's
        `train_metrics.jsonl` -- different `kind`s, different sampling
        cadence, kept apart so a reader never has to filter one out of the
        other by accident).

        `sink` has the exact same shape as `on_trace_step`'s (`TraceSink`,
        keyed by `(loop_step, layer_idx)`) -- a deliberate choice
        to reuse `TracePoint` for the probe path rather than a new type, so
        this method's cross-loop-step grouping logic is the only new code
        needed here; `router_layer_loop_record` is NOT called again on probe
        data (that's the free-tier #1/#2/#3 metrics on the TRAINING batch;
        re-running it on the probe batch would conflate two different token
        populations under the same field names).

        `per_token_loss`'s last column (`run_probe_forward` contract): the last
        position has no
        next-token target and is masked to exactly `0.0`
        (`ignore_index=-100`, guarded by their `test_last_position_is_masked`).
        Averaging that column in with the real losses would silently pull
        every bucket's mean down by `1/seq` (40 docs x 512 tokens -> a
        systematic ~0.2% downward bias, small enough to look like noise
        rather than a bug). This method drops the last column BEFORE
        `repeat_stratified_loss_records` ever sees it -- the pure function
        itself has no opinion on masking (it just means over whatever it's
        given), so the slicing has to happen here, at the one call site that
        knows this tensor came from `run_probe_forward` specifically.

        Raises if this collector was constructed without the probe-tier
        static config (`probe_doc_buckets`/`probe_n_buckets`/
        `probe_input_ids_sha8`) -- see `__init__`'s docstring for why those
        are constructor args, not per-call ones.
        """
        if self._probe_writer is None:
            raise ValueError(
                "on_probe_step called but this collector was constructed without "
                "probe_metrics_path -- pass it to __init__ if probe-tier metrics are wanted."
            )
        if self._probe_doc_buckets is None or self._probe_n_buckets is None or self._probe_input_ids_sha8 is None:
            raise ValueError(
                "on_probe_step called but probe_doc_buckets/probe_n_buckets/"
                "probe_input_ids_sha8 were not supplied to __init__ -- these are static "
                "for the whole run (fixed probe corpus) and have no meaningful default "
                "(see __init__'s docstring)."
            )

        # Overlap compares the SAME physical layer across loop steps, so this one
        # genuinely is per-(layer, loop_step) -- but only the loop's rows have
        # those coordinates. Head and tail run once and have no second visit to
        # compare against, so they are excluded by `block`, not by `loop_step == 0`
        # (which is also true of every head and tail row).
        by_layer: dict[int, dict[int, Tensor]] = {}
        for unrolled_pos, point in sorted(sink.items()):
            if point.block != "loop":
                continue
            by_layer.setdefault(point.layer_idx, {})[point.loop_step] = point.router_logits

        for layer_idx, by_loop in sorted(by_layer.items()):
            loop_steps = sorted(by_loop)
            router_logits_by_loop = torch.stack([by_loop[k] for k in loop_steps], dim=0)
            rec = overlap_record(
                step=step,
                layer_idx=layer_idx,
                router_logits_by_loop=router_logits_by_loop,
                top_k=meta.num_active,
                input_ids_sha8=self._probe_input_ids_sha8,
            )
            self._probe_writer.write(rec)

        # Drop the last (always-0.0, no-target) position -- see docstring above.
        valid_loss = per_token_loss[:, :-1]

        loss_recs = repeat_stratified_loss_records(
            step=step,
            per_token_loss=valid_loss,
            doc_buckets=self._probe_doc_buckets,
            seq=valid_loss.shape[1],
            n_buckets=self._probe_n_buckets,
            input_ids_sha8=self._probe_input_ids_sha8,
        )
        for rec in loss_recs:
            self._probe_writer.write(rec)

        summary: dict[str, float] = {}
        if loss_recs:
            summary["probe/mean_loss_all_buckets"] = sum(r["mean_loss"] for r in loss_recs) / len(loss_recs)
        return summary

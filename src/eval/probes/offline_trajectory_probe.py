"""Offline trajectory-diagnostic driver: depth-wise expert concentration
and cross-loop-step overlap computed post-hoc against a saved trajectory
checkpoint.

## Why this script exists

The computation logic for both metrics already exists in this repo
(`src.metrics.collector.router_layer_loop_record` / `overlap_record`,
themselves thin wrappers around `_vendored_loopdiag.moe`'s verified
formulas) and has been exercised in the training loop's own probe-tier hook
(`src.model.loop_trace.run_probe_forward` -> `LoopTraceCollector.
on_probe_step`). What has never existed is a way to run that same pipeline
**offline, against an already-saved checkpoint**, which is what a post-hoc
"microscope on a finished SFT run" measurement needs -- see `collector.py`'s
own module docstring for the documented gap this closes: the production
FSDP2 training path never actually populated `train_metrics.jsonl`'s
`"router"`/`"overlap"` rows, so there is no in-training data to read back;
these numbers must be produced by a fresh forward pass over the saved
weights.

This module is a **driver only** -- it does not redefine either metric. It
loads a checkpoint, builds the same fixed probe batch the training-time
probe tier would have used, runs one `no_grad` forward with a
`src.model.loop_trace.TraceSink` attached (reusing `run_probe_forward`
verbatim, not reimplementing the isolation guarantees it provides -- RNG
untouched, no autograd graph, model mode restored), and feeds the resulting
tensors to the two existing record functions (this module's output matches
those functions' schemas field-for-field).

## Deviation from `src/eval/run_eval.py`'s "do not import from
`pretrain` beyond two utilities" convention

That convention exists because `run_eval.py` runs in a separate
`looptrain-eval` env that need not carry this project's full training
dependency surface (torchtitan/FSDP2/etc). This module imports
`src.model.loop_trace`, `src.metrics.collector`, and
`src.data.probe_corpus` directly -- but those three modules are
pure-`torch` + stdlib (no torchtitan, no FSDP), and importing them is the
entire point of this script (reuse the existing formula
code, do not re-derive it). The checkpoint itself is still loaded the same
`from_pretrained(..., trust_remote_code=True)` way `run_eval.py` uses, so
the model class actually executed comes from the checkpoint's OWN bundled
`modeling_loop_lm.py` + `loop_trace.py` (`src.pretrain.checkpointing.store.
_bundle_modeling_code` bundles both, precisely so a checkpoint is
self-sufficient for this kind of offline analysis) -- this script's local
`src.model.loop_trace` import is used only for `run_probe_forward`
(a driver-side helper, not part of the model), never for constructing
`TracePoint`/`TraceMeta` instances themselves, so there is no cross-module
identity mismatch to worry about (every access below is attribute-based
duck typing, never `isinstance`).

## Probe corpus: hardcoded, not a CLI knob

The three-point comparison (base checkpoint and two later training stages) requires the
same probe input everywhere --
a CLI flag a caller could vary between invocations would silently break that
guarantee. `PROBE_BATCH`/`PROBE_SEQ` below are therefore fixed module
constants, not `argparse` options: `PROBE_BATCH=40` is every document the
probe corpus artifact has (`src/data/artifacts/corpus/pile10k_d4_b40_s512.json`
-- the "b40" in that filename), maximizing sample size rather than an
arbitrary subset; `PROBE_SEQ=512` matches that same file's "s512", the
sequence length the corpus was prepared at (`probe_corpus.py`'s docstring).
No other precedent in this repo has ever called `build_probe_batch` (this is
the first real caller -- the training-time probe tier documented in
`collector.py` was never actually wired up), so there was no existing
convention to match; deviating from these two numbers requires deliberately
editing this file, not passing a flag.

.. deprecated::
   Superseded by `offline_cross_arch_probe.py` for new work, and not adapted to
   the head/tail architecture: its position grid is still (loop_step, layer_idx),
   which cannot represent a layer outside the loop. Existing artifacts remain
   valid for the runs they describe.

   NOT removable: `src.posttrain.guardrail_eval` calls
   `compute_trajectory_metrics`, and the post-training line depends on it.
   Deprecated for new callers is not the same as deletable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.model.loop_trace import NullCollector, TraceMeta, TraceSink, run_probe_forward
from src.metrics.collector import overlap_record, router_layer_loop_record
from src.data.probe_corpus import DEFAULT_CORPUS_PATH, build_probe_batch, RULE_POSITION_0
from src.eval.probes.probe_dtype import add_probe_dtype_argument, dtype_fields, load_kwargs
from src.eval.probes.artifact_paths import assert_no_account_paths, corpus_block, named_checkpoint

# Re-exported for older EXTERNAL callers only. Everything in this
# repository now imports these from `src.data.probe_corpus`, where they are
# defined -- they describe the probe corpus, not this driver. Do not import them
# from here in new code: two import paths for one constant is how a value ends up
# with two meanings.
from src.data.probe_corpus import PROBE_BATCH, PROBE_SEQ, WEIGHTS_FILENAME  # noqa: F401


class _OfflineTrajectoryCollector(NullCollector):
    """One-shot collector: turns a single `on_probe_step` call's `sink` into
    the two aggregated blobs (expert concentration, cross-loop overlap),
    then stops (this driver runs exactly one probe forward per
    checkpoint, never a training loop, so there is no "next call" to guard
    against state leaking between calls -- unlike a real training-time
    collector, which must be reusable across many steps)."""

    def __init__(self, *, input_ids_sha8: str) -> None:
        self._input_ids_sha8 = input_ids_sha8
        self.expert_concentration: dict[str, Any] | None = None
        self.cross_loop_overlap: dict[str, Any] | None = None

    def on_probe_step(
        self, step: int, sink: TraceSink, meta: TraceMeta, per_token_loss: Tensor
    ) -> dict[str, float]:
        n_experts, top_k = meta.num_experts, meta.num_active
        n_loops, n_layers = meta.num_stacks, meta.num_layers_in_stack

        expected_keys = {(r, l) for r in range(n_loops) for l in range(n_layers)}
        # The sink is keyed by `unrolled_pos` (head/tail layers
        # run outside the loop and both report (0, 0), so the old tuple key put
        # three rows on one entry). This driver is deprecated and keeps its
        # (loop_step, layer_idx) output grid; only the LOOKUP is adapted, and
        # non-loop positions are skipped because that grid cannot hold them.
        loop_at = {
            (p.loop_step, p.layer_idx): pos
            for pos, p in sink.items()
            if getattr(p, "block", "loop") == "loop"
        }
        if set(loop_at) != expected_keys:
            raise ValueError(
                f"trace sink covers {sorted(sink)}, expected the full "
                f"(loop_step, layer_idx) grid {sorted(expected_keys)} -- "
                "both metrics require the complete R x L grid from a single "
                "forward, not a partial sample."
            )

        # Depth-wise expert concentration: one `router_layer_
        # loop_record` row per (layer_idx, loop_step), reusing the function's
        # full output verbatim rather than trimming it to the two fields the
        # spec's illustrative schema names -- the richer dict (maxvio_count/
        # mass, margin_mean/min, hard_dead_count) is free from the same call
        # and is exactly the kind of thing a "no field ever silently dropped"
        # discipline argues for keeping.
        by_layer_loop = []
        for loop_step in range(n_loops):
            for layer_idx in range(n_layers):
                point = sink[loop_at[(loop_step, layer_idx)]]
                by_layer_loop.append(
                    router_layer_loop_record(
                        step=step,
                        layer_idx=layer_idx,
                        loop_step=loop_step,
                        router_logits=point.router_logits,
                        topk_idx=point.topk_idx,
                        topk_weights=point.topk_weights,
                        n_experts=n_experts,
                        top_k=top_k,
                    )
                )
        self.expert_concentration = {
            "kind": "expert_concentration",
            "input_ids_sha8": self._input_ids_sha8,
            "n_experts": n_experts,
            "num_layers_in_stack": n_layers,
            "num_stacks": n_loops,
            "by_layer_loop": by_layer_loop,
        }

        # Cross-loop-step overlap: one `overlap_record` call per
        # layer_idx, over that layer's router logits at every loop step from
        # THIS SAME forward pass (they must come from one forward).
        by_layer = []
        for layer_idx in range(n_layers):
            router_logits_by_loop = torch.stack(
                [sink[loop_at[(loop_step, layer_idx)]].router_logits for loop_step in range(n_loops)],
                dim=0,
            )  # (n_loops, batch, seq, n_experts)
            rec = overlap_record(
                step=step,
                layer_idx=layer_idx,
                router_logits_by_loop=router_logits_by_loop,
                top_k=top_k,
                input_ids_sha8=self._input_ids_sha8,
            )
            # Theoretical random-chance baseline (the expected overlap of two
            # independent random choices of k experts out of n_experts) -- a closed-form
            # constant, not something `_vendored_loopdiag.moe` computes, so it
            # is derived here rather than invented as part of the vendored
            # formula.
            rec["random_baseline"] = top_k / n_experts
            by_layer.append(rec)
        self.cross_loop_overlap = {
            "kind": "cross_loop_overlap",
            "input_ids_sha8": self._input_ids_sha8,
            "num_layers_in_stack": n_layers,
            "num_stacks": n_loops,
            "by_layer": by_layer,
        }
        return {}


def compute_trajectory_metrics(
    model: Any,
    *,
    tokenizer: Any,
    device: str,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    probe_seed: int = 0,
) -> dict[str, Any]:
    """Run the fixed probe corpus through `model` and return the two
    aggregated blobs plus the `token_source` provenance dict that produced
    `input_ids_sha8` (the three-point comparison is only valid if
    all three checkpoints' outputs carry the identical hash).

    `model` is expected to already be on `device` and loaded via
    `AutoModelForCausalLM.from_pretrained(checkpoint_dir, trust_remote_code=True,
    ...)` (see `main()` below, or `src/eval/run_eval.py`'s identical
    loading pattern) -- this function itself does no checkpoint I/O, so it
    can also be called directly from a test with an in-memory toy model.

    `probe_seed`: **0 (default) reproduces this function's
    original behavior exactly** -- forwarded to `build_probe_batch`'s
    `seed`, see that function's docstring for what changes for a nonzero
    value (a per-document random token-offset window, reproducible per
    seed, real text only). These two metrics have no other source of run-to-run
    variance (unlike the perturbation probe's `--repeats`), so this is what allows
    error bars across repeated measurements. The PRIMARY three-point
    comparison always uses `probe_seed=0` -- never pass a
    nonzero value there, only for the additional error-bar layer.
    """
    input_ids, token_source = build_probe_batch(
        tokenizer, corpus_path, batch=PROBE_BATCH, seq=PROBE_SEQ, seed=probe_seed, exclusion_rule=RULE_POSITION_0
    )
    input_ids = input_ids.to(device)
    collector = _OfflineTrajectoryCollector(input_ids_sha8=token_source["input_ids_sha8"])
    run_probe_forward(model, input_ids, collector=collector, step=0)
    assert collector.expert_concentration is not None
    assert collector.cross_loop_overlap is not None
    # probe_seed lands directly in both output blobs (not just token_source)
    # so a standalone JSONL row is self-describing on its own -- a reader of
    # one line from a batch of 24 (6 checkpoints x seeds 0-4 error-bar repeats)
    # should not have to cross-reference a separate token_source blob to know
    # which seed produced it.
    collector.expert_concentration["probe_seed"] = probe_seed
    collector.cross_loop_overlap["probe_seed"] = probe_seed
    return {
        "expert_concentration": collector.expert_concentration,
        "cross_loop_overlap": collector.cross_loop_overlap,
        "token_source": token_source,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True, help="Trajectory checkpoint directory (HF format)")
    ap.add_argument("--out", required=True, help="Output JSONL path -- one line per metric kind")
    ap.add_argument(
        "--tokenizer", required=True,
        help="Tokenizer repo id or local directory (required: trajectory checkpoints do "
             "not bundle tokenizer files; the bundled SmolLM2 tokenizer is "
             "src/data/artifacts/tokenizer/smollm2).",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_probe_dtype_argument(ap)
    ap.add_argument(
        "--probe-seed", dest="probe_seed", type=int, default=0,
        help="0 (default) = the primary, comparable probe input (byte-identical to this "
             "script's original behavior). Nonzero = an additional error-bar repeat with a "
             "different real-text token window per document (see build_probe_batch's "
             "docstring) -- never use a nonzero value for the primary three-point comparison "
             "itself (base checkpoint and two later training stages), only for repeated error-bar measurements on top of it.",
    )
    args = ap.parse_args(argv)

    from src.pretrain.checkpointing.store import MANIFEST_NAME, is_complete
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not is_complete(checkpoint_dir):
        raise ValueError(
            f"{checkpoint_dir} has no valid manifest.json -- refusing to probe an "
            "incomplete/unverified checkpoint (same check as src/eval/run_eval.py's "
            "_read_manifest)."
        )
    weights_path = checkpoint_dir / WEIGHTS_FILENAME
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise ValueError(f"{checkpoint_dir} has a manifest but {WEIGHTS_FILENAME} is missing/empty.")
    with open(checkpoint_dir / MANIFEST_NAME) as fh:
        manifest = json.load(fh)

    tokenizer_repo = args.tokenizer

    print(f"[offline_trajectory_probe] checkpoint: {checkpoint_dir} "
          f"(step={manifest.get('step')}, kind={manifest.get('kind')})")
    print(f"[offline_trajectory_probe] tokenizer: {tokenizer_repo}")
    print(f"[offline_trajectory_probe] probe corpus: batch={PROBE_BATCH} seq={PROBE_SEQ} "
          f"(fixed, not a flag) probe_seed={args.probe_seed}")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, trust_remote_code=True, **load_kwargs(args.probe_dtype)
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

    result = compute_trajectory_metrics(
        model, tokenizer=tokenizer, device=args.device, probe_seed=args.probe_seed
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for kind in ("expert_concentration", "cross_loop_overlap"):
            # Additive field only -- no existing field changes, so consumers
            # reading this schema keep working.
            # Recorded so precision never again has to be inferred from grid
            # alignment of the values themselves.
            row = {
                "corpus": corpus_block(result["token_source"]),
                **named_checkpoint(checkpoint_dir),
                **dtype_fields(model, args.probe_dtype),
                **result[kind],
            }
            assert_no_account_paths(row, what=f"{out_path}")
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[offline_trajectory_probe] wrote {out_path} (2 lines: expert_concentration, cross_loop_overlap)")


if __name__ == "__main__":
    main()

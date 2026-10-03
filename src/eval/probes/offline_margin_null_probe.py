"""Offline top-k margin against a per-checkpoint null baseline, plus the
p25/p50/p75 quantiles the source figures report as a ribbon.

## Why this script exists

`src.metrics.collector.margin_null_record` (metric #10) defines the
measurement; see its docstring for the null construction, why the centering
step cannot be skipped, and why published margin thresholds (0.23 / 0.98 /
1.45) are not usable on this project's models. What that function needs is
an `(n_tokens, n_experts)` matrix of raw router logits per
`(layer_idx, loop_step)`, which only exists during a forward pass.

The same two facts that motivated `offline_zloss_probe.py` apply verbatim:
the runs are already finished, and the FSDP2 production path never
populated the router rows of `train_metrics.jsonl` (see `collector.py`'s
module docstring), so there is no in-training router data to read back.
The scores must be produced by a fresh forward over the saved weights.

## Why a fourth driver instead of extending an existing one

`offline_trajectory_probe.py`'s output schema is fixed and consumed by
downstream analysis; adding fields would change a schema someone else reads.
`offline_perturbation_probe.py` set the precedent for the alternative --
a separate driver importing the first one's shared constants -- and
`offline_zloss_probe.py` followed it. This module does the same, importing
`PROBE_BATCH` / `PROBE_SEQ` / `WEIGHTS_FILENAME` rather than redefining
(and risking drift in) the probe-input definition that makes results
comparable across checkpoints.

This is also why `margin_null_record`'s numbers are emitted as their own
`kind="margin_null"` rows and are NOT added to `kind="router"` rows.

## Several probe seeds, concatenated

The pre-registered precondition for the ratio is >= 50,000 tokens per
`(layer_idx, loop_step)` cell, but one probe forward yields
`PROBE_BATCH * PROBE_SEQ = 40 * 512 = 20,480`. The probe batch is a
hardcoded constant precisely so it cannot be varied between invocations,
so enlarging it is not an option.

`DEFAULT_PROBE_SEEDS = (0, 1, 2, 3)` therefore runs three forwards and
concatenates their tokens per cell: 61,440 >= 50,000. `probe_seed` is the
already-documented error-bar mechanism (a different real-text window per
document), not a knob invented for this script, and the pre-registered
threshold stays untouched. The alternative -- lowering the threshold to
20,000, which the spec's noise table would support -- was rejected because
it edits a pre-registered number.

Each seed's forward is independent; only the flattened score matrices are
concatenated, and the concatenation is along the token axis only, so every
row of the result is one real token's real score vector.

## Tensor lifetime

`loop_trace.py`'s contract forbids retaining trace tensors past the
callback. This module stores a **CPU copy** (`.detach().float().cpu()`)
and keeps no reference to the trace's own storage, so
`assert_trace_released` still passes -- a test asserts exactly this. The
copy is unavoidable: the null needs all three seeds' tokens for one cell
at once, and the three forwards cannot be in flight simultaneously.
The null itself is then computed back on `--device`, because 200 column
shuffles per cell is seconds on a GPU and minutes on a CPU.

## Forward precision: a REQUIRED `--probe-dtype`, no default

Router scores are computed in whatever dtype the model is loaded in --
`Router.forward` does `logits = self.gate(x)` with no float32 cast, so
loading in bfloat16 makes the logits bf16 and every downstream number
inherits ~3 decimal digits of precision.

That is not acceptable for this measurement, for two reasons:

1. **This metric family reads a SIGN.** `frac_z_negative` counts tokens
   with `z_t < 0`; bf16 quantization around zero can flip that comparison
   outright, so the error is not a small bias on a magnitude, it is a
   wrong answer on the quantity of interest.
2. **Cross-model comparison is the point.** The adapter work will read
   open-source MoE models and compare against ours; the comparison line
   is fp32 everywhere, and our own models cannot be the exception.

Empirically confirmed on this project's own model: loading
with `torch_dtype=bfloat16` puts 100% of router logits exactly on the
bf16 grid; fp32 does not. The same signature appeared in the
existing trajectory-probe outputs (752/752 `margin_mean` values on the
bf16 grid) -- which is how this was caught at all, and why
`forward_dtype` is now recorded in every output row rather than left to
be reverse-engineered from the numbers.

This was hardcoded while a single precision was the only correct answer.
The criteria now run TWO precision lines -- float32 for every
judgement, each model's native precision for reporting -- so the caller has
to say which: `--probe-dtype` is REQUIRED, with no default (see
`probe_dtype.py`). The artifact records `probe_dtype`, `forward_dtype` (the
older name, same value) and `probe_dtype_requested`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from src.model.loop_trace import NullCollector, TraceMeta, TraceSink, run_probe_forward
from src.metrics.collector import (
    QUANTITIES,
    MetricNotApplicable,
    archive_document_bootstrap,
    margin_null_record,
)
from src.data.probe_corpus import PROBE_BATCH, PROBE_SEQ, WEIGHTS_FILENAME
from src.data.probe_corpus import DEFAULT_CORPUS_PATH, build_probe_batch, RULE_POSITION_0
from src.eval.probes.probe_dtype import add_probe_dtype_argument, dtype_fields, load_kwargs
from src.eval.probes.artifact_paths import assert_no_account_paths, corpus_block, named_checkpoint

DEFAULT_PROBE_SEEDS = (0, 1, 2, 3)
"""Seeds whose tokens are concatenated. 0 is the primary comparable input;
the rest are the documented error-bar repeats, used here for their tokens.

FOUR seeds, not three: different seeds draw different windows, but on a
short document those windows OVERLAP, so the raw count `n_seeds * batch *
seq` overstates how much distinct text was actually seen. Measured on this
corpus: three seeds give 49,646 distinct tokens (below the pre-registered
50,000), four give 61,930 of 81,920 raw. The precondition is checked
against the DEDUPLICATED count."""

DEFAULT_N_SHUFFLES = 200
"""Null draws per cell. Pre-registered in the spec (section 4.3)."""


class _OfflineMarginCollector(NullCollector):
    """Captures one probe forward's raw router logits as flat CPU copies,
    keyed by `(loop_step, layer_idx)`.

    One instance is used per forward and is not reused across steps, so it
    carries no between-call state to guard. It stores COPIES, never the
    trace's own tensors -- see the module docstring on tensor lifetime.
    """

    def __init__(self) -> None:
        self.scores: dict[tuple[int, int], Tensor] = {}
        self.meta: TraceMeta | None = None

    def on_probe_step(
        self, step: int, sink: TraceSink, meta: TraceMeta, per_token_loss: Tensor
    ) -> dict[str, float]:
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
                f"(loop_step, layer_idx) grid {sorted(expected_keys)}. This "
                "measurement is reported per (layer_idx, loop_step); a partial "
                "grid would silently produce a table with missing cells that "
                "looks like a complete one."
            )

        self.meta = meta
        # Keyed back to (loop_step, layer_idx) because this driver's downstream
        # grid is still that pair; non-loop positions are skipped, since that
        # grid cannot represent a layer outside the loop.
        for key, pos in loop_at.items():
            point = sink[pos]
            logits = point.router_logits
            if logits.ndim != 3:
                raise ValueError(
                    f"router_logits at {key} has shape {tuple(logits.shape)}; "
                    "expected (batch, seq, n_experts)."
                )
            # Flatten (batch, seq) into one token axis, then copy off the
            # trace's storage. `.cpu()` on a CUDA tensor already copies;
            # `.clone()` covers the CPU-device case, where it would not.
            self.scores[key] = logits.detach().float().reshape(-1, logits.shape[-1]).cpu().clone()
        return {}


def _one_value(token_sources: list[dict[str, Any]], key: str) -> Any:
    """The value of `key`, required to agree across every seed's metadata.

    Disagreement means the seeds were built against different corpus
    definitions, which would make concatenating their tokens meaningless --
    and would do so silently, since the concatenation itself still works.
    """
    values = {t[key] for t in token_sources}
    if len(values) != 1:
        raise ValueError(
            f"probe seeds disagree on {key}: {sorted(values)}. The seeds were "
            "built against different corpus definitions; concatenating their "
            "tokens would mix two corpora into one reading."
        )
    return values.pop()


def _effective_token_count(token_sources: list[dict[str, Any]]) -> int:
    """Distinct tokens across seeds, deduplicating overlapping windows.

    Each seed draws a `[start, start+seq)` window per document. On a short
    document, different seeds' windows overlap, so summing `n_seeds * batch *
    seq` counts the same text several times and reports a sample size that
    does not exist. This unions the position intervals per document instead.

    Documents are identified by name (falling back to position) because the
    same document appears once per seed and the union must be taken per
    document, not per (document, seed).
    """
    seen: dict[str, set[int]] = {}
    for source in token_sources:
        for position, doc in enumerate(source["docs"]):
            key = str(doc.get("name", position))
            # Both are recorded POST-drop by the corpus layer, so this counts
            # exactly the positions that take part in a reduction.
            start = int(doc["window_start"])
            seen.setdefault(key, set()).update(range(start, start + int(doc["window_len"])))
    return sum(len(v) for v in seen.values())


def compute_margin_null(
    model: Any,
    *,
    tokenizer: Any,
    device: str,
    probe_dtype: str,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    probe_seeds: tuple[int, ...] = DEFAULT_PROBE_SEEDS,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    n_bootstrap: int = 200,
    n_shuffles_bootstrap: int = 15,
    shuffle_seed: int = 0,
    min_tokens: int,
    n_archive_bootstrap: int = 0,
    archive_quantities: Sequence[str] = ("within_selection",),
) -> dict[str, Any]:
    """Run the fixed probe corpus through `model` once per seed in
    `probe_seeds`, concatenate the per-cell score matrices, and return one
    `margin_null` row per `(layer_idx, loop_step)` plus provenance.

    `model` is expected to already be on `device`. This function does no
    checkpoint I/O, so a test can call it with an in-memory toy model.
    """
    if not probe_seeds:
        raise ValueError("probe_seeds is empty -- there would be no tokens to measure.")
    if len(set(probe_seeds)) != len(probe_seeds):
        raise ValueError(
            f"probe_seeds={probe_seeds} repeats a seed. Repeats would concatenate "
            "the SAME tokens twice, inflating n_tokens past the precondition "
            "without adding any information -- the exact failure the token "
            "threshold exists to prevent."
        )

    per_seed_scores: list[dict[tuple[int, int], Tensor]] = []
    token_sources: list[dict[str, Any]] = []
    meta: TraceMeta | None = None

    for seed in probe_seeds:
        input_ids, token_source = build_probe_batch(
            tokenizer, corpus_path, batch=PROBE_BATCH, seq=PROBE_SEQ, seed=seed, exclusion_rule=RULE_POSITION_0
        )
        collector = _OfflineMarginCollector()
        run_probe_forward(model, input_ids.to(device), collector=collector, step=0)
        assert collector.meta is not None
        if meta is not None and set(collector.scores) != set(per_seed_scores[0]):
            raise ValueError(
                f"probe_seed={seed} produced grid {sorted(collector.scores)} but "
                f"seed {probe_seeds[0]} produced {sorted(per_seed_scores[0])}. "
                "Concatenating mismatched grids would mix cells."
            )
        meta = collector.meta
        per_seed_scores.append(collector.scores)
        token_sources.append({"probe_seed": seed, **token_source})

    assert meta is not None
    # Position 0 of every window is dropped below, so the window each document
    # contributes is [start+1, start+seq) -- counting from `start` would report
    # tokens that no longer take part in any reduction.
    # Geometry comes from the corpus metadata, not from PROBE_SEQ: the corpus
    # layer drops position 0 of every window, so a driver that assumed the
    # nominal length would reshape 511-token documents as if they were 512 and
    # silently mix neighbouring documents' tokens into one block.
    tokens_per_doc = _one_value(token_sources, "tokens_per_doc")
    # R5 self-assertion: every document in every seed batch must have the same
    # slot count. Equal length is what makes "mean over all tokens" identical
    # to "each document weighted 1/40" and what lets the position bins be one
    # set of cut points; a ragged batch would break both QUIETLY, since the
    # reductions still run and still produce plausible numbers.
    for source in token_sources:
        slots = source["slots_per_document"]
        if set(slots) != {tokens_per_doc}:
            raise ValueError(
                f"probe seed {source.get('seed')} has documents of differing length "
                f"({sorted(set(slots))} against tokens_per_doc={tokens_per_doc}). "
                "Equal-length documents are assumed by the per-document weighting, "
                "the document-scoped shuffle and the position bins alike."
            )
    # The resampling unit for the archive bootstrap. Every seed reads the SAME
    # documents and varies only the window, so slots = documents x seeds and
    # only the document count is a sample size. Checked here rather than
    # assumed: the grouping below relies on document `d` meaning the same text
    # in every seed, which is true only if the order matches too.
    doc_names_per_seed = [tuple(d["name"] for d in t["docs"]) for t in token_sources]
    if len(set(doc_names_per_seed)) != 1:
        raise ValueError(
            "probe seeds do not read the same documents in the same order "
            f"({len(set(doc_names_per_seed))} distinct orderings). The archive "
            "bootstrap groups slot s*n_docs+d as document d, which is only "
            "meaningful when every seed's d-th document is the same text."
        )
    n_distinct_documents = len(set(doc_names_per_seed[0]))
    if n_distinct_documents != len(doc_names_per_seed[0]):
        raise ValueError(
            f"a probe seed reads {len(doc_names_per_seed[0])} documents but only "
            f"{n_distinct_documents} are distinct; a repeated document would be "
            "resampled as if it were two."
        )

    exclusion = {
        "first_token_dropped": _one_value(token_sources, "first_token_dropped"),
        "special_tokens_excluded": sum(int(t["special_tokens_excluded"]) for t in token_sources),
        "tokens_per_doc": tokens_per_doc,
        "slots_per_document": token_sources[0]["slots_per_document"],
    }
    n_tokens_effective = _effective_token_count(token_sources)

    rows = []
    archive_cells: list[dict[str, Any]] = []
    for loop_step in range(meta.num_stacks):
        for layer_idx in range(meta.num_layers_in_stack):
            key = (loop_step, layer_idx)
            pooled = torch.cat([s[key] for s in per_seed_scores], dim=0).to(device)
            # Kept for the archive-level bootstrap below, which resamples
            # documents ONCE and applies that one draw to every grid point.
            # Centered exactly as `margin_null_record` centers -- per token,
            # before anything else -- so the archive statistic is built from
            # the same numbers as the per-cell rows and not a second,
            # separately-derived version of them.
            pooled_f = pooled.float()
            archive_cells.append(
                {
                    "centered": pooled_f - pooled_f.mean(dim=-1, keepdim=True),
                    "layer_idx": layer_idx,
                    "loop_step": loop_step,
                }
            )

            # BOTH shuffle scopes, side by side. They encode different null
            # hypotheses and disagree for a router that specializes by domain;
            # picking one here would be choosing the answer. See
            # `collector._column_shuffle`.
            # Domains come from the probe corpus itself (each document carries
            # one). Seeds are concatenated, so the document blocks repeat the
            # 40-document list once per seed, in the same order.
            missing = [
                d.get("name", f"#{i}")
                for source in token_sources
                for i, d in enumerate(source["docs"])
                if not d.get("domain")
            ]
            if missing:
                # Was `d.get("domain", "unknown")`. That silently relabels a
                # corpus change as a domain: the grouped readout would show a
                # plausible "unknown" bucket and nothing would report that the
                # probe corpus had stopped carrying domain labels.
                raise ValueError(
                    f"{len(missing)} probe document(s) carry no `domain` label "
                    f"(first few: {missing[:3]}). The grouped diagnostic exists to "
                    "check whether a difference is an artefact of which text was "
                    "probed, which it cannot do with unlabelled documents."
                )
            doc_domains = [d["domain"] for source in token_sources for d in source["docs"]]
            for scope in ("batch", "document"):
                rows.append(
                    margin_null_record(
                        step=0,
                        layer_idx=layer_idx,
                        loop_step=loop_step,
                        router_logits=pooled,
                        top_k=meta.num_active,
                        tokens_per_doc=tokens_per_doc,
                        shuffle_scope=scope,
                        n_shuffles=n_shuffles,
                        n_bootstrap=n_bootstrap,
                        n_shuffles_bootstrap=n_shuffles_bootstrap,
                        shuffle_seed=shuffle_seed,
                        min_tokens=min_tokens,
                        n_tokens_effective=n_tokens_effective,
                        doc_domains=doc_domains,
                    )
                )

    # --- archive-level document bootstrap ---------------------------------
    # The difference threshold is three times the largest DOCUMENT-level bootstrap deviation of an
    # ARCHIVE-level statistic (the mean over this checkpoint's grid points, and
    # its worst point) -- not of a single grid point. It therefore cannot be
    # assembled from the per-cell `rho_cal_boot_*` fields above, which are
    # separate draws per cell: combining those discards the correlation between
    # points and yields a deviation that is too small, hence a threshold that
    # is too loose. `archive_document_bootstrap` owns that reasoning and the
    # assertion that enforces it.
    archive_bootstrap: list[dict[str, Any]] = []
    if n_archive_bootstrap > 0:
        for quantity_name in archive_quantities:
            for scope in ("batch", "document"):
                block: dict[str, Any] = {
                    "quantity": quantity_name,
                    "shuffle_scope": scope,
                }
                try:
                    block.update(
                        archive_document_bootstrap(
                            archive_cells,
                            top_k=meta.num_active,
                            n_experts=meta.num_experts,
                            n_bootstrap=n_archive_bootstrap,
                            n_shuffles_bootstrap=n_shuffles_bootstrap,
                            tokens_per_doc=tokens_per_doc,
                            n_distinct_documents=n_distinct_documents,
                            shuffle_scope=scope,
                            shuffle_seed=shuffle_seed,
                            quantity_name=quantity_name,
                        )
                    )
                    block["status"] = "ok"
                except MetricNotApplicable as exc:
                    # `selection_separation` is undefined when k == E: there is
                    # no (k+1)-th expert to separate from. Recorded as a stated
                    # status rather than dropped, so a consumer sees "this
                    # configuration has no such quantity" instead of silently
                    # finding one fewer block than it expected.
                    block["status"] = "not_applicable"
                    block["reason"] = str(exc)
                archive_bootstrap.append(block)

    # Code provenance in the form the whole project uses:
    # `producer.code_revision`, structured exactly as
    # `provenance.code_revision()` returns it, with no renaming. `dirty` and
    # `diff_sha256` matter because a commit alone does not say whether the
    # running tree matched it.
    # Provenance comes from the one function that owns this decision.
    # `code_revision()` uses git when available and otherwise reads the
    # value handed over via CODE_REVISION_JSON, validating it and raising if
    # neither source exists -- the fallback used to live here, and in the
    # adapter's driver, and was about to be copied into posttrain's, which is
    # exactly how "just write a placeholder" gets introduced somewhere.
    from src.metrics.provenance import code_revision

    revision = code_revision()

    return {
        "producer": {
            "name": "offline_margin_null_probe",
            "code_revision": revision,
        },
        "margin_null": {
            "kind": "margin_null_probe",
            "num_layers_in_stack": meta.num_layers_in_stack,
            "num_stacks": meta.num_stacks,
            "num_experts": meta.num_experts,
            "num_active": meta.num_active,
            # Recorded, never inferred: the existing trajectory probe did not
            # record it, and its precision had to be reverse-engineered from
            # grid alignment of the values themselves. See the module docstring.
            **dtype_fields(model, probe_dtype),
            "probe_seeds": list(probe_seeds),
            "n_shuffles": n_shuffles,
            "n_bootstrap": n_bootstrap,
            "n_shuffles_bootstrap": n_shuffles_bootstrap,
            "shuffle_seed": shuffle_seed,
            "min_tokens": min_tokens,
            # Always recorded, including the 0 that means "not run": a reader
            # who finds no archive block must be able to tell "switched off"
            # from "the run died before writing it".
            "n_archive_bootstrap": int(n_archive_bootstrap),
            # Recorded next to the slot counts on purpose: `n_tokens_nominal`
            # and the per-cell `n_tokens` scale with seeds, and only this one
            # is a sample size.
            "n_distinct_documents": n_distinct_documents,
            "archive_quantities": list(archive_quantities),
            "n_tokens_raw": PROBE_BATCH * PROBE_SEQ * len(probe_seeds),
            # The "nominal" count is the rows actually REDUCED OVER
            # before de-duplication -- which, since the corpus layer drops
            # position 0 of every window, is 511 per document, not 512. It is
            # therefore NOT the same number as `n_tokens_raw` any more: that
            # one is the pre-drop count, and the two differ by one token per
            # document.
            "n_tokens_nominal": PROBE_BATCH * tokens_per_doc * len(probe_seeds),
            "n_tokens_effective": n_tokens_effective,
            **exclusion,
            "shuffle_scopes": ["batch", "document"],
            "by_layer_loop": rows,
        },
        "archive_bootstrap": archive_bootstrap,
        "token_sources": token_sources,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory (HF format)")
    ap.add_argument("--out", required=True, help="Output JSONL path (one line)")
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer repo id or local directory (required: trajectory checkpoints do "
             "not bundle tokenizer files; the bundled SmolLM2 tokenizer is "
             "src/data/artifacts/tokenizer/smollm2).",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_probe_dtype_argument(ap)
    ap.add_argument(
        "--min-tokens", dest="min_tokens", type=int, required=True,
        help="Pre-registered minimum effective tokens per cell. REQUIRED and "
        "with no default: the criteria file carries this number, and a default "
        "here would be a second copy of it maintained independently.",
    )
    ap.add_argument(
        "--probe-seeds",
        dest="probe_seeds",
        required=True,
        help="Comma-separated probe seeds to concatenate, e.g. '0,1,2,3' (criteria "
        "group A) or '0' (group F). REQUIRED and with no default: the seed set "
        "determines how much distinct text the reading covers, so two runs of the "
        "same command line must not be able to differ on it silently.",
    )
    ap.add_argument(
        "--n-shuffles-bootstrap",
        dest="n_shuffles_bootstrap",
        type=int,
        required=True,
        help="Null shuffles drawn INSIDE each bootstrap draw. REQUIRED and with "
        "no default: it is not the same quantity as the criteria file's "
        "min_n_shuffles (which governs the per-cell null), the two names look "
        "alike, and one reader has already confused them. Recorded in the "
        "artifact, because two sides agreeing today cannot be shown later from "
        "an artifact that does not carry the number.",
    )
    ap.add_argument(
        "--n-archive-bootstrap",
        dest="n_archive_bootstrap",
        type=int,
        required=True,
        help="Draws for the ARCHIVE-level document bootstrap (the difference threshold "
        "needs >= 1000; 0 switches it off). REQUIRED and with no default: this "
        "is what separates a run that can lock the threshold from one that "
        "cannot, and the two runs would otherwise share a command line.",
    )
    ap.add_argument(
        "--archive-quantities",
        dest="archive_quantities",
        required=True,
        help="Comma-separated quantities to bootstrap at archive level: "
        "'within_selection', 'selection_separation', or both. REQUIRED: the "
        "judgement quantity is the larger of the two rho_cal, so which ones "
        "were bootstrapped decides what the resulting threshold applies to.",
    )
    ap.add_argument(
        "--shuffle-seed",
        dest="shuffle_seed",
        type=int,
        default=0,
        help="Seed for the null column shuffles. Recorded in every row so the "
        "null is reproducible bit for bit.",
    )
    args = ap.parse_args(argv)

    from src.pretrain.checkpointing.store import MANIFEST_NAME, is_complete
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not is_complete(checkpoint_dir):
        raise ValueError(
            f"{checkpoint_dir} has no valid manifest.json -- refusing to probe an "
            "incomplete/unverified checkpoint (same check as offline_trajectory_probe.py)."
        )
    weights_path = checkpoint_dir / WEIGHTS_FILENAME
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise ValueError(
            f"{checkpoint_dir} has a manifest but {WEIGHTS_FILENAME} is missing/empty."
        )
    with open(checkpoint_dir / MANIFEST_NAME) as fh:
        manifest = json.load(fh)

    tokenizer_repo = args.tokenizer

    print(
        f"[offline_margin_null_probe] checkpoint: {checkpoint_dir} "
        f"(step={manifest.get('step')}, kind={manifest.get('kind')})"
    )
    print(f"[offline_margin_null_probe] tokenizer: {tokenizer_repo}")
    seeds = tuple(int(x) for x in args.probe_seeds.split(",") if x.strip())
    quantities = tuple(x.strip() for x in args.archive_quantities.split(",") if x.strip())
    unknown = [q for q in quantities if q not in QUANTITIES]
    if unknown:
        raise ValueError(
            f"--archive-quantities names {unknown}, which is not among "
            f"{sorted(QUANTITIES)}. Checked here rather than inside the loop so "
            "a typo fails before the model is loaded, not after the forward pass."
        )
    print(
        f"[offline_margin_null_probe] probe corpus: batch={PROBE_BATCH} seq={PROBE_SEQ} "
        # The seeds ACTUALLY requested, not the module default: the run above
        # printed "probe_seeds=(0, 1, 2, 3) -> 81920 tokens/cell" for a run
        # that used seed 0 and 20,440 tokens. A log that disagrees with the
        # run is worse than no log -- it is what someone checks first.
        f"probe_seeds={seeds} -> {PROBE_BATCH * (PROBE_SEQ - 1) * len(seeds)} tokens/cell"
    )
    print(
        f"[offline_margin_null_probe] archive bootstrap: n={args.n_archive_bootstrap} "
        f"n_shuffles_bootstrap={args.n_shuffles_bootstrap} "
        f"quantities={quantities} scopes=('batch', 'document')"
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, trust_remote_code=True, **load_kwargs(args.probe_dtype)
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

    result = compute_margin_null(
        model,
        tokenizer=tokenizer,
        device=args.device,
        probe_dtype=args.probe_dtype,
        probe_seeds=seeds,
        min_tokens=args.min_tokens,
        shuffle_seed=args.shuffle_seed,
        n_archive_bootstrap=args.n_archive_bootstrap,
        n_shuffles_bootstrap=args.n_shuffles_bootstrap,
        archive_quantities=quantities,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "corpus": corpus_block(result["token_sources"]),
        **named_checkpoint(checkpoint_dir),
        # Carried from the checkpoint's own manifest so a row is readable
        # without re-opening the checkpoint. Only fields `MANIFEST_TIER1`
        # actually guarantees are read here -- `run_name` is deliberately
        # not among them (it is not a manifest field; `.get` would have
        # written a silent null into every row). Run identity comes from
        # the output path, which the submitting script controls.
        "manifest_step": manifest["step"],
        "manifest_kind": manifest["kind"],
        **result,
    }
    # Checked before the file is opened, so a refusal leaves the previous
    # artifact untouched rather than half-written.
    assert_no_account_paths(row, what=f"{out_path}")
    with open(out_path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"[offline_margin_null_probe] wrote {out_path}")


if __name__ == "__main__":
    main()

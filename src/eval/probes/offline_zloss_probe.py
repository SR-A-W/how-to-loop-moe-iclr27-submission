"""Offline router z-loss state sensor: `mean(z_t)` and `Pr[z_t<0]` computed
post-hoc against an already-saved checkpoint.

## Why this script exists

`src.metrics.collector.zloss_state_record` (metric #9) defines the two
scalars; see its docstring for the derivation, the reduction order, and why
the sign of `z_t` -- not its magnitude -- is the quantity of interest. What
that function needs is a `(batch, seq, n_experts)` tensor of raw router
logits per `(layer_idx, loop_step)`, which only exists during a forward
pass.

Two facts make an offline driver the right shape for this measurement
rather than a training-loop hook:

1. **The runs are already finished.** The question being answered ("did the
   z-loss mechanism documented at E=512 ever engage in OUR E in {8,16,64}
   runs?") is about checkpoints that already exist. Waiting for the next
   training run would answer it months later.
2. **The production FSDP2 training path never populated the router rows.**
   `collector.py`'s module docstring records this known gap: `on_trace_step`
   was never fed a `sink` in the production path, so `train_metrics.jsonl`
   carries only `loss` rows. There is no in-training z-state data to read
   back even for the runs that have finished; it must be produced by a
   fresh forward pass over the saved weights.

This module is a **driver only** -- it does not define the metric. It
mirrors `offline_trajectory_probe.py` and `offline_perturbation_probe.py`
step for step (load checkpoint -> build the same fixed probe batch -> one
`run_probe_forward` with a `TraceSink` attached -> feed the existing record
function), reusing `run_probe_forward` verbatim rather than reimplementing
its isolation guarantees (no autograd graph, RNG restored, model mode
restored even on exception).

## Why a third driver instead of extending `offline_trajectory_probe.py`

That script's output schema is fixed and consumed by downstream analysis
(the three-point comparison of expert concentration and cross-loop overlap).
Adding fields to its rows would change a schema someone else already reads.
`offline_perturbation_probe.py` set the precedent for the alternative --
a separate driver importing this one's shared constants -- and this module
follows it, importing `PROBE_BATCH` / `PROBE_SEQ` / `WEIGHTS_FILENAME` from
`offline_trajectory_probe` rather than redefining (and risking drift in)
the probe-input definition that makes results comparable across scripts.

## Probe corpus: hardcoded, not a CLI knob

Same reason as `offline_trajectory_probe.py`: comparability across
checkpoints is the entire point of this measurement (the deliverable is a
curve over training steps and a comparison across three architectures), and
a flag a caller could vary between invocations would silently break it.
`probe_seed` remains a flag because it is the documented error-bar
mechanism, and `probe_seed=0` is the primary, comparable input.

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

.. deprecated::
   Superseded by `offline_cross_arch_probe.py` for new work, and not adapted to
   the head/tail architecture: its position grid is still (loop_step, layer_idx),
   which cannot represent a layer outside the loop. Existing artifacts remain
   valid for the runs they describe.

   NOT removable: `src.posttrain.guardrail_eval` calls `compute_zloss_state`.
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
from src.metrics.collector import zloss_state_record
from src.data.probe_corpus import PROBE_BATCH, PROBE_SEQ, WEIGHTS_FILENAME
from src.data.probe_corpus import DEFAULT_CORPUS_PATH, build_probe_batch, RULE_POSITION_0
from src.eval.probes.probe_dtype import add_probe_dtype_argument, dtype_fields, load_kwargs
from src.eval.probes.artifact_paths import assert_no_account_paths, corpus_block, named_checkpoint


class _OfflineZLossCollector(NullCollector):
    """One-shot collector: turns a single `on_probe_step` call's `sink` into
    one `zloss_state` row per `(layer_idx, loop_step)`.

    Like `_OfflineTrajectoryCollector`, this runs exactly one probe forward
    per checkpoint and is not reused across steps, so it carries no
    between-call state to guard.
    """

    def __init__(self, *, input_ids_sha8: str) -> None:
        self._input_ids_sha8 = input_ids_sha8
        self.zloss_state: dict[str, Any] | None = None

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
                "grid would silently produce a curve with missing points that "
                "looks like a complete one."
            )

        by_layer_loop = [
            zloss_state_record(
                step=step,
                layer_idx=layer_idx,
                loop_step=loop_step,
                router_logits=sink[loop_at[(loop_step, layer_idx)]].router_logits,
            )
            for loop_step in range(n_loops)
            for layer_idx in range(n_layers)
        ]

        # Depth-averaged mirror of what the model's own `z_loss` scalar
        # reduces to, so a reader can compare this row against a
        # training-time `loss_record` without recomputing anything. The
        # model averages its per-layer `mean(z_t**2)` over the unrolled
        # depth (`modeling_loop_lm.LoopMoEModel.forward` divides by
        # `total_layers`), so the matching reduction here is the unweighted
        # mean over the R x L grid -- every (loop_step, layer_idx) cell saw
        # the same number of tokens, so an unweighted mean is also the
        # token-weighted one.
        z_sq_mean_depth_avg = sum(r["z_sq_mean"] for r in by_layer_loop) / len(by_layer_loop)

        self.zloss_state = {
            "kind": "zloss_state_probe",
            "input_ids_sha8": self._input_ids_sha8,
            "num_layers_in_stack": n_layers,
            "num_stacks": n_loops,
            "num_experts": meta.num_experts,
            "num_active": meta.num_active,
            "z_sq_mean_depth_avg": z_sq_mean_depth_avg,
            "by_layer_loop": by_layer_loop,
        }
        return {}


def compute_zloss_state(
    model: Any,
    *,
    tokenizer: Any,
    device: str,
    probe_dtype: str,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    probe_seed: int = 0,
) -> dict[str, Any]:
    """Run the fixed probe corpus through `model` and return the
    `zloss_state` blob plus the `token_source` provenance dict.

    `model` is expected to already be on `device`. This function does no
    checkpoint I/O, so a test can call it with an in-memory toy model.

    `probe_seed=0` (default) is the primary, comparable probe input; a
    nonzero value selects a different real-text token window per document
    (see `build_probe_batch`) and exists only for error-bar repeats.
    """
    input_ids, token_source = build_probe_batch(
        tokenizer, corpus_path, batch=PROBE_BATCH, seq=PROBE_SEQ, seed=probe_seed, exclusion_rule=RULE_POSITION_0
    )
    input_ids = input_ids.to(device)
    collector = _OfflineZLossCollector(input_ids_sha8=token_source["input_ids_sha8"])
    run_probe_forward(model, input_ids, collector=collector, step=0)
    assert collector.zloss_state is not None
    collector.zloss_state["probe_seed"] = probe_seed
    # Recorded, never inferred -- see the module docstring on why.
    collector.zloss_state.update(dtype_fields(model, probe_dtype))
    return {"zloss_state": collector.zloss_state, "token_source": token_source}


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
        "--probe-seed",
        dest="probe_seed",
        type=int,
        default=0,
        help="0 (default) = the primary, comparable probe input. Nonzero = an "
        "error-bar repeat with a different real-text window per document.",
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
        f"[offline_zloss_probe] checkpoint: {checkpoint_dir} "
        f"(step={manifest.get('step')}, kind={manifest.get('kind')})"
    )
    print(f"[offline_zloss_probe] tokenizer: {tokenizer_repo}")
    print(
        f"[offline_zloss_probe] probe corpus: batch={PROBE_BATCH} seq={PROBE_SEQ} "
        f"(fixed, not a flag) probe_seed={args.probe_seed}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, trust_remote_code=True, **load_kwargs(args.probe_dtype)
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

    result = compute_zloss_state(
        model, tokenizer=tokenizer, device=args.device, probe_seed=args.probe_seed,
        probe_dtype=args.probe_dtype
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "corpus": corpus_block(result["token_source"]),
        **named_checkpoint(checkpoint_dir),
        # Carried from the checkpoint's own manifest so a row is readable
        # without re-opening the checkpoint (which may since have been
        # deleted to stay inside the shared disk quota).
        # Only fields `MANIFEST_TIER1` actually guarantees are read here.
        # `run_name` is deliberately NOT among them: it is not a manifest
        # field, and `manifest.get("run_name")` would have written a silent
        # `null` into every row. The run identity comes from the output
        # path the caller chooses.
        "checkpoint_step": manifest["step"],
        "checkpoint_kind": manifest["kind"],
        "checkpoint_tokens_consumed": manifest["tokens_consumed"],
        "checkpoint_git_commit": manifest["git_commit"],
        **result["zloss_state"],
        "token_source": result["token_source"],
    }
    # Checked BEFORE the file is opened: a guard that fires after `open("w")`
    # has already truncated the previous artifact, turning a refusal to write
    # a bad row into the loss of a good one.
    assert_no_account_paths(row, what=f"{out_path}")
    with open(out_path, "w") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        f"[offline_zloss_probe] wrote {out_path} "
        f"({len(row['by_layer_loop'])} (layer_idx, loop_step) cells)"
    )


if __name__ == "__main__":
    main()

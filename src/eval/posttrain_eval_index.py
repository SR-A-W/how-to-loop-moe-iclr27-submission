"""Post-training model evaluation, minimal runnable version.

## What this is -- and why it is small

A post-training (SFT) checkpoint written by `src/posttrain/sft_probe.py` (or
`sft_formal.py`) is saved via the SAME `src.pretrain.checkpointing.store.
save_trajectory` call a pretraining trajectory checkpoint uses -- same HF
layout, same `manifest.json` contract, same `is_complete()` check. That one
fact is what keeps this module small:

1. **The collapse-metrics half needs zero new code.** `src.posttrain.
   guardrail_eval.build_guardrail_row()` already loads any HF-trust_remote_
   code-loadable LoopMoE checkpoint and computes z-loss/margin-null/MaxVio
   against it -- it does not know or care whether the checkpoint came from
   pretraining or post-training. This module calls it UNCHANGED against
   the SFT checkpoint directory.
2. **The benchmark-results half already exists.** `src/posttrain/sft_probe.py`
   writes `eval_results.json` (lm-eval benchmark scores) and
   `instruction_probe_results.json` (instruction-following probes) into
   the SAME checkpoint directory as the weights -- `out_dir` is both the
   checkpoint and the results directory (see that module's `main()`).

What this module adds is the one thing that did not exist: a small index
row that names, for one SFT checkpoint, where all three pieces are (named
root + relative path, per the project-wide no-absolute-paths rule), so a
reader does not have to know this directory-reuse fact to find
them. It does not redefine, re-copy, or re-validate any of the three
files' own internal fields.

## Field list

    {
      "schema_version": "1.0",
      "kind": "posttrain_eval_index",
      "run_name": ...,           # the SFT run's own identifier, e.g. "sft_dvf_a_1p3b-lr2e-5"
      "run_kind": "posttrain",   # the dashboard contract's "another kind of run" declaration
      "base_run_name": ...,      # MUST hit src/pretrain/recipe.py's RUN_LIST -- the pretraining key
      "measured_at": ...,
      "checkpoint": {"root": ..., "relpath": ..., "step": ...},
      "producer": {"name": ..., "version": ..., "code_revision": {...}},
      "artifacts": {
        "eval_results": {"root": ..., "relpath": ...} | null,
        "eval_results_status": "ok" | "not_run",
        "instruction_probe_results": {"root": ..., "relpath": ...} | null,
        "instruction_probe_results_status": "ok" | "not_run",
        "guardrail_eval": {"root": ..., "relpath": ...},  # ALWAYS present, no *_status --
                                                            # see note below
      }
    }

`guardrail_eval` has no `*_status` sibling, unlike the other two: it is
**mandatory**, not optional. `main()` refuses (raises) before assembling
anything if that file does not exist yet (this module never computes
collapse metrics itself -- see `main()`'s own check). A row without a
`guardrail_eval` pointer therefore never gets written at all, so there is
no "not_run" state for a reader to represent -- do not add a
`guardrail_eval_status` field or a null-handling branch for it; that
branch would never execute.

Three design points:

1. **`kind` renamed `posttrain_eval_report` -> `posttrain_eval_index`**:
   this row only ever holds pointers, never a copied-in number -- a copied
   score would exist in two places (this index and eval_results.json)
   with no way to tell which one is authoritative if they ever disagree.
   `_index` names that constraint; `_report` did not.
2. **`run_name` alone does not identify an SFT run.** `RUN_LIST` only has
   pretraining run names; an SFT checkpoint's own name (e.g.
   `sft_dvf_a_1p3b-lr2e-5`) is never in it, and the same base checkpoint
   can fan out into several SFT runs (different LR, different recipe).
   `base_run_name` (required, validated against `RUN_LIST`) is the actual
   pretrain-side join key; `run_name` stays the SFT run's own identifier;
   `run_kind: "posttrain"` declares which regime this row is (the
   dashboard contract requires declaring `run_kind` whenever `run_name`
   does not resolve against `RUN_LIST` -- same rule the open-source-model
   adapter rows use).
3. **A skipped eval stage gets an explicit status, not just a `null`.**
   `null` alone cannot distinguish "this stage was not run" from "the
   field was forgotten" -- `eval_results_status`/`instruction_probe_results_status`
   say which.

This module is NOT merged into the dashboard's `index.json` (the page's own
dataset directory) -- different subject (one row per SFT run here vs. one
global file there), different lifecycle (this is regenerated every run,
`index.json` changes when the page's panel set changes). `index.json`
registers `posttrain_eval_index` as one `kind`, once; this module never
touches that file.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.posttrain.guardrail_eval import (
    RUNS_ROOT_NAME,
    _assert_no_leaked_absolute_paths,
    _code_revision,
    _named_relpath,
)

SCHEMA_VERSION = "1.0"
KIND = "posttrain_eval_index"
PRODUCER_NAME = "posttrain_eval_report"
PRODUCER_VERSION = "0.1.0"
RUN_KIND = "posttrain"

EVAL_RESULTS_NAME = "eval_results.json"
INSTRUCTION_PROBE_RESULTS_NAME = "instruction_probe_results.json"


def _optional_artifact(path: Path, *, runs_root: Path) -> tuple[dict[str, str] | None, str]:
    """`({"root", "relpath"}, "ok")` if `path` exists, else `(None,
    "not_run")` -- never a broken pointer, and never a bare `null` with no
    way to tell "skipped" from "forgotten".
    A MISSING runs_root relationship (path exists but is not under
    runs_root) still raises via `_named_relpath` -- that would silently
    produce a wrong pointer, which is a different failure than "not run"."""
    if not path.is_file():
        return None, "not_run"
    return _named_relpath(path, root=runs_root, root_name=RUNS_ROOT_NAME), "ok"


def build_posttrain_eval_report(
    *, run_name: str, base_run_name: str, checkpoint_dir: Path, runs_root: Path,
    manifest: dict[str, Any], guardrail_eval_path: Path,
) -> dict[str, Any]:
    """Assemble one `posttrain_eval_index` row. Does no checkpoint I/O of
    its own beyond checking which sibling result files exist -- the
    guardrail_eval row itself must already have been produced by a
    separate call to `guardrail_eval.main()`/`build_guardrail_row()`
    against the same `checkpoint_dir` (this module does not invoke that
    itself, so a caller can choose to skip re-computing an already-fresh
    collapse-metrics row).

    `base_run_name` must resolve against `src.pretrain.recipe.RUN_LIST` --
    checked here, not left to the caller, because a typo'd or stale join
    key is exactly the kind of silent-until-someone-notices bug this
    project's "raise rather than guess" discipline exists to catch."""
    from src.pretrain.recipe import RUN_LIST

    valid_base_names = {r.name for r in RUN_LIST}
    if base_run_name not in valid_base_names:
        raise ValueError(
            f"base_run_name={base_run_name!r} is not in src/pretrain/recipe.py's RUN_LIST "
            f"({sorted(valid_base_names)}) -- this is the pretrain-side join key and must "
            "resolve, or guardrail_eval/curves.v1.json readers cannot connect this SFT "
            "run back to the base checkpoint it was tuned from."
        )

    revision = _code_revision()
    eval_results, eval_results_status = _optional_artifact(checkpoint_dir / EVAL_RESULTS_NAME, runs_root=runs_root)
    instr_results, instr_status = _optional_artifact(
        checkpoint_dir / INSTRUCTION_PROBE_RESULTS_NAME, runs_root=runs_root
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "run_name": run_name,
        "run_kind": RUN_KIND,
        "base_run_name": base_run_name,
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": {
            **_named_relpath(checkpoint_dir, root=runs_root, root_name=RUNS_ROOT_NAME),
            "step": manifest["step"],
        },
        "producer": {"name": PRODUCER_NAME, "version": PRODUCER_VERSION, "code_revision": revision},
        "artifacts": {
            "eval_results": eval_results,
            "eval_results_status": eval_results_status,
            "instruction_probe_results": instr_results,
            "instruction_probe_results_status": instr_status,
            "guardrail_eval": _named_relpath(guardrail_eval_path, root=runs_root, root_name=RUNS_ROOT_NAME),
        },
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True, help="SFT checkpoint directory (also holds eval_results.json etc.)")
    ap.add_argument("--runs-root", required=True, help="Root checkpoint-dir must be under -- see guardrail_eval.py's --runs-root")
    ap.add_argument("--run-name", required=True, help="The SFT run's own identifier, e.g. sft_dvf_a_1p3b-lr2e-5")
    ap.add_argument(
        "--base-run-name", required=True,
        help="Must resolve against src/pretrain/recipe.py's RUN_LIST -- the pretraining checkpoint this SFT run was tuned from.",
    )
    ap.add_argument("--guardrail-eval-path", required=True, help="Path to this checkpoint's guardrail_eval.<run>.v1.jsonl (must already exist)")
    ap.add_argument("--out", required=True, help="Output path for the report JSON (overwritten, single object, not JSONL)")
    args = ap.parse_args(argv)

    from src.pretrain.checkpointing.store import MANIFEST_NAME, is_complete

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not is_complete(checkpoint_dir):
        raise ValueError(f"{checkpoint_dir} has no valid manifest.json -- refusing an incomplete checkpoint.")
    with open(checkpoint_dir / MANIFEST_NAME) as fh:
        manifest = json.load(fh)

    guardrail_eval_path = Path(args.guardrail_eval_path).resolve()
    if not guardrail_eval_path.is_file():
        raise ValueError(
            f"{guardrail_eval_path} does not exist -- run guardrail_eval.py against this "
            "checkpoint first; this module does not compute collapse metrics itself."
        )

    report = build_posttrain_eval_report(
        run_name=args.run_name, base_run_name=args.base_run_name, checkpoint_dir=checkpoint_dir,
        runs_root=Path(args.runs_root).resolve(), manifest=manifest,
        guardrail_eval_path=guardrail_eval_path,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    _assert_no_leaked_absolute_paths(serialized)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(serialized + "\n")
    print(f"[posttrain_eval_report] wrote {out_path}")


if __name__ == "__main__":
    main()

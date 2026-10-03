"""Readout for metric #10 (`kind="margin_null"`), gated by a criteria file.

Thin by design: everything about pre-registration -- parsing the criteria,
gating on `status`, checking preconditions, stamping `first_data_seen_at` --
lives in `criteria_gate.py` and is shared by every metric. What is specific
to this metric, and therefore all this module contains, is: how to find the
per-cell readings inside an `offline_margin_null_probe` output file, and how
to lay them out for a human.

This module prints NO threshold of its own and computes no metric. The
supports/contradicts/inconclusive rules live in the criteria file; the
numbers come from `collector.py::margin_null_record`.

`--criteria` is required with no default: the rules in force must be named
on the command line, so a printed table and the criteria it was read under
stay recoverable together.

Usage:
    python -m src.metrics.readout --criteria <file.yaml> \
        --out-dir <dir with *.jsonl from offline_margin_null_probe>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from src.metrics.criteria_gate import (
    Criteria,
    failed_preconditions,
    load_criteria,
    stamp_first_data_seen,
)

# Preconditions this metric's readout checks itself rather than per-cell.
# Declared so `failed_preconditions` does not silently skip them.
HANDLED_ELSEWHERE = ("require_same_probe_seeds",)


def check_single_top_k(cells: list[dict[str, Any]]) -> None:
    """Refuse outright to summarise rho across cells with different `top_k`.

    `margin = top1 - top_k` spans the selected block, so its direction
    depends on k: at k == 2 specialization RAISES rho, while at k > 2 block
    specialization LOWERS it. Pooling the two produces a number whose sign
    means opposite things for different rows.

    This raises rather than warning. A warning would leave a comparison
    printed on the page, and a printed comparison is the thing consumers
    copy -- the same reason an incomplete batch blocks verdicts outright.
    """
    ks = sorted({c["top_k"] for c in cells})
    if len(ks) > 1:
        raise ValueError(
            f"readings span multiple top_k values {ks}. rho is comparable only "
            "within one k: at k == 2 the margin is the top-2 gap and "
            "specialization raises rho, while at k > 2 it spans the selected "
            "block and block specialization LOWERS it. A pooled summary would "
            "mean opposite things for different rows. Read each k separately."
        )


def check_batch_complete(out_dir: str | Path) -> str | None:
    """Return a warning if the output directory is not a COMPLETE batch.

    A probe directory mid-run holds valid, individually-correct files for a
    subset of checkpoints. Nothing about those files says the batch is
    partial, so a consumer reading at that moment renders a subset as if it
    were everything -- no error anywhere. The producer therefore writes
    `DONE.json` only after the last file lands, and this refuses to treat an
    unmarked or short directory as complete. (A consumer once read such a
    directory at 2 of 9 files.)
    """
    out_dir = Path(out_dir)
    marker = out_dir / "DONE.json"
    present = len(list(out_dir.glob("*.jsonl")))
    if not marker.is_file():
        return (
            f"⚠️ INCOMPLETE BATCH: {out_dir} has no DONE.json. It holds "
            f"{present} .jsonl file(s), which may be a run in progress. "
            "Numbers below must not be read as the whole batch."
        )
    done = json.loads(marker.read_text())
    if done.get("produced_files_count") != done.get("expected_files"):
        return (
            f"⚠️ INCOMPLETE BATCH: DONE.json reports "
            f"{done.get('produced_files_count')} of {done.get('expected_files')} files."
        )
    if present != done.get("expected_files"):
        return (
            f"⚠️ BATCH CHANGED SINCE COMPLETION: DONE.json expects "
            f"{done.get('expected_files')} files, directory now holds {present}."
        )
    return None


def iter_cells(out_dir: str | Path) -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Yield `(source_name, probe_blob, cell)` for every per-cell reading."""
    paths = sorted(Path(out_dir).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no *.jsonl found under {out_dir}")
    for path in paths:
        for line in path.read_text().strip().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            blob = row["margin_null"]
            for cell in blob["by_layer_loop"]:
                yield path.stem, blob, cell


def check_shared_probe_seeds(blobs: list[dict[str, Any]], criteria: Criteria) -> str | None:
    """`require_same_probe_seeds`: two runs are only comparable if they were
    measured on the same probe tokens. Returns a warning line, or None."""
    if not criteria.preconditions.get("require_same_probe_seeds"):
        return None
    seen = {tuple(b["probe_seeds"]) for b in blobs}
    if len(seen) > 1:
        return (
            f"⚠️ readings used DIFFERENT probe seeds {sorted(seen)}. The "
            "criteria file requires the same probe tokens across anything "
            "compared; cross-run comparison of these readings is confounded."
        )
    return None


def render(out_dir: str | Path, criteria: Criteria, *, stamp: bool = False) -> int:
    cells = list(iter_cells(out_dir))
    check_single_top_k([c for _, _, c in cells])
    blobs = list({id(b): b for _, b, _ in cells}.values())

    incomplete = check_batch_complete(out_dir)
    if incomplete:
        print(incomplete)
        print()

    print(f"criteria file  : {criteria.path}")
    print(f"criteria status: {criteria.status.upper()}")
    print(f"preconditions  : {dict(criteria.preconditions)}")
    warning = criteria.preregistration_order_warning()
    if warning:
        print(warning)
    print()

    hdr = (
        f"{'source':<30} {'E':>3} {'k':>2} {'cell':>8} {'tokens':>7} "
        f"{'margin':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'null':>8} "
        f"{'rho':>7} {'null band':>16} {'pre':>4}"
    )
    print(hdr)
    print("-" * len(hdr))

    n_failing = 0
    n_inside_own_null = 0
    for name, blob, c in cells:
        failed = failed_preconditions(c, criteria, handled_elsewhere=HANDLED_ELSEWHERE)
        n_failing += 1 if failed else 0
        if c["rho_null_lo"] <= c["rho"] <= c["rho_null_hi"]:
            n_inside_own_null += 1
        band = f"[{c['rho_null_lo']:.3f},{c['rho_null_hi']:.3f}]"
        cell_id = f"L{c['layer_idx']}/R{c['loop_step']}"
        print(
            f"{name:<30} {c['n_experts']:>3} {c['top_k']:>2} {cell_id:>8} "
            f"{c['n_tokens']:>7} {c['margin_mean']:>8.4f} {c['margin_p25']:>8.4f} "
            f"{c['margin_p50']:>8.4f} {c['margin_p75']:>8.4f} "
            f"{c['null_margin_p50']:>8.4f} {c['rho']:>7.4f} {band:>16} "
            f"{'ok' if not failed else 'NO':>4}"
        )

    print("-" * len(hdr))
    print(f"cells: {len(cells)}   failing preconditions: {n_failing}")
    # Descriptive, not a verdict: it compares a cell against ITS OWN null and
    # applies no threshold and no cross-model comparison. This is the one
    # statement a draft-status readout may make, and it is the criteria
    # file's most important inconclusive clause in raw form.
    print(
        f"cells whose rho sits INSIDE their own null band (that cell has no "
        f"measurable specialization): {n_inside_own_null}"
    )

    seed_warning = check_shared_probe_seeds(blobs, criteria)
    if seed_warning:
        print(seed_warning)

    print()
    # An incomplete batch blocks a verdict regardless of criteria status.
    # Without this, a RATIFIED criteria file over a half-written directory
    # prints verdicts with a warning above them -- and a warning above a
    # verdict is not a refusal, it is a caption that gets cropped.
    if incomplete:
        print(
            "NO VERDICT: the batch is incomplete (see the warning above). "
            "Numbers are shown for inspection only; a verdict over a subset "
            "would be a verdict about different data than it claims."
        )
    elif not criteria.verdicts_allowed:
        print(criteria.refusal_text())
        unset = criteria.raw.get("cross_axis_conflict", {}).get("decision") is None
        if unset:
            print(
                "  also unset: cross_axis_conflict.decision (what to do when "
                "the two axes disagree)."
            )
    else:
        print(
            "RATIFIED criteria in force. A supports/contradicts verdict is a "
            "CROSS-RUN comparison; pass the two runs' readings once a "
            "comparator run exists. The per-cell table above stands alone."
        )

    if stamp:
        written = stamp_first_data_seen(criteria)
        if written:
            print(f"\nstamped first_data_seen_at = {written} into {criteria.path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--criteria", required=True, help="Pre-registered criteria YAML (no default)")
    ap.add_argument("--out-dir", required=True, help="Directory of offline_margin_null_probe JSONL")
    ap.add_argument(
        "--stamp-first-data-seen",
        dest="stamp",
        action="store_true",
        help="Record in the criteria file that real values have now been read "
        "(written once; never overwritten).",
    )
    args = ap.parse_args(argv)
    return render(args.out_dir, load_criteria(args.criteria), stamp=args.stamp)


if __name__ == "__main__":
    raise SystemExit(main())

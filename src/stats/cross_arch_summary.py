"""One summary row per archive from a cross-architecture metric batch.

Each archive produced one row per `(loop_step, layer_idx)` cell. This collapses
those to two numbers per quantity -- the mean over cells and the worst cell --
and writes the per-cell `L2` curve alongside, so the depth trend the spec asks
about survives the summary rather than being averaged out of it.

## "Worst" is not the same direction for every quantity

For the concentration metrics the worst cell is the LARGEST value: more load
piled onto fewer experts. For the confidence metric it is the SMALLEST: a
router with no preference among the experts it picked. Taking a maximum for
both would report the least collapsed cell for half the table, which is a
mistake that produces entirely plausible numbers -- so the direction is
declared per quantity here and recorded in the output.

Aggregation across cells is done ONLY here, never in the metric function, and
the spec is explicit that model-level aggregation is undecided. These summaries
exist to be read next to the curve, not instead of it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

#: `field -> which end is the collapsed one`. Anything not listed is summarised
#: by mean only, because its worst end has no agreed meaning.
WORST_END = {
    "L2_starved_fraction": "max",
    "Linf_starved_fraction": "max",
    "L2_unit_scaled": "max",
    "Linf_unit_scaled": "max",
    "L2_starved_fraction_count": "max",
    "Linf_starved_fraction_count": "max",
    "L2_mass_minus_count": "max",
    "maxvio_mass": "max",
    "maxvio_count": "max",
    "C_mean": "min",
    "C_adj": "min",
    "k_eff_mean": "max",
    "margin_mean": "min",
}

MEAN_ONLY = ("C_null", "sigma_hat")


def summarise(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in open(path)]
    if not rows:
        raise ValueError(f"{path} has no rows")
    head = rows[0]
    expected = head["num_stacks"] * head["num_layers_in_stack"]
    if len(rows) != expected:
        raise ValueError(
            f"{path} has {len(rows)} cells but the grid is "
            f"{head['num_stacks']}x{head['num_layers_in_stack']}={expected}. A "
            "summary over a partial grid reads like a complete one."
        )

    out: dict[str, Any] = {
        "archive": path.stem,
        "E": head["E"],
        "k": head["k"],
        "N": head["N"],
        "n_cells": len(rows),
        "corpus_id": head["corpus_id"],
        "manifest_step": head["manifest_step"],
        "manifest_kind": head["manifest_kind"],
        "code_commit": head["producer"]["code_revision"]["commit"][:12],
    }
    for field, end in WORST_END.items():
        if field not in head:
            continue
        values = [r[field] for r in rows]
        out[f"{field}__mean"] = sum(values) / len(values)
        out[f"{field}__worst"] = max(values) if end == "max" else min(values)
        # Recorded so a reader never has to infer which end was taken -- the
        # two directions are both plausible and the field name does not say.
        out[f"{field}__worst_end"] = end
    for field in MEAN_ONLY:
        if field in head:
            out[f"{field}__mean"] = sum(r[field] for r in rows) / len(rows)

    out["L2_curve"] = [
        {
            "loop_step": r["loop_step"],
            "layer_idx": r["layer_idx"],
            "L2_starved_fraction": r["L2_starved_fraction"],
            "C_mean": r["C_mean"],
        }
        for r in sorted(rows, key=lambda r: (r["loop_step"], r["layer_idx"]))
    ]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-dir", required=True, help="Directory of per-archive .jsonl files")
    ap.add_argument("--out-csv", required=True, help="One summary row per archive")
    ap.add_argument("--out-curves", required=True, help="Per-cell L2/C curves, JSONL")
    ap.add_argument(
        "--archived-cells-dir", required=True,
        help="The per-archive .jsonl directory that is COMMITTED. Its file count "
        "must equal the number of summary rows. Required because the table and "
        "the archived cells are what a reader sees together, and a guard that "
        "only checks the file it just wrote cannot notice that the committed "
        "copy fell behind.",
    )
    args = ap.parse_args(argv)

    batch = Path(args.batch_dir)
    files = sorted(batch.glob("*.jsonl"))
    if not files:
        raise ValueError(
            f"{batch} holds no .jsonl files. Refusing to write an empty summary: "
            "a table with no rows and no complaint reads like a measurement that "
            "found nothing."
        )
    summaries = [summarise(p) for p in files]
    # The aggregate must account for every artifact: a table can otherwise be
    # built short while some products are absent or newly added, with nothing
    # comparing the two. Checked in code, not by eye.
    if len(summaries) != len(files):
        raise ValueError(
            f"{len(files)} artifacts produced {len(summaries)} summary rows. "
            "Refusing to write a table that does not account for every archive."
        )

    curves_path = Path(args.out_curves)
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    with open(curves_path, "w") as fh:
        for s in summaries:
            fh.write(json.dumps({"archive": s["archive"], "E": s["E"], "k": s["k"],
                                 "cells": s["L2_curve"]}) + "\n")

    flat = [{k: v for k, v in s.items() if k != "L2_curve"} for s in summaries]
    columns = list(flat[0])
    for row in flat:
        if list(row) != columns:
            raise ValueError(
                f"{row['archive']} has different columns from the first archive. "
                "Writing them into one table would silently align values under "
                "the wrong headers."
            )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    archived = sorted(Path(args.archived_cells_dir).glob("*.jsonl"))
    if len(archived) != len(flat):
        raise ValueError(
            f"{len(flat)} summary rows but {len(archived)} archived cell files in "
            f"{args.archived_cells_dir}. The table and the cells are read together; "
            "one of them is behind the other. Refusing to write a table that "
            "disagrees with the artifacts committed beside it."
        )
    print(f"[cross_arch_summary] {len(flat)} archives -> {out_csv}")
    print(f"[cross_arch_summary] row-count check: {len(flat)} rows == {len(files)} "
          f"source artifacts == {len(archived)} archived cells")
    print(f"[cross_arch_summary] curves -> {curves_path}")


if __name__ == "__main__":
    main()

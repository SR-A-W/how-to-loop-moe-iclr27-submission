"""Compare two runs of the same probe over the same checkpoints, cell by cell.

Used to check locally-produced comparator artifacts against the ones the
training cluster returned: same driver, same commit, same corpus, same parameters, different
machine. Agreement is evidence the two sides are running the same measurement;
disagreement is a finding, and this script reports it rather than explaining it.

## Why a tolerance rather than bit-equality

Matrix multiplication reduction order depends on the hardware and on batch
shape, so logits differ in the last places and everything downstream inherits
that. Demanding bit-equality here would fail for a reason unrelated to what is
being checked, and "make it pass" would then mean loosening the criterion until
it did -- which is how a check stops meaning anything.

`topk_idx` is NOT compared here because the dumps are not both available; the
selection-stability question is asked separately against a chunk-size sweep.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

#: Reported individually because they are the quantities conclusions are drawn
#: from. Every other numeric field is still compared -- see `compare_rows`.
PRINCIPAL = (
    "L2_starved_fraction",
    "Linf_starved_fraction",
    "L2_unit_scaled",
    "Linf_unit_scaled",
    "C_mean",
    "C_null",
    "C_adj",
    "margin_mean",
    "maxvio_mass",
)

#: Compared for exact equality instead of tolerance: these identify WHICH
#: measurement this is, and a near-match would mean the two sides are not
#: describing the same thing at all.
IDENTITY = ("E", "k", "N", "layer_idx", "loop_step", "num_stacks", "num_layers_in_stack",
            "corpus_id", "manifest_step", "n_documents", "doc_chunk",
            "c_null_tokens", "c_null_seed")

#: Run configuration that does NOT enter any metric. Reported separately when
#: the two sides differ, never compared against the numeric tolerance.
#:
#: `hidden_subsample_tokens` decides whether residual vectors are written to the
#: dump. One side may run a checkpoint as a final state (50000) while the other
#: runs it as a trajectory archive (0). Every metric can then agree to 5e-5,
#: yet run through a numeric tolerance the parameter shows up as a 5.0e+04
#: "disagreement" and the archive is marked failed.
#:
#: Excluding it is NOT the same as ignoring it: a difference here is reported
#: as a configuration difference with both values, because it is real and
#: someone should know the two sides were not configured identically. What it
#: must not do is masquerade as a measurement disagreement.
CONFIG = ("hidden_subsample_tokens", "hidden_seed", "n_forward_chunks")


def _rel(a: float, b: float) -> float:
    """Relative difference, falling back to absolute when both are ~0."""
    scale = max(abs(a), abs(b))
    return abs(a - b) if scale < 1e-12 else abs(a - b) / scale


def _abs(a: float, b: float) -> float:
    return abs(a - b)


#: ⚠️ Both differences are reported, and which one the verdict uses matters.
#: A relative difference is undefined in practice for a quantity that sits near
#: zero: `L2_mass_minus_count` is a subtraction of two nearly equal numbers, so
#: an absolute difference of 3.7e-6 shows up as a 3.8e-2 "relative" difference
#: and reads like a serious disagreement. Count-share fields have the same
#: shape -- one token's worth of share is already ~1e-5.
#:
#: The verdict therefore uses the ABSOLUTE difference, which is the quantity
#: the hardware explanation actually bounds: different reduction order changes
#: the last places of a sum, not its ratio to zero.


def compare_rows(mine: dict[str, Any], theirs: dict[str, Any]) -> dict[str, float]:
    """`{field: relative difference}` over every shared numeric field.

    Identity fields must match exactly; a mismatch raises, because a tolerance
    on "which layer is this" is meaningless.
    """
    for field in IDENTITY:
        if field in mine and field in theirs and mine[field] != theirs[field]:
            raise ValueError(
                f"identity field {field!r} differs: {mine[field]!r} vs {theirs[field]!r}. "
                "These two rows do not describe the same measurement, so comparing "
                "their values would produce a number with no meaning."
            )
    out = {}
    config_diffs = {}
    for field in CONFIG:
        a, b = mine.get(field), theirs.get(field)
        if a is not None and b is not None and a != b:
            config_diffs[field] = {"mine": a, "theirs": b}
    if config_diffs:
        out["__config_differences__"] = config_diffs
    for field, value in mine.items():
        other = theirs.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not isinstance(other, (int, float)) or isinstance(other, bool):
            continue
        if field in IDENTITY or field in CONFIG:
            continue
        if math.isnan(value) or math.isnan(other):
            raise ValueError(f"{field} is NaN on one side: {value} vs {other}")
        out[field] = {
            "rel": _rel(float(value), float(other)),
            "abs": _abs(float(value), float(other)),
            "mine": float(value),
            "theirs": float(other),
        }
    return out


def _cells_by_position(npz: Any, num_layers_in_stack: int) -> "dict[int, tuple]":
    """`{unrolled_pos: (topk_idx, topk_weight or None)}` for either dump layout.

    Two layouts exist and must be comparable, because the archives being
    rechecked predate the head/tail change:

    * `topk_idx_p{unrolled_pos}` -- written after the change;
    * `topk_idx_r{loop_step}_l{layer_idx}` -- written before it, where the depth
      coordinate is `unrolled_position(loop_step, layer_idx, num_layers_in_stack)`.

    Keying both on the depth coordinate is what lets a flip be counted per cell
    rather than per key name. Comparing by key name instead would report every
    cell as missing -- which is a loud failure, but the wrong answer to a
    question that does have a right one.
    """
    from src.model.loop_trace import unrolled_position

    out: dict[int, tuple] = {}
    for key in npz.files:
        if not key.startswith("topk_idx"):
            continue
        rest = key[len("topk_idx_"):]
        if rest.startswith("p"):
            pos = int(rest[1:])
        elif rest.startswith("r") and "_l" in rest:
            r, l = rest[1:].split("_l")
            pos = unrolled_position(int(r), int(l), num_layers_in_stack)
        else:
            raise ValueError(f"unrecognised dump key {key!r}")
        wkey = key.replace("topk_idx", "topk_weight")
        out[pos] = (npz[key], npz[wkey] if wkey in npz.files else None)
    return out


def compare_topk(mine_npz: "Path", theirs_npz: "Path", num_layers_in_stack: int) -> dict[str, Any]:
    """Selection flips between two dumps of the same checkpoint, adjudicated.

    ## What this can and cannot say

    A flip is reported, never failed on. Whether bit-identical selection is even
    expected depends on the hardware both sides ran on, and the older
    archives record no hostname, GPU model or library version anywhere -- so the
    premise cannot be established. Demanding exactness and then loosening it
    when flips appeared would be choosing the criterion after seeing the result.

    Each flip is classified by whether the two sides' rank-2 weights agree to
    within one float16 spacing at that magnitude. **This is INDIRECT evidence
    and it is ASYMMETRIC**:

    - agreement supports "a near-tie that two reduction orders resolved
      differently";
    - disagreement does NOT establish a real difference. For a boundary flip the
      displaced expert's weight was never dumped (only the top k are stored), so
      we cannot see how far away it was.

    The threshold is `np.spacing(w)` per element, NOT a constant. The float16 gap
    runs from 6.10e-05 at 0.1 to 9.77e-04 at 1.0 -- a factor of 16. One constant
    would be far too loose at one end and far too strict at the other, and both
    mistakes produce entirely plausible numbers. (The same reason a fixed 1e-4
    threshold is unusable here: at 0.5 two adjacent float16 values already differ
    by 4.88e-04, five times that.)
    """
    import numpy as np

    a, b = np.load(mine_npz), np.load(theirs_npz)
    # The two sides may use DIFFERENT dump layouts, and comparing them is the
    # whole point: archives written before the head/tail change key cells
    # `r{loop_step}_l{layer_idx}`, ones written after key them `p{unrolled_pos}`.
    # Both are mapped onto the depth coordinate before pairing, using the same
    # `unrolled_position` arithmetic the offline reader uses.
    ka = _cells_by_position(a, num_layers_in_stack)
    kb = _cells_by_position(b, num_layers_in_stack)
    if not ka:
        raise ValueError(f"{mine_npz.name} holds no topk_idx arrays")
    only_mine, only_theirs = sorted(set(ka) - set(kb)), sorted(set(kb) - set(ka))
    if only_mine or only_theirs:
        raise ValueError(
            f"cell sets differ: {mine_npz.name} has {only_mine or 'none'} that "
            f"{theirs_npz.name} lacks, and it has {only_theirs or 'none'} extra. "
            "A flip count over different cell sets is not defined."
        )
    idx_keys = sorted(ka)

    total = flips = tie_like = other = 0
    per_cell = {}
    for k in idx_keys:
        ia, ib = ka[k][0], kb[k][0]
        if ia.shape != ib.shape:
            raise ValueError(f"{k}: shapes differ, {ia.shape} vs {ib.shape}")
        differing = ia != ib
        n_flip = int(differing.any(axis=1).sum())
        total += int(ia.shape[0])
        flips += n_flip
        if not n_flip:
            continue
        cell_tie = cell_other = 0
        wa, wb = ka[k][1], kb[k][1]
        if wa is not None and wb is not None:
            rows = np.where(differing.any(axis=1))[0]
            # The LAST selected slot is the boundary one -- the rank most easily
            # displaced by a near-tie with the first unselected expert.
            la, lb = wa[rows, -1].astype(np.float16), wb[rows, -1].astype(np.float16)
            within = np.abs(la.astype(np.float64) - lb.astype(np.float64)) <= np.spacing(
                np.maximum(np.abs(la), np.abs(lb))
            )
            cell_tie = int(within.sum())
            cell_other = int(n_flip - cell_tie)
        else:
            cell_other = n_flip
        tie_like += cell_tie
        other += cell_other
        per_cell[f"p{k}"] = {"flips": n_flip, "tie_like": cell_tie, "unexplained": cell_other}

    return {
        "tokens_compared": total,
        "flips": flips,
        "flip_rate": (flips / total) if total else 0.0,
        "tie_like": tie_like,
        "unexplained": other,
        "per_cell": per_cell,
        "evidence": (
            "indirect and asymmetric: rank-2 weight agreement within one float16 "
            "spacing supports a near-tie resolved differently; disagreement does "
            "not establish a real difference, because the displaced expert's "
            "weight is not in the dump"
        ),
        "verdict_affecting": False,
    }


def compare_files(mine_path: Path, theirs_path: Path) -> dict[str, Any]:
    mine = [json.loads(line) for line in open(mine_path)]
    theirs = [json.loads(line) for line in open(theirs_path)]
    if len(mine) != len(theirs):
        raise ValueError(
            f"{mine_path.name}: {len(mine)} cells here, {len(theirs)} there. "
            "A per-cell comparison over different grids is not defined."
        )
    key = lambda r: (r["loop_step"], r["layer_idx"])
    mine.sort(key=key); theirs.sort(key=key)

    worst: dict[str, dict[str, Any]] = {}
    config_differences: dict[str, Any] = {}
    for a, b in zip(mine, theirs):
        if key(a) != key(b):
            raise ValueError(f"{mine_path.name}: cell {key(a)} has no counterpart {key(b)}")
        row_diffs = compare_rows(a, b)
        for field, cfg in row_diffs.pop("__config_differences__", {}).items():
            config_differences[field] = cfg
        for field, diff in row_diffs.items():
            # The two maxima are tracked INDEPENDENTLY. They are usually at
            # different cells; storing them in one dict replaced whenever a new
            # absolute maximum arrives would discard the running relative
            # maximum and silently under-report it.
            slot = worst.setdefault(
                field,
                {"abs_diff": -1.0, "rel_diff": -1.0, "mine": None, "theirs": None,
                 "cell": None, "max_rel_diff_seen": -1.0, "max_rel_cell": None,
                 "max_rel_mine": None, "max_rel_theirs": None},
            )
            if diff["abs"] > slot["abs_diff"]:
                slot.update(
                    abs_diff=diff["abs"], rel_diff=diff["rel"],
                    mine=diff["mine"], theirs=diff["theirs"], cell=list(key(a)),
                )
            if diff["rel"] > slot["max_rel_diff_seen"]:
                slot.update(
                    max_rel_diff_seen=diff["rel"], max_rel_cell=list(key(a)),
                    max_rel_mine=diff["mine"], max_rel_theirs=diff["theirs"],
                )
    return {
        "archive": mine_path.stem,
        "n_cells": len(mine),
        "worst_by_field": worst,
        "max_abs_diff": max((w["abs_diff"] for w in worst.values()), default=0.0),
        "max_abs_diff_field": max(worst, key=lambda f: worst[f]["abs_diff"]) if worst else None,
        "max_rel_diff": max((w["max_rel_diff_seen"] for w in worst.values()), default=0.0),
        "config_differences": config_differences,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mine-dir", required=True)
    ap.add_argument("--theirs-dir", required=True)
    ap.add_argument("--pairs", required=True,
                    help="Lines of '<my stem> <their stem>' naming the files to compare.")
    ap.add_argument("--tolerance", type=float, required=True,
                    help="Maximum acceptable ABSOLUTE difference. REQUIRED and with "
                    "no default: a default here would be a second copy of a number "
                    "that belongs in the instruction, and loosening it silently is "
                    "exactly how this check would stop meaning anything.")
    ap.add_argument("--out-json", required=True)
    ap.add_argument(
        "--compare-topk", action="store_true",
        help="Also compare the topk_idx dumps and report selection flips. "
        "REPORTED, never failed on: whether bit-identical selection is expected "
        "depends on hardware the old archives do not record. Needs a .npz beside "
        "each .jsonl.",
    )
    args = ap.parse_args(argv)

    results = []
    failures = []
    for line in open(args.pairs):
        line = line.strip()
        if not line:
            continue
        mine_stem, theirs_stem = line.split()
        res = compare_files(
            Path(args.mine_dir) / f"{mine_stem}.jsonl",
            Path(args.theirs_dir) / f"{theirs_stem}.jsonl",
        )
        if args.compare_topk:
            # From the rows themselves, not a flag: the two dumps must be mapped
            # onto the same depth coordinate and this is the number that does it.
            n_layers = int(json.loads(open(Path(args.mine_dir) / f"{mine_stem}.jsonl")
                                      .readline())["num_layers_in_stack"])
            mine_npz = (Path(args.mine_dir) / f"{mine_stem}.npz")
            theirs_npz = (Path(args.theirs_dir) / f"{theirs_stem}.npz")
            if mine_npz.exists() and theirs_npz.exists():
                res["topk"] = compare_topk(mine_npz, theirs_npz, n_layers)
            else:
                # Say which side is absent. "no flips" and "never looked" must
                # not read the same way in the report.
                res["topk"] = {"status": "dump_missing",
                               "mine_npz": mine_npz.exists(),
                               "theirs_npz": theirs_npz.exists()}
        results.append(res)
        if res["max_abs_diff"] > args.tolerance:
            failures.append(res)
        cfg = res["config_differences"]
        note = f"  ⚠ config differs: {cfg}" if cfg else ""
        print(f"[xcheck] {mine_stem}: max abs {res['max_abs_diff']:.3e} "
              f"({res['max_abs_diff_field']}), max rel {res['max_rel_diff']:.3e}{note}", flush=True)
        t = res.get("topk")
        if t and t.get("status") != "dump_missing":
            print(f"           topk: {t['flips']} flips / {t['tokens_compared']} tokens "
                  f"({t['flip_rate']:.2e}), {t['tie_like']} tie-like, "
                  f"{t['unexplained']} unexplained -- REPORTED, not a verdict", flush=True)
        elif t:
            print("           topk: dump missing on one side, not compared", flush=True)

    summary = {
        "tolerance": args.tolerance,
        "n_compared": len(results),
        "n_over_tolerance": len(failures),
        "criterion": "absolute difference",
        "max_abs_diff_overall": max((r["max_abs_diff"] for r in results), default=0.0),
        "max_rel_diff_overall": max((r["max_rel_diff"] for r in results), default=0.0),
        "archives_with_config_differences": {
            r["archive"]: r["config_differences"] for r in results if r["config_differences"]
        },
        "results": results,
    }
    if args.compare_topk:
        cmp = [r["topk"] for r in results if r.get("topk", {}).get("status") != "dump_missing"]
        summary["topk_summary"] = {
            "archives_compared": len(cmp),
            "total_flips": sum(t["flips"] for t in cmp),
            "total_tie_like": sum(t["tie_like"] for t in cmp),
            "total_unexplained": sum(t["unexplained"] for t in cmp),
            "affects_verdict": False,
        }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f"[xcheck] {len(results)} archives, worst abs {summary['max_abs_diff_overall']:.3e}, "
          f"{len(failures)} over tolerance -> {args.out_json}")
    if failures:
        raise SystemExit(
            f"{len(failures)} archive(s) exceeded the tolerance: "
            f"{[f['archive'] for f in failures]}"
        )


if __name__ == "__main__":
    main()

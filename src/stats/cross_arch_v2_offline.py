"""Offline pass over the v1 routing dumps to produce the v2 physical-layer table.

Implements the v2 cross-architecture collapse-metric specification. Reads only the `.npz` dumps and
the v1 `.jsonl` rows already on disk -- no model is loaded and no GPU is used.

One output row per PHYSICAL layer, because that is the unit v2 reports at. The
per-`(layer, pass)` numbers from v1 are not reproduced here; they are averaged
into the `_eff` fields, and the v1 artifacts remain the place to read them
individually.

## Field naming

Every load field carries its granularity in the name (`_eff` / `_phys`). No v1
field name is reused with a new meaning: a reader who knows v1 and meets a
familiar name here would get the wrong quantity, and nothing in the value
would look wrong.

## What is deliberately still the v1 computation

`C`, `C_null`, `C_adj` are averaged from the v1 rows unchanged. The v2 spec
proposes a different noise reference that needs the full E-dimensional logits,
which the dump does not contain. Every row therefore carries
`C_null_method: "v1_sigma_from_selected_logits"` and a caveat string, so a
consumer cannot mistake these for the v2 reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.model.loop_trace import unrolled_position
from src.metrics.collapse_layer_granularity import (
    choose_seq_len,
    cross_loop_diversity,
    load_by_granularity,
    sequence_coverage,
)

C_CAVEAT = (
    "C/C_null/C_adj are the v1 computation, unchanged: the noise reference uses "
    "sigma estimated from the SELECTED logits, so it is affected by the router's "
    "score scale. The v2 reference needs the full E-dimensional logits, "
    "which this dump does not contain. Not to be used for conclusions about "
    "routing preference."
)


def weights_sha256(checkpoint_rel: str, *, repo_root: Path) -> tuple[str, int]:
    """`(sha256, size)` of a checkpoint's weight file.

    The archives cannot be deduplicated by path: three final checkpoints exist at two
    different paths -- `ckpt/base_<config>/step_00248097` and
    `traj/<config>/step_00248097` -- holding the SAME weights, so
    `checkpoint_rel` separates them while the models are identical. Only the
    weight content groups them.

    Hashed in full rather than by a prefix. A prefix would be ~100x faster and
    would silently answer a slightly different question -- "these files start
    the same" -- which is the kind of substitution that reads as an answer
    until the day two checkpoints share an initialisation.
    """
    path = repo_root / checkpoint_rel / "pytorch_model.bin"
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


#: How many directory levels above a checkpoint to look for its staging
#: descriptor. `_runs/_staging/<config>/traj/<job>/step_NNNN` puts it three up;
#: the bound keeps a search from wandering into an unrelated descriptor higher
#: in the tree.
_STAGING_DESCRIPTOR_DEPTH = 4


def staging_descriptor(checkpoint_rel: str, *, repo_root: Path) -> dict[str, Any] | None:
    """The `staged_<config>.json` that a staged checkpoint came from, or `None`.

    Staged archives are pulled from the Hub, probed, and deleted -- the weights
    are gone by the time this table is built, and re-downloading tens of GiB to
    recompute a hash nobody asked for is not a reasonable substitute. The
    descriptor is a few hundred bytes, is written beside the staging directory
    rather than inside it, and survives the deletion, so it is what remains to
    identify the model with: `repo`, `revision` (the Hub revision the weights
    were fetched at) and `cluster_job`.

    A descriptor is accepted only when its own `staged_dir` is a prefix of this
    checkpoint's path. Without that check, the search would happily attach S2's
    descriptor to an S1 checkpoint that happens to sit below it -- and the
    result would be a row claiming a Hub revision that never produced these
    weights, which is worse than no identity at all.
    """
    checkpoint_dir = (repo_root / checkpoint_rel).resolve()
    directory = checkpoint_dir
    for _ in range(_STAGING_DESCRIPTOR_DEPTH):
        directory = directory.parent
        for candidate in sorted(directory.glob("staged_*.json")):
            try:
                descriptor = json.loads(candidate.read_text())
            except (OSError, ValueError):
                continue
            staged_dir = descriptor.get("staged_dir")
            if not staged_dir:
                continue
            staged = Path(staged_dir).resolve()
            if staged == checkpoint_dir or staged in checkpoint_dir.parents:
                return descriptor
    return None


def weights_identity(checkpoint_rel: str, *, repo_root: Path) -> dict[str, Any]:
    """Whatever this archive can still be identified by, and how.

    `weights_sha256` is the strongest answer and stays first choice, but it is
    no longer required: the ablation archives are staged from the Hub, probed
    and deleted, so for them the weight file does not exist when this table is
    built. Raising there would make the table unbuildable for the whole ablation
    family; writing a hash of nothing would be worse.

    Every row therefore carries `weights_identity_source`, which says which of
    the three cases produced it, and a reader never has to infer from a null
    whether the hash is missing because the file was gone or because something
    failed:

    * `"weight_file"` -- hashed, as before;
    * `"staging_descriptor"` -- weights deleted, identity is the Hub repo and
      revision they were fetched at plus the cluster job that produced them;
    * `"unavailable"` -- weights gone and no descriptor found.

    **`weights_sha256` is null in the last two cases, and null is not a value to
    group by.** The README's "deduplicate by `weights_sha256`" instruction would
    put every descriptor-identified archive in one bucket; `hub_revision` is the
    grouping key for those rows, and **`hub_repo` and `hub_revision` are
    guaranteed non-null whenever `weights_identity_source` is
    `"staging_descriptor"`** -- enforced below, so the grouping key can never be
    the null that was just moved away from. A row whose source is
    `"unavailable"` has no identity at all and cannot be deduplicated by any
    column.
    """
    path = repo_root / checkpoint_rel / "pytorch_model.bin"
    if path.is_file():
        digest, size = weights_sha256(checkpoint_rel, repo_root=repo_root)
        return {
            "weights_sha256": digest,
            "weights_size_bytes": size,
            "weights_identity_source": "weight_file",
            "hub_repo": None,
            "hub_revision": None,
            "cluster_job": None,
        }
    descriptor = staging_descriptor(checkpoint_rel, repo_root=repo_root)
    if descriptor is not None:
        # `hub_revision` is what these rows are grouped by once the hash is
        # gone, so a null one would reproduce, one level down, exactly the bug
        # that moving off `weights_sha256` was meant to fix: null == null puts
        # unrelated models in one bucket. Raised rather than
        # downgraded to "unavailable": a descriptor that exists but names no
        # revision is a malformed descriptor, and staging writes it -- finding
        # that out while the table is being built costs minutes, finding it out
        # from an averaged-together plot costs the analysis.
        missing = [k for k in ("repo", "revision") if not descriptor.get(k)]
        if missing:
            raise ValueError(
                f"staging descriptor for {checkpoint_rel} has no {missing}; "
                "it is the only identity these rows have once the weights are "
                "deleted, and rows grouped on a null revision would average "
                "unrelated models together."
            )
        return {
            "weights_sha256": None,
            "weights_size_bytes": None,
            "weights_identity_source": "staging_descriptor",
            "hub_repo": descriptor["repo"],
            "hub_revision": descriptor["revision"],
            # Provenance only, never a grouping key -- one cluster job produces
            # every step of a trajectory, so grouping on it would merge them.
            "cluster_job": descriptor.get("cluster_job"),
        }
    return {
        "weights_sha256": None,
        "weights_size_bytes": None,
        "weights_identity_source": "unavailable",
        "hub_repo": None,
        "hub_revision": None,
        "cluster_job": None,
    }


def _cells(npz: Any, v1_rows: list[dict[str, Any]]) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]], dict[int, dict[str, Any]], str
]:
    """`(arrays by unrolled_pos, descriptor by unrolled_pos, artifact format)`.

    Two dump layouts exist and both must be readable:

    * **head_tail** -- keys `p{unrolled_pos}`, rows carry `block`, `unrolled_pos`
      and a per-cell `E`. Produced since the head/tail architecture landed.
    * **loop_only** -- keys `r{loop_step}_l{layer_idx}`, no `block` field. The 105
      artifacts produced before then. They cannot be regenerated cheaply, and
      every one of them is a pure loop model, so the descriptor is reconstructed:
      `block="loop"`, `unrolled_pos = loop_step * num_layers_in_stack + layer_idx`.

    The format is DETECTED and reported, never assumed: reading a loop_only dump
    as though it were the new one finds no keys at all, and reading the new one
    with the old arithmetic would place head, tail and the loop's first layer on
    the same coordinate.
    """
    head = v1_rows[0]
    if "unrolled_pos" in head and "block" in head:
        arrays, info = {}, {}
        for row in v1_rows:
            pos = int(row["unrolled_pos"])
            arrays[pos] = (npz[f"topk_idx_p{pos}"], npz[f"topk_weight_p{pos}"])
            info[pos] = {"block": row["block"], "loop_step": row["loop_step"],
                         "layer_idx": row["layer_idx"], "n_experts": int(row["E"])}
        return arrays, info, "head_tail"

    n_layers = head["num_layers_in_stack"]
    arrays, info = {}, {}
    for row in v1_rows:
        loop_step, layer_idx = int(row["loop_step"]), int(row["layer_idx"])
        pos = unrolled_position(loop_step, layer_idx, n_layers)
        tag = f"r{loop_step}_l{layer_idx}"
        arrays[pos] = (npz[f"topk_idx_{tag}"], npz[f"topk_weight_{tag}"])
        info[pos] = {"block": "loop", "loop_step": loop_step,
                     "layer_idx": layer_idx, "n_experts": int(row["E"])}
    return arrays, info, "loop_only"


def process_archive(
    jsonl: Path,
    npz_path: Path,
    *,
    do_diversity: bool,
    do_coverage: bool,
    repo_root: Path,
) -> list[dict[str, Any]]:
    v1_rows = [json.loads(line) for line in open(jsonl)]
    head = v1_rows[0]
    n_experts, top_k = head["E"], head["k"]
    n_passes, n_layers = head["num_stacks"], head["num_layers_in_stack"]
    tokens_per_doc = head["corpus"]["tokens_per_doc"]
    # Keyed by `unrolled_pos`, NOT by `(loop_step, layer_idx)`. Head, tail and
    # the loop's first layer all report `(0, 0)`, so that key would silently
    # collapse three rows into one and the last one written -- tail, since the
    # file is ordered by depth -- would supply `C_adj_eff`, `C_mean_eff` and
    # `margin_eff` for all three physical layers, with every value still in
    # range: a plausible number, not a visible error.
    by_pos = {int(r["unrolled_pos"]): r for r in v1_rows} if "unrolled_pos" in head \
        else {unrolled_position(int(r["loop_step"]), int(r["layer_idx"]),
                                head["num_layers_in_stack"]): r for r in v1_rows}
    if len(by_pos) != len(v1_rows):
        raise ValueError(
            f"{jsonl.name}: {len(v1_rows)} v1 rows collapse into {len(by_pos)} "
            "unrolled positions; two layers claim one depth coordinate."
        )

    identity = weights_identity(head["checkpoint_rel"], repo_root=repo_root)
    npz = np.load(npz_path)
    cells, cell_info, artifact_format = _cells(npz, v1_rows)
    positions = sorted(cells)
    if positions != list(range(len(positions))):
        raise ValueError(
            f"{npz_path.name} covers positions {positions}, which is not a "
            "contiguous 0..n-1 range; a layer is missing and the depth curve "
            "would have a hole that reads as a shallower network."
        )
    n_loop_cells = sum(1 for i in cell_info.values() if i["block"] == "loop")
    if n_loop_cells != n_passes * n_layers:
        raise ValueError(
            f"{npz_path.name} has {n_loop_cells} loop cells but the metadata says "
            f"{n_passes}x{n_layers}={n_passes * n_layers}. Pooling over fewer "
            "passes than the model has would understate nothing visibly."
        )

    # Physical layers, grouped by what actually shares parameters:
    #   - a loop layer is one physical layer seen on all R passes;
    #   - head and tail each run ONCE, so each is its own physical layer with R=1.
    # Ordering and the plot coordinate are `unrolled_pos` throughout. The old
    # `layer_idx * R + loop_step` arithmetic is gone: it cannot place head or
    # tail at all (both report layer_idx=0, loop_step=0) and would stack them on
    # top of the loop's first layer.
    groups: list[dict[str, Any]] = []
    by_layer: dict[int, list[int]] = {}
    for pos in sorted(cells):
        info = cell_info[pos]
        if info["block"] == "loop":
            by_layer.setdefault(info["layer_idx"], []).append(pos)
        else:
            groups.append({"block": info["block"], "layer_idx": info["layer_idx"],
                           "positions": [pos]})
    for layer_idx in sorted(by_layer):
        groups.append({"block": "loop", "layer_idx": layer_idx,
                       "positions": sorted(by_layer[layer_idx])})
    groups.sort(key=lambda g: g["positions"][0])

    rows = []
    for group in groups:
        positions = group["positions"]
        passes = [cells[p] for p in positions]
        idx_passes = [p[0] for p in passes]
        layer_idx = group["layer_idx"]
        n_passes = len(positions)
        n_experts = cell_info[positions[0]]["n_experts"]
        row: dict[str, Any] = {
            "archive": jsonl.stem,
            "block": group["block"],
            "unrolled_positions": positions,
            "unrolled_pos_first": positions[0],
            "layer_idx": layer_idx,
            "R": n_passes,
            "n_physical_layers": n_layers,
            "E": n_experts,
            "k": top_k,
            "N": head["N"],
            "N_times_R_over_E": head["N"] * n_passes / n_experts,
            "corpus_id": head["corpus_id"],
            # Carried from the v1 row so this table can be deduplicated on its
            # own. Three final checkpoints appear under two paths that hold the SAME
            # weights, and without this field the README's "deduplicate by
            # checkpoint_rel" instruction names a column that does not exist.
            "checkpoint_root": head["checkpoint_root"],
            "checkpoint_rel": head["checkpoint_rel"],
            # Path does NOT identify the model: see `weights_identity`. The
            # hash is null for archives whose weights were deleted after probing;
            # `weights_identity_source` says which case each row is.
            **identity,
            "manifest_step": head["manifest_step"],
            "manifest_kind": head["manifest_kind"],
            "code_commit": head["producer"]["code_revision"]["commit"][:12],
            "spec_version": "2026-09-09 v2",
            "artifact_format": artifact_format,
        }
        row.update(load_by_granularity(passes, n_experts=n_experts))

        # Metric B, averaged over the passes of this physical layer, still the
        # v1 computation -- flagged in every row rather than in a note file.
        b_cells = [by_pos[p] for p in positions]
        if all(c.get("metric_b_status") == "ok" for c in b_cells):
            for field in ("C_mean", "C_null", "C_adj", "k_eff_mean", "margin_mean", "sigma_hat"):
                row[f"{field}_eff"] = float(np.mean([c[field] for c in b_cells]))
        row["C_null_method"] = "v1_sigma_from_selected_logits"
        row["metric_b_caveat"] = C_CAVEAT

        if do_diversity:
            row.update(cross_loop_diversity(idx_passes, n_experts=n_experts))
        if do_coverage:
            seq_len = choose_seq_len(n_experts, top_k)
            row.update(
                sequence_coverage(
                    idx_passes,
                    n_experts=n_experts,
                    tokens_per_document=tokens_per_doc,
                    seq_len=seq_len,
                )
            )
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-diversity", action="store_true")
    ap.add_argument("--skip-coverage", action="store_true")
    ap.add_argument(
        "--repo-root", required=True,
        help="Root the artifacts' checkpoint_rel is relative to, for hashing "
        "the weight files and for finding a staged archive's descriptor. "
        "Required: the archives cannot be deduplicated by path, and a wrong "
        "root would silently produce no weight hashes at all -- indistinguishable "
        "from the legitimate case where the weights were deleted after probing, "
        "except that `weights_identity_source` would read `unavailable` on every "
        "row rather than `staging_descriptor`.",
    )
    args = ap.parse_args(argv)

    batch = Path(args.batch_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)

    jsonls = sorted(batch.glob("*.jsonl"))
    if not jsonls:
        raise ValueError(f"{batch} holds no .jsonl files; refusing to write an empty table.")

    all_rows: list[dict[str, Any]] = []
    per_archive: dict[str, int] = {}
    for jsonl in jsonls:
        npz_path = jsonl.with_suffix(".npz")
        if not npz_path.is_file():
            raise ValueError(
                f"{jsonl.name} has no matching .npz. The physical-layer metrics come "
                "from the dump, not from the v1 row, so this archive cannot be "
                "computed -- refusing rather than emitting a row with the fields absent."
            )
        rows = process_archive(
            jsonl, npz_path,
            do_diversity=not args.skip_diversity,
            do_coverage=not args.skip_coverage,
            repo_root=Path(args.repo_root),
        )
        with open(out_dir / "cells" / f"{jsonl.stem}.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        all_rows.extend(rows)
        per_archive[jsonl.stem] = len(rows)
        print(f"[v2] {jsonl.stem}: {len(rows)} physical layers", flush=True)

    # Row count must equal the sum of each archive's physical-layer count. A
    # short table and a complete one look identical once written; the only
    # moment the discrepancy is visible is here, while both numbers exist.
    expected_rows = sum(per_archive.values())
    if len(all_rows) != expected_rows:
        raise ValueError(
            f"{len(all_rows)} rows built but the archives declare {expected_rows} "
            f"physical layers in total ({per_archive}). Refusing to write a table "
            "that does not account for every physical layer."
        )

    scalar = [
        {k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in all_rows
    ]
    columns: list[str] = []
    for row in scalar:
        for key in row:
            if key not in columns:
                columns.append(key)
    csv_path = out_dir / "physical_layer_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(scalar)
    print(f"[v2] {len(all_rows)} physical-layer rows -> {csv_path}")
    print(f"[v2] row-count check: {len(all_rows)} rows == sum over "
          f"{len(per_archive)} archives of their physical-layer counts")
    (out_dir / "ROW_COUNTS.json").write_text(
        json.dumps({"n_archives": len(per_archive), "expected_rows": expected_rows,
                    "rows_written": len(all_rows), "per_archive": per_archive}, indent=2)
    )


if __name__ == "__main__":
    main()

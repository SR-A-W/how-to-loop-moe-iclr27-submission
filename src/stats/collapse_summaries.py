"""Build the two missing collapse-metric summary tables from probe artifacts.

`stats/collapse_metrics/` holds SUMMARY tables only -- the per-checkpoint probe
JSONL stays out of git. The margin table is produced elsewhere; this produces
the other two:

* **z-loss**: one row per (archive, layer_idx, loop_step), carrying the sensor
  fields and the pre-registered verdict inputs, from `zloss_probe/out/*.jsonl`.
* **open-source router**: one row per (batch directory, model, layer), from
  `router_probe_runs/<batch>/router_probe.jsonl`.

## Why the router table takes only the unsuffixed batch directories

`*_pre<sha>` directories are pre-change reruns kept for comparison. Their
readings agree with the current ones except for `rho_ref`/`rho_cal` in the 7th
decimal, so including them would double every model's rows with values that
differ by nothing that matters -- and a later reader would have no way to tell
which row was current. A `_pre<sha>` directory is skipped only when the
unsuffixed directory actually exists beside it, so a batch whose name merely
contains `_pre` is never dropped silently; every skip is printed with its
reason.

## Absolute paths are arguments, not constants

Both source trees live outside the repo under a scratch root that contains an
account name. So the tables record paths RELATIVE to
that root, and the root itself is passed in.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ZLOSS_COLUMNS = [
    "source", "archive_kind", "archive_kind_source", "checkpoint_step", "n_experts", "num_active",
    "num_layers_in_stack", "num_stacks", "layer_idx", "loop_step",
    "z_mean", "z_min", "z_max", "z_std", "frac_z_negative", "n_tokens",
    "z_sq_mean_depth_avg", "probe_seed", "probe_dtype", "forward_dtype", "input_ids_sha8",
    # The corpus is identified by named root + relative path + digest, never by
    # an absolute path: an absolute path here carries the account name into a
    # file that goes into git, and it is also wrong on any other machine.
    "corpus_root", "corpus_rel", "corpus_file_sha256",
]

ROUTER_COLUMNS = [
    "batch_dir", "model_id", "revision", "variant", "step", "layer_idx",
    "n_experts", "top_k", "scoring_order", "combine_weights_normalized_by_model",
    # margin_null quantities: pooled over all probe seeds, one column pair per
    # shuffle scope. The two scopes are different null hypotheses, so they are
    # side by side rather than collapsed.
    "n_tokens", "n_tokens_effective", "qualified", "margin_mean",
    "rho_batch", "rho_ref_batch", "rho_cal_batch",
    "rho_doc", "rho_ref_doc", "rho_cal_doc",
    "rho_ref_normal_prediction",
    # MaxVio, both conventions, averaged over probe seeds with the spread kept
    # visible -- the router rows are PER SEED while the margin rows are pooled,
    # and hiding that difference in grain inside a single mean is how two
    # quantities end up in one row looking like they were measured together.
    "n_probe_seeds",
    "maxvio_count_mean", "maxvio_count_min", "maxvio_count_max",
    "maxvio_mass_mean", "maxvio_mass_min", "maxvio_mass_max",
    "maxvio_mass_model_weights_mean", "maxvio_mass_model_weights_min",
    "maxvio_mass_model_weights_max",
    "probe_dtype", "forward_dtype", "probe_seeds", "snapshot_commit",
    "code_revision_dirty",
]



def _archive_kind(source: str, recorded: str | None) -> tuple[str, str]:
    """(kind, where it came from) for one probe artifact.

    Three kinds, because the recorded `checkpoint_kind` says "trajectory" for
    ALL of them -- pretraining finals, fine-tuned products and mid-training
    archives alike -- so it cannot separate any of them -- a consumer grouping by it silently mixes 3 final
    checkpoints into a 16-point training curve. The file name can separate
    them (`ckpt_` vs `traj_`), so that is used, and WHERE the answer came from
    is recorded next to it rather than left for a reader to guess.
    """
    if source.startswith("traj_"):
        return "trajectory_archive", "source_prefix"
    # BEFORE the plain `ckpt_` test: fine-tuned products are also `ckpt_*`, and
    # checking the shorter prefix first would swallow them into
    # "final_checkpoint". The criteria treat them as a robustness check that
    # explicitly does NOT enter the main judgement, so a table that cannot
    # separate them invites exactly the comparison the criteria forbid.
    if source.startswith("ckpt_sft_"):
        return "sft_checkpoint", "source_prefix"
    if source.startswith("ckpt_"):
        return "final_checkpoint", "source_prefix"
    return (recorded or "unknown"), "checkpoint_kind_field"


def _named_root_path(absolute: str | None, roots: list[tuple[str, Path]]) -> tuple[str, str]:
    """Split an absolute path into (root name, path relative to that root).

    Project rule: artifacts carry a named root plus a relative path, never an
    absolute path. Two reasons, and the second is the one that bites later --
    an absolute path leaks the account name into a file that goes into git,
    and it is simply wrong on any other machine, so a consumer that tries to
    open it gets a missing file rather than a clear "different layout".

    An unrecognised path is returned under the root name "unknown" with the
    BASENAME only, so nothing account-bearing survives even if the roots list
    is incomplete.
    """
    if not absolute:
        return "", ""
    candidate = Path(absolute)
    for name, root in roots:
        try:
            return name, str(candidate.relative_to(root))
        except ValueError:
            continue
    return "unknown", candidate.name


def _assert_no_absolute_paths(path: Path) -> None:
    """Refuse to leave a table containing an account-bearing absolute path.

    Checked on the written file rather than on the rows, because the failure
    being guarded against is what ENDS UP in git -- a column added later, or a
    nested blob serialised by str(), would otherwise reintroduce it silently.
    """
    text = path.read_text()
    for needle in ("/" + "scratch/", "/" + "home/"):
        if needle in text:
            line = next(i for i, l in enumerate(text.splitlines(), 1) if needle in l)
            raise AssertionError(
                f"{path} contains {needle!r} (first at line {line}). Artifacts carry a "
                "named root plus a relative path, never an absolute one."
            )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def zloss_table(zloss_out: Path, roots: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    """One row per (archive, grid cell). Empty input is an error, not an empty table."""
    files = sorted(zloss_out.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no probe artifacts under {zloss_out}")
    out: list[dict[str, Any]] = []
    for path in files:
        for record in _rows(path):
            ts = record.get("token_source") or {}
            # New artifacts already carry the split form; older ones only have
            # the absolute path, which is why the fallback still exists.
            archive_kind, archive_kind_source = _archive_kind(
                path.stem, record.get("checkpoint_kind")
            )
            corpus_root = ts.get("corpus_root")
            corpus_rel = ts.get("corpus_rel")
            if corpus_root is None:
                corpus_root, corpus_rel = _named_root_path(ts.get("corpus_file"), roots)
            for cell in record["by_layer_loop"]:
                out.append({
                    "source": path.stem,
                    "archive_kind": archive_kind,
                    "archive_kind_source": archive_kind_source,
                    "checkpoint_step": record.get("checkpoint_step"),
                    "n_experts": record.get("num_experts"),
                    "num_active": record.get("num_active"),
                    "num_layers_in_stack": record.get("num_layers_in_stack"),
                    "num_stacks": record.get("num_stacks"),
                    "layer_idx": cell["layer_idx"],
                    "loop_step": cell["loop_step"],
                    "z_mean": cell.get("z_mean"),
                    "z_min": cell.get("z_min"),
                    "z_max": cell.get("z_max"),
                    "z_std": cell.get("z_std"),
                    "frac_z_negative": cell.get("frac_z_negative"),
                    "n_tokens": cell.get("n_tokens"),
                    "z_sq_mean_depth_avg": record.get("z_sq_mean_depth_avg"),
                    "probe_seed": record.get("probe_seed"),
                    # New artifacts carry probe_dtype; older ones only the legacy
                    # name. Falling back keeps one column readable across both
                    # rather than leaving it blank for half the table.
                    "probe_dtype": record.get("probe_dtype") or record.get("forward_dtype"),
                    "forward_dtype": record.get("forward_dtype"),
                    "input_ids_sha8": record.get("input_ids_sha8"),
                    "corpus_root": corpus_root,
                    "corpus_rel": corpus_rel,
                    "corpus_file_sha256": ts.get("corpus_file_sha256"),
                })
    return out


def _spread(values: list[float]) -> tuple[Any, Any, Any]:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return None, None, None
    return sum(clean) / len(clean), min(clean), max(clean)


def router_table(runs_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """One row per (batch directory, layer). Returns rows and skipped batches."""
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for batch in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        probe = batch / "router_probe.jsonl"
        if not probe.exists():
            skipped.append(f"{batch.name} (no router_probe.jsonl)")
            continue
        # Pre-change reruns: superseded by the unsuffixed directory beside them.
        if "_pre" in batch.name and (runs_root / batch.name.split("_pre")[0]).is_dir():
            skipped.append(f"{batch.name} (superseded variant)")
            continue
        records = _rows(probe)
        done = json.loads((batch / "DONE.json").read_text()) if (batch / "DONE.json").exists() else {}
        revision = done.get("code_revision") or {}
        args = done.get("args") or {}
        meta = next((r for r in records if r.get("kind") == "adapter_meta"), {})

        by_kind: dict[str, dict[int, list[dict[str, Any]]]] = {}
        for record in records:
            kind = record.get("kind")
            if kind in ("router", "zloss_state", "adapter_router_extra", "margin_null"):
                by_kind.setdefault(kind, {}).setdefault(int(record["layer_idx"]), []).append(record)

        for layer in sorted(by_kind.get("margin_null", {})):
            scopes = {r.get("shuffle_scope"): r for r in by_kind["margin_null"][layer]}
            batch_scope = scopes.get("batch", {})
            doc_scope = scopes.get("document", {})
            router_rows = by_kind.get("router", {}).get(layer, [])
            extra_rows = by_kind.get("adapter_router_extra", {}).get(layer, [])
            cnt = _spread([r.get("maxvio_count") for r in router_rows])
            mass = _spread([r.get("maxvio_mass") for r in router_rows])
            mass_mw = _spread([r.get("maxvio_mass_model_weights") for r in extra_rows])
            rows.append({
                "batch_dir": batch.name,
                "model_id": meta.get("model_id") or args.get("repo"),
                "revision": meta.get("revision") or args.get("revision"),
                "variant": meta.get("variant"),
                "step": args.get("step"),
                "layer_idx": layer,
                "n_experts": batch_scope.get("n_experts") or meta.get("n_experts"),
                "top_k": batch_scope.get("top_k") or meta.get("top_k"),
                "scoring_order": meta.get("scoring_order"),
                "combine_weights_normalized_by_model": meta.get("combine_weights_normalized_by_model"),
                "n_tokens": batch_scope.get("n_tokens"),
                "n_tokens_effective": batch_scope.get("n_tokens_effective"),
                "qualified": batch_scope.get("qualified"),
                "margin_mean": batch_scope.get("margin_mean"),
                "rho_batch": batch_scope.get("rho"),
                "rho_ref_batch": batch_scope.get("rho_ref"),
                "rho_cal_batch": batch_scope.get("rho_cal"),
                "rho_doc": doc_scope.get("rho"),
                "rho_ref_doc": doc_scope.get("rho_ref"),
                "rho_cal_doc": doc_scope.get("rho_cal"),
                "rho_ref_normal_prediction": batch_scope.get("rho_ref_normal_prediction"),
                "n_probe_seeds": len(router_rows),
                "maxvio_count_mean": cnt[0], "maxvio_count_min": cnt[1], "maxvio_count_max": cnt[2],
                "maxvio_mass_mean": mass[0], "maxvio_mass_min": mass[1], "maxvio_mass_max": mass[2],
                "maxvio_mass_model_weights_mean": mass_mw[0],
                "maxvio_mass_model_weights_min": mass_mw[1],
                "maxvio_mass_model_weights_max": mass_mw[2],
                "probe_dtype": batch_scope.get("probe_dtype") or args.get("dtype"),
                "forward_dtype": meta.get("forward_dtype") or batch_scope.get("forward_dtype"),
                "probe_seeds": "|".join(str(x) for x in (args.get("probe_seeds") or [])),
                "snapshot_commit": revision.get("commit"),
                "code_revision_dirty": revision.get("dirty"),
            })
    return rows, skipped


def _write(path: Path, columns: list[str], rows: list[dict[str, Any]], header_note: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        if header_note:
            # A comment line, not a column: consumers must skip lines starting
            # with '#'. It is here because the access restriction travels with
            # the file, not with the message that delivered it.
            handle.write(f"# {header_note}\n")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-scratch-root", required=True,
                        help="Directory holding zloss_probe/ and router_probe_runs/")
    parser.add_argument("--out-dir", required=True, help="Usually stats/collapse_metrics")
    parser.add_argument("--repo-root", default=".", help="Repository root, for the named-root split")
    args = parser.parse_args(argv)

    root = Path(args.probe_scratch_root)
    out_dir = Path(args.out_dir)

    # Longest first: the repo lives UNDER the scratch root, so checking the
    # scratch root first would relativise every repo path against it.
    roots = [("repo_root", Path(args.repo_root).resolve()), ("probe_scratch_root", root.resolve())]

    zrows = zloss_table(root / "zloss_probe" / "out", roots)
    _write(out_dir / "zloss_sensor_cells.csv", ZLOSS_COLUMNS, zrows, None)
    _assert_no_absolute_paths(out_dir / "zloss_sensor_cells.csv")
    print(f"zloss_sensor_cells.csv: {len(zrows)} rows from {len(set(r['source'] for r in zrows))} archives")

    rrows, skipped = router_table(root / "router_probe_runs")
    _write(out_dir / "opensource_router_cells.csv", ROUTER_COLUMNS, rrows,
           "Includes baseline readings (OLMoE / JetMoE / Granite).")
    print(f"opensource_router_cells.csv: {len(rrows)} rows from "
          f"{len(set(r['batch_dir'] for r in rrows))} batches")
    _assert_no_absolute_paths(out_dir / "opensource_router_cells.csv")
    for name in skipped:
        print(f"  skipped: {name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Download raw FineWeb-Edu parquet shards and record their sha256 (source-of-truth manifest).

Started as a small-scale (`sample-10BT`) dev-pipeline verification tool; now also the script
used for the full 100BT pull (`--config sample-100BT`, ~140 shards / ~286GB). At that scale a
single batch job WILL get interrupted (walltime, preemption) before finishing, so this is
resumable by construction (long jobs write their progress in segments):

  <out-dir>/<config>/download_ledger.jsonl   -- one JSON line APPENDED (and fsynced) right after
                                                 each shard's download+hash succeeds. A killed
                                                 and restarted run reads this first and skips
                                                 any shard already recorded here with a matching
                                                 on-disk sha256 -- no re-download, no lost work.
  <out-dir>/<config>/raw_manifest.json       -- rebuilt at the end of every run from the ledger
                                                 (cheap, pure function of the ledger's contents)
                                                 in canonical shard order. This is what
                                                 tokenize_to_bin_idx.py actually reads -- ledger
                                                 format is an internal, append-only implementation
                                                 detail, not a second manifest schema to keep in
                                                 sync by hand.

Usage (run as a module from the repo root -- now imports `src.data._ledger_utils`, an
absolute import like the rest of this directory, see torchtitan_adapter.py's docstring for why
that means `python -m` from the repo root, not bare `python download_shards.py`):
  python -m src.data.download_shards --config sample-10BT --num-shards 2 \
      --out-dir pretrain/data/raw/
  # full 100BT pull, safe to Ctrl-C / walltime-kill and rerun the identical command:
  python -m src.data.download_shards --config sample-100BT --num-shards 999999 \
      --out-dir pretrain/data/raw/

Writes:
  <out-dir>/<config>/NNN_00000.parquet       (as downloaded, filenames from the HF repo)
  <out-dir>/<config>/download_ledger.jsonl   (append-only completion log, see above)
  <out-dir>/<config>/raw_manifest.json       (sha256 + size + source repo/revision per file --
                                              this is the provenance record: every data
                                              shard carries its sha256)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

from src.data._ledger_utils import append_ledger, fsync_path, read_ledger

DATASET_REPO = "HuggingFaceFW/fineweb-edu"


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def list_shard_files(config: str) -> list:
    """List available parquet files for a sample-<N>BT config, in canonical (sorted) order.

    Canonical order = sorted filename order. This becomes part of the deterministic corpus
    order downstream (packed_dataset.py), so it must be stable across runs -- sorting a
    listing from the HF API guarantees that (the API itself doesn't promise order).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = f"sample/{config.split('sample-')[-1]}/"
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    shard_files = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not shard_files:
        raise RuntimeError(f"No parquet files found under {prefix} in {DATASET_REPO}")
    return shard_files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help='e.g. "sample-10BT" (small dev sample)')
    ap.add_argument("--num-shards", type=int, required=True, help="how many shard files to download")
    ap.add_argument("--out-dir", required=True, help="destination directory")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    out_dir = os.path.join(args.out_dir, args.config)
    os.makedirs(out_dir, exist_ok=True)
    ledger_path = os.path.join(out_dir, "download_ledger.jsonl")

    shard_files = list_shard_files(args.config)[: args.num_shards]
    if len(shard_files) < args.num_shards:
        print(
            f"{len(shard_files)} shards exist for config {args.config} "
            f"(requested up to {args.num_shards}) -- downloading all of them",
            file=sys.stderr,
        )

    ledger = read_ledger(ledger_path)
    n_skipped = 0
    for i, rel_path in enumerate(shard_files):
        filename = os.path.basename(rel_path)
        dest_path = os.path.join(out_dir, filename)

        prior = ledger.get(filename)
        if prior is not None and os.path.exists(dest_path) and sha256_file(dest_path) == prior["sha256"]:
            n_skipped += 1
            continue  # already downloaded and verified in a prior (possibly killed) run

        print(f"[{i + 1}/{len(shard_files)}] downloading {rel_path} ...", flush=True)
        local_path = hf_hub_download(
            DATASET_REPO, rel_path, repo_type="dataset", revision=args.revision
        )
        # Materialize a real copy under out_dir (not a symlink into the shared HF cache) so
        # this shard set is a self-contained, independently-hashable artifact.
        import shutil

        shutil.copyfile(local_path, dest_path)
        fsync_path(dest_path)
        digest = sha256_file(dest_path)
        size = os.path.getsize(dest_path)
        print(f"  sha256={digest} size={size}", flush=True)

        entry = {
            "key": filename,
            "filename": filename,
            "source_path": rel_path,
            "sha256": digest,
            "size_bytes": size,
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        append_ledger(ledger_path, entry)  # durable BEFORE moving to the next shard
        ledger[filename] = entry

    if n_skipped:
        print(f"skipped {n_skipped}/{len(shard_files)} shards already completed in a prior run", flush=True)

    # raw_manifest.json is a pure rebuild from the ledger -- same schema as before, still the
    # thing tokenize_to_bin_idx.py reads. Only shards actually requested this run go in, in
    # canonical (sorted-filename) order.
    manifest = {
        "dataset_repo": DATASET_REPO,
        "config": args.config,
        "revision": args.revision,
        "shards": [
            {
                "filename": ledger[os.path.basename(rel_path)]["filename"],
                "source_path": ledger[os.path.basename(rel_path)]["source_path"],
                "sha256": ledger[os.path.basename(rel_path)]["sha256"],
                "size_bytes": ledger[os.path.basename(rel_path)]["size_bytes"],
            }
            for rel_path in shard_files
        ],
    }
    manifest_path = os.path.join(out_dir, "raw_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    fsync_path(manifest_path)
    print(f"wrote {manifest_path} ({len(manifest['shards'])} shards)")


if __name__ == "__main__":
    main()

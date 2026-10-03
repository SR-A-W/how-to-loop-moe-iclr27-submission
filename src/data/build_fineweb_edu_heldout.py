#!/usr/bin/env python
"""Build a fixed, pretraining-never-seen FineWeb-Edu held-out slice for validation loss.

Design: every model reports task_loss on the SAME
held-out slice, sourced from FineWeb-Edu but never used in pretraining -- pretraining used all
140 parquet files of the `sample-100BT` config (confirmed exhaustive: the HF repo lists exactly
140 files under `sample/100BT/`, matching `src/data/artifacts/fw100bt_raw_manifest.json`'s
140-shard record), so a held-out slice cannot come from that config at all
-- there is nothing left in it to hold out.

Source and non-overlap proof
-----------------------------
Candidates are drawn from `sample-350BT`, a config confirmed via the HF API to be an
INDEPENDENTLY drawn sample (same-named files across the two configs have different sha256 --
not a superset relationship), NOT a superset of `sample-100BT`. Independent sampling means
document-level overlap is possible by chance, so this script proves non-overlap explicitly
rather than assuming it:

  1. Download all 140 `sample-100BT` parquet files (the exact ones pretraining used, by
     filename, from raw_manifest.json) and collect every document's `id` field (FineWeb-Edu's
     globally unique per-document UUID, format `<urn:uuid:...>`) into a set. Each parquet is
     deleted immediately after its `id` column is extracted -- this script only needs ~1.6GB of
     UUIDs in memory, not the ~286GB of raw text.
  2. Scan `sample-350BT` files in canonical order, keeping only documents whose `id` is NOT in
     that set, until the target token budget (smollm2 tokenization, matching pretraining's
     tokenizer and per-document framing exactly) is reached.
  3. As a second, independent check (not just trusting the `id` field), the accepted documents'
     `url` field is also cross-checked against the training set's URLs, and each accepted
     document's raw text is sha256'd into the manifest so the exact bytes are auditable later.

Output
------
  <out-dir>/shard_000.bin / .idx      -- Megatron IndexedDataset, smollm2-tokenized, one
                                          "document" per accepted document (packing into
                                          seq_len windows happens at read time, same convention
                                          as pretraining's packed_dataset.py).
  <out-dir>/documents.jsonl           -- one line per accepted document: id, url, dump,
                                          text_sha256, token_count (smollm2, including the
                                          trailing separator) -- the audit trail behind the
                                          non-overlap claim.
  <out-dir>/manifest.json             -- source config/revision, doc count, token count,
                                          tokenizer identity, bin/idx sha256, generating
                                          command, generating git commit.

Usage (module, repo root):
  python -m src.data.build_fineweb_edu_heldout \
      --out-dir _runs/eval/heldout/fineweb_edu_heldout_v1 \
      --target-tokens 3000000 --tokenizer-name smollm2
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys

from src.data._heldout_utils import (
    download_with_timeout as _download_with_timeout_raw,
    git_commit as _git_commit,
    hf_token as _hf_token,
    repo_root as _repo_root,
    write_bin_idx as _write_bin_idx_raw,
)

DATASET_REPO = "HuggingFaceFW/fineweb-edu"
TRAIN_CONFIG = "sample-100BT"
CANDIDATE_CONFIG = "sample-350BT"

# The 140 filenames pretraining actually consumed -- read from the repo's own provenance
# record rather than re-listing sample-100BT (which is redundant: that config has exactly 140
# files and all of them were used, but naming them explicitly from OUR record, not the
# dataset's current listing, is what "the exact files training used" means if the dataset
# repo is ever revised).
TRAIN_MANIFEST_PATH = "src/data/artifacts/fw100bt_raw_manifest.json"


def _download_with_timeout(rel_path: str, token: str, local_dir: str) -> str:
    """This module's `hf_hub_download`, bounded -- see `_heldout_utils.download_with_timeout`
    for why. Thin wrapper fixing `repo_id=DATASET_REPO`, this module's only repo."""
    return _download_with_timeout_raw(DATASET_REPO, rel_path, token, local_dir)


def _train_shard_filenames() -> list:
    path = os.path.join(_repo_root(), TRAIN_MANIFEST_PATH)
    with open(path) as f:
        raw = json.load(f)
    if raw["dataset_repo"] != DATASET_REPO or raw["config"] != TRAIN_CONFIG:
        raise ValueError(
            f"{TRAIN_MANIFEST_PATH} records config {raw['config']!r} of {raw['dataset_repo']!r}, "
            f"expected {TRAIN_CONFIG!r} of {DATASET_REPO!r} -- this script's non-overlap proof "
            "is only valid against the exact shards it names."
        )
    return sorted(s["filename"] for s in raw["shards"])


def build_train_id_set(token: str, tmp_dir: str) -> tuple:
    """Download every `sample-100BT` shard training used, return (id_set, url_set).

    Each shard is deleted right after its `id`/`url` columns are read -- disk footprint stays
    at "one shard at a time" (~2GB), not the full ~286GB. Resumable: `ids_urls_ledger.jsonl`
    gets one line per completed shard, so a run interrupted (timeout, kill, crash) partway
    through 140 shards restarts from where it left off instead of re-downloading everything --
    same pattern as `download_shards.py`'s ledger.
    """
    import pyarrow.parquet as pq

    from src.data._ledger_utils import append_ledger, read_ledger

    filenames = _train_shard_filenames()
    os.makedirs(tmp_dir, exist_ok=True)
    ledger_path = os.path.join(tmp_dir, "ids_urls_ledger.jsonl")
    ledger = read_ledger(ledger_path)

    ids: set = set()
    urls: set = set()
    for filename, entry in ledger.items():
        ids.update(entry["ids"])
        urls.update(entry["urls"])
    n_resumed = len(ledger)
    if n_resumed:
        print(f"resuming: {n_resumed}/{len(filenames)} shards already scanned", file=sys.stderr, flush=True)

    for i, filename in enumerate(filenames):
        if filename in ledger:
            continue
        rel_path = f"sample/{TRAIN_CONFIG.split('sample-')[-1]}/{filename}"
        print(f"[train-ids {i + 1}/{len(filenames)}] {rel_path}", file=sys.stderr, flush=True)
        local_path = _download_with_timeout(rel_path, token, tmp_dir)
        table = pq.read_table(local_path, columns=["id", "url"])
        shard_ids = table.column("id").to_pylist()
        shard_urls = table.column("url").to_pylist()
        ids.update(shard_ids)
        urls.update(shard_urls)
        try:
            os.remove(local_path)
        except OSError:
            pass
        # Storing the full id/url lists (not just counts) is what makes the ledger a real
        # resume point rather than a progress bar -- a restart rebuilds the exact same sets
        # from it without re-downloading.
        append_ledger(ledger_path, {"key": filename, "ids": shard_ids, "urls": shard_urls})
    print(f"train doc ids collected: {len(ids)}", file=sys.stderr, flush=True)
    return ids, urls


def select_heldout_documents(
    token: str, tmp_dir: str, train_ids: set, train_urls: set, tok, target_tokens: int,
) -> list:
    """Scan `sample-350BT` in canonical order, accepting non-overlapping documents until
    `target_tokens` (smollm2-tokenized, this project's exact `encode_document` framing) is
    reached. Returns a list of dicts: id, url, dump, text, token_ids, text_sha256.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    prefix = f"sample/{CANDIDATE_CONFIG.split('sample-')[-1]}/"
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    candidate_files = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not candidate_files:
        raise RuntimeError(f"no parquet files found under {prefix} in {DATASET_REPO}")

    os.makedirs(tmp_dir, exist_ok=True)
    accepted: list = []
    total_tokens = 0
    seen_ids_this_run: set = set()  # guards against duplicate ids within the candidate config

    for rel_path in candidate_files:
        if total_tokens >= target_tokens:
            break
        filename = os.path.basename(rel_path)
        print(f"[select] scanning {rel_path} (have {total_tokens}/{target_tokens} tokens)",
              file=sys.stderr, flush=True)
        local_path = _download_with_timeout(rel_path, token, tmp_dir)
        table = pq.read_table(local_path, columns=["id", "url", "dump", "text"])
        n = table.num_rows
        ids_col = table.column("id").to_pylist()
        urls_col = table.column("url").to_pylist()
        dumps_col = table.column("dump").to_pylist()
        texts_col = table.column("text").to_pylist()
        for i in range(n):
            if total_tokens >= target_tokens:
                break
            doc_id, url, dump, text = ids_col[i], urls_col[i], dumps_col[i], texts_col[i]
            if doc_id in train_ids or doc_id in seen_ids_this_run:
                continue
            if url in train_urls:  # second, independent overlap signal
                continue
            token_ids = tok.encode_document(text)
            accepted.append({
                "id": doc_id,
                "url": url,
                "dump": dump,
                "text": text,
                "token_ids": token_ids,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
            seen_ids_this_run.add(doc_id)
            total_tokens += len(token_ids)
        try:
            os.remove(local_path)
        except OSError:
            pass

    if total_tokens < target_tokens:
        raise RuntimeError(
            f"only accumulated {total_tokens} tokens (< target {target_tokens}) after scanning "
            f"all {len(candidate_files)} candidate files -- either raise more candidate files "
            "or lower --target-tokens"
        )
    print(f"accepted {len(accepted)} documents, {total_tokens} smollm2 tokens", file=sys.stderr)
    return accepted


def write_bin_idx(documents: list, out_dir: str) -> dict:
    """Write shard_000.bin/.idx -- thin wrapper over `_heldout_utils.write_bin_idx`, which
    takes plain token-id lists rather than this module's document dicts."""
    return _write_bin_idx_raw([doc["token_ids"] for doc in documents], out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--tokenizer-name", required=True, choices=["llama3", "smollm2"])
    ap.add_argument("--tmp-dir", default=None, help="scratch download dir (default: <out-dir>/.tmp)")
    args = ap.parse_args()

    from src.data.tokenizer import load_named_tokenizer, manifest_fields

    tok = load_named_tokenizer(args.tokenizer_name)
    token = _hf_token()
    tmp_dir = args.tmp_dir or os.path.join(args.out_dir, ".tmp")

    train_scan_dir = os.path.join(tmp_dir, "train_scan")
    train_ids, train_urls = build_train_id_set(token, train_scan_dir)
    candidate_scan_dir = os.path.join(tmp_dir, "candidate_scan")
    documents = select_heldout_documents(
        token, candidate_scan_dir, train_ids, train_urls, tok,
        args.target_tokens,
    )

    bin_idx_info = write_bin_idx(documents, args.out_dir)

    docs_path = os.path.join(args.out_dir, "documents.jsonl")
    with open(docs_path, "w") as f:
        for doc in documents:
            f.write(json.dumps({
                "id": doc["id"], "url": doc["url"], "dump": doc["dump"],
                "text_sha256": doc["text_sha256"], "token_count": len(doc["token_ids"]),
            }) + "\n")

    total_tokens = bin_idx_info["num_tokens"]
    manifest = {
        # `shards` + `is_complete`: the exact schema `packed_dataset.PackedSeqDataset.refresh()`
        # reads (same as tokenize_to_bin_idx.py's manifests) -- this directory is meant to be
        # used directly as `run_eval.py --val-tokenized-dir <out-dir> --val-seq-len 4096`.
        "shards": [bin_idx_info],
        "is_complete": True,
        "total_docs": bin_idx_info["num_docs"],
        "total_tokens": total_tokens,
        # Provenance, beyond what the reader needs (source, doc/token counts, sha256,
        # generating commit/command -- every data shard carries its sha256).
        "source_dataset_repo": DATASET_REPO,
        "source_config": CANDIDATE_CONFIG,
        "source_revision": "main",
        "excluded_config": TRAIN_CONFIG,
        "excluded_shard_count": len(_train_shard_filenames()),
        "num_documents": len(documents),
        "num_tokens": total_tokens,
        "non_overlap_proof": {
            "method": "per-document FineWeb-Edu `id` (urn:uuid) set membership against all "
                      f"{len(_train_shard_filenames())} training shards, plus an independent "
                      "`url`-field cross-check",
            "train_doc_ids_scanned": len(train_ids),
            "documents_jsonl": "documents.jsonl (id/url/dump/text_sha256/token_count per doc, "
                                "for audit)",
        },
        "tokenizer": manifest_fields(tok),
        "generating_commit": _git_commit(),
        "generating_command": " ".join(sys.argv),
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # The training-shard scan leaves about 12 GB here; with the default --tmp-dir it
    # sits inside --out-dir and would be uploaded along with the slice. Neither scan
    # directory is needed after a successful run (only train_scan resumes, via its
    # ledger, and only while the run is unfinished).
    for scan_dir in (train_scan_dir, candidate_scan_dir):
        if os.path.exists(scan_dir):
            shutil.rmtree(scan_dir)
    print(f"wrote {len(documents)} docs / {total_tokens} tokens to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

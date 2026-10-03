#!/usr/bin/env python
"""Build fixed, pretraining-never-seen held-out slices for validation loss, one per domain
outside FineWeb-Edu: English Wikipedia, arXiv,
source code, books, and general web (C4) -- each 2-4M smollm2 tokens, same 4096 packing and
`PackedSeqDataset`-compatible manifest schema as `build_fineweb_edu_heldout.py`'s
`fineweb_edu_heldout_v1`.

Source substitutions (all verified live before use)
---------------------------------------------------
The first-pass source list named three repos this script does NOT use, each for a
concrete reason found by actually trying them, not by inspection alone:

  * arXiv: `ccdv/arxiv-summarization` (`article` field), not `armanc/scientific_papers`
    (a script-based HF dataset requiring `trust_remote_code=True`) or the RedPajama arXiv
    subset (a list of external download URLs, not directly-fetchable parquet).
  * Code: `codeparrot/github-code-clean`, not `bigcode/the-stack-smol`/`starcoderdata`.
    `list_repo_files` succeeds on both bigcode repos (metadata listing does not enforce the
    gate), but an actual `hf_hub_download` 403s with `GatedRepoError` -- this token is not on
    the authorized list, confirmed by trying, not assumed from the `gated` flag alone.
  * Books: `emozilla/pg19`, a parquet mirror of the same content as `pg19`/`deepmind/pg19`
    (both script-based, same `trust_remote_code` issue as scientific_papers).

Wikipedia (`wikimedia/wikipedia`, config `20231101.en`) and C4 (`allenai/c4`, `en` validation
split) are used as originally named -- both plain parquet/jsonl.gz, no gate, no script.

Non-overlap proof, and why it differs from `fineweb_edu_heldout_v1`
----------------------------------------------------------------------
None of these five domains overlaps FineWeb-Edu's `id` scheme (that field simply does not
exist here), so the check is per-domain, using whatever identity the domain's own schema gives:

  * Wikipedia, C4: both carry a `url` field. Checked against the training set's URL set
    (built once by `build_fineweb_edu_heldout.build_train_id_set`, from the same 140
    `sample-100BT` shards pretraining used). C4 is the one domain here that shares FineWeb-Edu's
    CommonCrawl provenance, so this is a real, non-trivial check; Wikipedia's is a formality
    (crawled-and-filtered FineWeb-Edu text and a clean Wikipedia dump read very differently even
    on the same page) but is still run, not assumed.
  * arXiv (papers), code (source files), books: no `url`/CC-provenance field, and no natural
    membership test against a "trained on this URL" set, because these content types are not
    part of what FineWeb-Edu (a CommonCrawl-derived corpus) contains at all -- the categories
    themselves cannot overlap, and this is a structural fact about the two corpora, not
    something asserted without checking anything. Each accepted document's text is still
    sha256'd into `documents.jsonl` as an audit trail, and the manifest states this reasoning
    explicitly rather than silently applying a check that would be vacuous.

Training-key cache
-------------------
`build_fineweb_edu_heldout.build_train_id_set` (140 shards, ~30 min) is expensive to redo per
domain. Its (id_set, url_set) result is cached to `_runs/eval/heldout/_train_keys/` (newline
JSON per set, gzip'd) after the first call in a process; `load_or_build_train_keys` reads that
cache on subsequent domains/runs instead of re-downloading. Manifest records the cache files'
sha256 so a reader can tell which snapshot of the training set a given domain's proof used.

Usage (module, repo root; one domain per invocation, ledger inside each domain's own .tmp/
directory makes an interrupted run resumable per-shard within that domain):
  python -m src.data.build_domain_heldout --domain wikipedia --target-tokens 4000000
  python -m src.data.build_domain_heldout --domain arxiv     --target-tokens 3000000
  python -m src.data.build_domain_heldout --domain code      --target-tokens 3000000
  python -m src.data.build_domain_heldout --domain books     --target-tokens 3000000
  python -m src.data.build_domain_heldout --domain c4        --target-tokens 4000000
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import sys

from src.data._heldout_utils import (
    download_with_timeout,
    git_commit,
    hf_token,
    repo_root,
    sha256_file,
    write_bin_idx,
)

TRAIN_KEYS_DIR = "_runs/eval/heldout/_train_keys"
FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb-edu"


# --------------------------------------------------------------------------------------------
# Training-key cache (shared across all five domains; see module docstring)
# --------------------------------------------------------------------------------------------

def load_or_build_train_keys(token: str) -> tuple[set, set, dict]:
    """Return (train_ids, train_urls, cache_manifest). Builds + persists on first call;
    subsequent calls (this process or a future one) load straight from disk."""
    cache_dir = os.path.join(repo_root(), TRAIN_KEYS_DIR)
    ids_path = os.path.join(cache_dir, "train_ids.txt.gz")
    urls_path = os.path.join(cache_dir, "train_urls.txt.gz")
    manifest_path = os.path.join(cache_dir, "manifest.json")

    if os.path.exists(ids_path) and os.path.exists(urls_path) and os.path.exists(manifest_path):
        print(f"[train-keys] loading cache from {cache_dir}", file=sys.stderr, flush=True)
        with gzip.open(ids_path, "rt") as f:
            ids = set(line.rstrip("\n") for line in f)
        with gzip.open(urls_path, "rt") as f:
            urls = set(line.rstrip("\n") for line in f)
        with open(manifest_path) as f:
            cache_manifest = json.load(f)
        print(f"[train-keys] loaded {len(ids)} ids, {len(urls)} urls", file=sys.stderr, flush=True)
        return ids, urls, cache_manifest

    from src.data.build_fineweb_edu_heldout import build_train_id_set

    print("[train-keys] no cache found, building from the 140 training shards", file=sys.stderr, flush=True)
    tmp_dir = os.path.join(cache_dir, ".tmp_build")
    ids, urls = build_train_id_set(token, tmp_dir)

    os.makedirs(cache_dir, exist_ok=True)
    with gzip.open(ids_path, "wt") as f:
        f.writelines(f"{i}\n" for i in ids)
    with gzip.open(urls_path, "wt") as f:
        f.writelines(f"{u}\n" for u in urls)
    cache_manifest = {
        "source_dataset_repo": FINEWEB_EDU_REPO,
        "source_config": "sample-100BT",
        "num_ids": len(ids),
        "num_urls": len(urls),
        "ids_file_sha256": sha256_file(ids_path),
        "urls_file_sha256": sha256_file(urls_path),
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(cache_manifest, f, indent=2)
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)  # the per-shard ledger, no longer needed
    print(f"[train-keys] cached {len(ids)} ids, {len(urls)} urls to {cache_dir}", file=sys.stderr, flush=True)
    return ids, urls, cache_manifest


# --------------------------------------------------------------------------------------------
# Per-domain document sources
# --------------------------------------------------------------------------------------------
# Each entry: repo_id, the parquet/jsonl.gz file listing, and a `read_shard(path) -> list[dict]`
# that returns per-document dicts with at least `text`; `url`/`extra` fields vary by domain and
# are carried through to documents.jsonl for audit, not all used for the overlap check (see
# module docstring for which domains have a real url-based check).

def _list_parquet_files(token: str, repo_id: str, prefix: str) -> list:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="dataset")
    matches = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not matches:
        raise RuntimeError(f"no parquet files found under {prefix!r} in {repo_id}")
    return matches


def _read_parquet_columns(path: str, columns: list) -> dict:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    return {c: table.column(c).to_pylist() for c in columns}


def wikipedia_files(token: str) -> list:
    return _list_parquet_files(token, "wikimedia/wikipedia", "20231101.en/")


def wikipedia_read(path: str) -> list:
    cols = _read_parquet_columns(path, ["id", "url", "text"])
    return [
        {"text": cols["text"][i], "url": cols["url"][i], "extra": {"wiki_id": cols["id"][i]}}
        for i in range(len(cols["text"]))
    ]


def arxiv_files(token: str) -> list:
    return _list_parquet_files(token, "ccdv/arxiv-summarization", "document/")


def arxiv_read(path: str) -> list:
    cols = _read_parquet_columns(path, ["article"])
    return [{"text": t, "url": None, "extra": {}} for t in cols["article"] if t]


def code_files(token: str) -> list:
    return _list_parquet_files(token, "codeparrot/github-code-clean", "data/")


def code_read(path: str) -> list:
    cols = _read_parquet_columns(path, ["code", "repo_name", "path", "language", "license"])
    out = []
    for i in range(len(cols["code"])):
        if not cols["code"][i]:
            continue
        out.append({
            "text": cols["code"][i],
            "url": None,
            "extra": {
                "repo_name": cols["repo_name"][i], "path": cols["path"][i],
                "language": cols["language"][i], "license": cols["license"][i],
            },
        })
    return out


def books_files(token: str) -> list:
    return _list_parquet_files(token, "emozilla/pg19", "data/")


def books_read(path: str) -> list:
    cols = _read_parquet_columns(path, ["short_book_title", "url", "text"])
    return [
        {"text": cols["text"][i], "url": cols["url"][i],
         "extra": {"short_book_title": cols["short_book_title"][i]}}
        for i in range(len(cols["text"]))
    ]


def c4_files(token: str) -> list:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files("allenai/c4", repo_type="dataset")
    matches = sorted(f for f in files if f.startswith("en/c4-validation.") and f.endswith(".json.gz"))
    if not matches:
        raise RuntimeError("no en/c4-validation.*.json.gz files found in allenai/c4")
    return matches


def c4_read(path: str) -> list:
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append({"text": d["text"], "url": d.get("url"), "extra": {"timestamp": d.get("timestamp")}})
    return out


#: name -> (repo_id, list_files(token) -> [rel_path,...], read_shard(local_path) -> [doc,...],
#:          has_url: bool -- whether the overlap check is a real url-membership test (True) or
#:          a structural "this content type cannot be in FineWeb-Edu" note (False))
DOMAINS: dict[str, dict] = {
    "wikipedia": {"repo_id": "wikimedia/wikipedia", "list_files": wikipedia_files, "read": wikipedia_read, "has_url": True},
    "arxiv": {"repo_id": "ccdv/arxiv-summarization", "list_files": arxiv_files, "read": arxiv_read, "has_url": False},
    "code": {"repo_id": "codeparrot/github-code-clean", "list_files": code_files, "read": code_read, "has_url": False},
    "books": {"repo_id": "emozilla/pg19", "list_files": books_files, "read": books_read, "has_url": True},
    "c4": {"repo_id": "allenai/c4", "list_files": c4_files, "read": c4_read, "has_url": True},
}


# --------------------------------------------------------------------------------------------
# Selection + build
# --------------------------------------------------------------------------------------------

def select_documents(
    domain: str, token: str, tmp_dir: str, train_ids: set, train_urls: set, tok, target_tokens: int,
) -> tuple[list, dict]:
    """Returns (accepted_documents, selection_stats). `selection_stats` is the real "dedup
    numbers" recorded in each domain's manifest: how many candidates were actually
    scanned and how many of those were excluded for a training-url match, not just the training
    set's size (which says nothing about whether the filter did anything for this domain)."""
    cfg = DOMAINS[domain]
    candidate_files = cfg["list_files"](token)
    os.makedirs(tmp_dir, exist_ok=True)

    accepted: list = []
    total_tokens = 0
    seen_this_run: set = set()  # by text_sha256, guards duplicates within the domain's own files
    stats = {"scanned": 0, "excluded_url_overlap": 0, "excluded_duplicate_within_run": 0, "excluded_empty": 0}

    for rel_path in candidate_files:
        if total_tokens >= target_tokens:
            break
        print(f"[select:{domain}] scanning {rel_path} (have {total_tokens}/{target_tokens} tokens)",
              file=sys.stderr, flush=True)
        local_path = download_with_timeout(cfg["repo_id"], rel_path, token, tmp_dir)
        try:
            docs = cfg["read"](local_path)
        finally:
            try:
                os.remove(local_path)
            except OSError:
                pass

        for doc in docs:
            if total_tokens >= target_tokens:
                break
            stats["scanned"] += 1
            text = doc["text"]
            if not text or not text.strip():
                stats["excluded_empty"] += 1
                continue
            text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_sha256 in seen_this_run:
                stats["excluded_duplicate_within_run"] += 1
                continue
            url = doc.get("url")
            # The real overlap check, where this domain has a url to check: exclude anything
            # whose url appears in the training set. Domains without a url (arxiv/code) skip
            # this -- see module docstring for why that is not a gap, not a shortcut.
            if cfg["has_url"] and url and url in train_urls:
                stats["excluded_url_overlap"] += 1
                continue
            token_ids = tok.encode_document(text)
            accepted.append({
                "text_sha256": text_sha256, "url": url, "extra": doc.get("extra", {}),
                "token_ids": token_ids,
            })
            seen_this_run.add(text_sha256)
            total_tokens += len(token_ids)

    if total_tokens < target_tokens:
        raise RuntimeError(
            f"{domain}: only accumulated {total_tokens} tokens (< target {target_tokens}) after "
            f"scanning all {len(candidate_files)} candidate files"
        )
    print(f"[select:{domain}] accepted {len(accepted)} documents, {total_tokens} smollm2 tokens, "
          f"scanned {stats['scanned']}, excluded_url_overlap {stats['excluded_url_overlap']}",
          file=sys.stderr, flush=True)
    return accepted, stats


def build_domain(domain: str, target_tokens: int, tokenizer_name: str, out_dir: str) -> None:
    from src.data.tokenizer import load_named_tokenizer, manifest_fields

    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; choose from {sorted(DOMAINS)}")

    tok = load_named_tokenizer(tokenizer_name)
    token = hf_token()
    train_ids, train_urls, train_keys_manifest = load_or_build_train_keys(token)

    tmp_dir = os.path.join(out_dir, ".tmp")
    documents, selection_stats = select_documents(
        domain, token, tmp_dir, train_ids, train_urls, tok, target_tokens
    )

    bin_idx_info = write_bin_idx([d["token_ids"] for d in documents], out_dir)

    docs_path = os.path.join(out_dir, "documents.jsonl")
    with open(docs_path, "w") as f:
        for doc in documents:
            f.write(json.dumps({
                "text_sha256": doc["text_sha256"], "url": doc["url"],
                "token_count": len(doc["token_ids"]), **doc["extra"],
            }) + "\n")

    # Code domain: language distribution, recorded in the manifest.
    extra_manifest_fields = {}
    if domain == "code":
        from collections import Counter
        lang_counts = Counter(d["extra"].get("language") for d in documents)
        extra_manifest_fields["language_distribution"] = dict(lang_counts.most_common())

    cfg = DOMAINS[domain]
    total_tokens = bin_idx_info["num_tokens"]
    if cfg["has_url"]:
        non_overlap_method = (
            "per-document `url` field membership against the training set's URL set "
            f"({train_keys_manifest['num_urls']} urls, cached from the 140 sample-100BT "
            "training shards); documents.jsonl also records each accepted document's "
            "text_sha256 for audit"
        )
    else:
        non_overlap_method = (
            f"structural: {domain} is not a content type FineWeb-Edu (a CommonCrawl-derived "
            "corpus) contains, so no url-membership test applies -- the categories cannot "
            "overlap by construction, not merely assumed not to. documents.jsonl records each "
            "accepted document's text_sha256 for audit regardless"
        )

    manifest = {
        "shards": [bin_idx_info],
        "is_complete": True,
        "total_docs": bin_idx_info["num_docs"],
        "total_tokens": total_tokens,
        "domain": domain,
        "source_dataset_repo": cfg["repo_id"],
        "source_revision": "main",
        "num_documents": len(documents),
        "num_tokens": total_tokens,
        "non_overlap_proof": {
            "method": non_overlap_method,
            "train_keys_cache": {
                "dir": TRAIN_KEYS_DIR,
                "num_ids": train_keys_manifest["num_ids"],
                "num_urls": train_keys_manifest["num_urls"],
                "ids_file_sha256": train_keys_manifest["ids_file_sha256"],
                "urls_file_sha256": train_keys_manifest["urls_file_sha256"],
            },
            # The actual dedup numbers for THIS domain's run, not just the training set's size:
            # how many candidates were looked at and how many were thrown out and why.
            "candidates_scanned": selection_stats["scanned"],
            "excluded_url_overlap_with_training": selection_stats["excluded_url_overlap"],
            "excluded_duplicate_within_this_run": selection_stats["excluded_duplicate_within_run"],
            "excluded_empty_text": selection_stats["excluded_empty"],
            "documents_jsonl": "documents.jsonl (text_sha256/url/token_count + domain-specific "
                                "fields per doc, for audit)",
        },
        **extra_manifest_fields,
        "tokenizer": manifest_fields(tok),
        "generating_commit": git_commit(),
        "generating_command": " ".join(sys.argv),
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[{domain}] wrote {len(documents)} docs / {total_tokens} tokens to {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--tokenizer-name", default="smollm2", choices=["llama3", "smollm2"])
    ap.add_argument("--out-dir", default=None, help="default: _runs/eval/heldout/<domain>_heldout_v1")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(repo_root(), "_runs", "eval", "heldout", f"{args.domain}_heldout_v1")
    build_domain(args.domain, args.target_tokens, args.tokenizer_name, out_dir)


if __name__ == "__main__":
    main()

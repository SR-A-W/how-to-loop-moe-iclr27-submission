#!/usr/bin/env python
"""Tokenize raw FineWeb-Edu parquet shards into Megatron bin/idx shards.

Engineering rule (don't reinvent the wheel): the bin/idx writer is NOT reimplemented here. We import
`megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder` from the `megatron-core` pip
package (installed --no-deps; it only needs numpy/torch which we already have) and use it
as-is. This guarantees byte-for-byte the same .bin/.idx format Megatron-Core reads, which is
the whole point of choosing this format.

One raw parquet file -> one bin/idx shard (1:1), each document -> one "document" in the
Megatron index (add_document, not add_item -- keeps document boundaries in the index so
downstream packing can respect them if needed later; we still consume it as a flat token
stream for GPT-style packing, see packed_dataset.py).

Usage (run as a module from the repo root -- see torchtitan_adapter.py docstring for why: this
file uses absolute `src.data.*` imports, which only resolve when the repo root is on
sys.path, which `python -m` from the repo root gives you and bare `python script.py` does not):
  python -m src.data.tokenize_to_bin_idx --raw-dir pretrain/data/raw/sample-10BT \
      --out-dir pretrain/data/tokenized/sample-10BT --tokenizer-name smollm2 --num-workers 4 \
      --max-docs-per-shard 20000   # omit for full shard

Writes per shard:
  <out-dir>/shard_<NNN>.bin / .idx
Writes repeatedly (see --publish-every below):
  <out-dir>/manifest.json   -- sha256 of each .bin/.idx pair + tokenizer identity + doc/token
                                counts + megatron-core version + is_complete. This is the
                                tokenized-shard provenance record (every data shard
                                carries its sha256).

Publish is ATOMIC: `<out-dir>` is only ever a symlink,
swapped into place in one syscall after a batch (bin/idx for every shard in the batch +
manifest.json) is written and fsynced into a private staging directory. Readers never observe a
partially written `<out-dir>` -- either the complete previous publish or the complete new one.
See `_atomic_publish()` docstring for the failure mode this replaces.

Publish is INCREMENTAL, not just once at the end ("train while
downloading/tokenizing"): `--publish-every N` (default 1) atomically republishes
after every N shards complete, each publish's manifest.json marked `is_complete: false` until
the very last one. This lets training start consuming data before the whole ~17-18h tokenize
run finishes -- see packed_dataset.py's "Streaming / incremental publish" docstring section for
what a reader does with a growing, not-yet-complete corpus.

Tokenization itself is RESUMABLE per-shard (100BT: ~140 shards, ~7.5min each at measured
throughput -- a single-process run is ~17-18h, inside a typical 168h job-walltime ceiling but
still long enough on a shared cluster to be worth surviving a kill/requeue).
`<out-dir>.staging/tokenize_ledger.jsonl` gets one fsynced line per shard the moment its bin/idx
are written and verified; a restarted run reads it first and skips any shard whose ledger entry
still matches both the source raw shard's sha256 AND the on-disk bin/idx hashes AND the current
tokenizer's identity. The staging directory is intentionally NOT what gets published directly --
see `_publish_batch()`'s docstring for why publishing a directory that resumable runs keep
mutating would be unsafe.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import glob
import hashlib
import importlib.metadata
import json
import os
import uuid

import numpy as np

from src.data._ledger_utils import append_ledger, fsync_path, read_ledger
from src.data.tokenizer import TOKENIZER_CLASSES, load_named_tokenizer, manifest_fields
from src.metrics.provenance import resolve_code_revision

# Megatron-Core's own bin/idx writer -- see module docstring.
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder, get_bin_path, get_idx_path

# int32 is enough for a 128,256-entry vocab (fits in uint16 even, but Megatron's own default
# for GPT-style corpora is int32 and downstream torch embedding lookups expect signed ints
# anyway -- no reason to be clever here).
TOKEN_DTYPE = np.int32


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_publish(build_dir: str, public_path: str) -> None:
    """Publish `build_dir` as `public_path` via a single atomic symlink swap.

    Why (observed failure): a reader (`PackedSeqDataset`) hit
    `FileNotFoundError` mid-rebuild because the old approach wrote directly into `public_path`,
    which is a multi-second-to-hours window (full 100BT scale) where the directory has some
    files but not others. That case failed loudly. The quieter, worse case: a reader opening
    the directory *between* the new bin/idx being written and manifest.json being overwritten
    would get new token data paired with a stale manifest (or vice versa) -- sha256 checks might
    still individually pass while the pairing is wrong, and "what was fed" must be exactly
    reconstructable after the fact, which a torn read violates silently.

    Scheme: `public_path` is always a symlink to a real, immutable, uniquely-named build
    directory. Republishing = build fully into a NEW directory (readers can't see it, it's not
    at `public_path` yet), then atomically repoint the symlink -- a single `rename()` syscall,
    POSIX-guaranteed atomic. A reader either sees the complete old build or the complete new one,
    never a mix, and never a "doesn't exist" window either (only the very first publish has no
    prior symlink to preserve).

    First-time migration: if `public_path` already exists as a plain directory (pre-dates this
    scheme), it is moved aside to `<public_path>.pre-atomic-publish-<pid>` rather than deleted,
    so nothing already on disk is silently destroyed.

    NOTE: previous build directories (and any moved-aside legacy directory) are intentionally
    NOT auto-deleted after a successful swap -- same reasoning as the failure above: an
    automatic delete of "the old one, we don't need it anymore" is exactly the kind of
    destructive op that should have a human decide when disk pressure actually requires it,
    not run unconditionally on every republish. At toy/dev scale this is a few hundred MB of
    accumulation; at 100BT production scale, space management should fold this cleanup in
    deliberately, not implicitly here.
    """
    build_dir = os.path.abspath(build_dir)
    public_path = os.path.abspath(public_path)
    parent = os.path.dirname(public_path)
    tmp_link = os.path.join(parent, f".{os.path.basename(public_path)}.newlink-{uuid.uuid4().hex[:8]}")
    rel_target = os.path.relpath(build_dir, parent)  # relative symlink: tree stays relocatable
    os.symlink(rel_target, tmp_link)
    try:
        if os.path.lexists(public_path) and not os.path.islink(public_path):
            aside = f"{public_path}.pre-atomic-publish-{uuid.uuid4().hex[:8]}"
            os.rename(public_path, aside)
            print(
                f"moved pre-existing non-symlink {public_path} aside to {aside} "
                "(one-time migration to atomic-publish scheme; not deleted)",
                flush=True,
            )
        os.replace(tmp_link, public_path)  # atomic: renames the symlink itself, single syscall
    except Exception:
        if os.path.lexists(tmp_link):
            os.unlink(tmp_link)
        raise
    fsync_path(parent)


def tokenize_one_shard(parquet_path: str, out_prefix: str, tok,
                        max_docs: int | None, batch_size: int = 512) -> dict:
    import pyarrow.parquet as pq

    builder = IndexedDatasetBuilder(get_bin_path(out_prefix), dtype=TOKEN_DTYPE, multimodal=False)

    pf = pq.ParquetFile(parquet_path)
    num_docs = 0
    num_tokens = 0
    done = False
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        texts = batch.column("text").to_pylist()
        if max_docs is not None:
            remaining = max_docs - num_docs
            if remaining <= 0:
                break
            texts = texts[:remaining]
        encoded = tok.encode_documents_batch(texts)
        for ids in encoded:
            import torch

            builder.add_document(torch.tensor(ids, dtype=torch.int64), [len(ids)])
            num_docs += 1
            num_tokens += len(ids)
        if max_docs is not None and num_docs >= max_docs:
            done = True
            break
    builder.finalize(get_idx_path(out_prefix))

    return {"num_docs": num_docs, "num_tokens": num_tokens, "truncated": max_docs is not None}


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Module-global, set exactly once per PROCESS by _init_worker (either the main process itself,
# for --num-workers=1, or once inside each ProcessPoolExecutor worker process). Not passed as a
# function argument to _tokenize_shard_in_worker because ProcessPoolExecutor pickles every
# argument to every submitted call -- re-pickling/reloading a whole HF tokenizer per SHARD
# (instead of once per WORKER PROCESS) would both be wasteful and risk non-determinism if the
# pickled-and-reloaded tokenizer object ever behaved subtly differently from a freshly
# constructed one. Loading once via the pool's `initializer` avoids both concerns.
_WORKER_TOKENIZER = None  # type: ignore[var-annotated]  # a src.data.tokenizer._NamedBpeTokenizer


def _init_worker(tokenizer_name: str, tokenizer_repo: str | None, tokenizer_revision: str | None) -> None:
    """`ProcessPoolExecutor(initializer=...)` hook: runs exactly once when a worker process
    starts (or, for `--num-workers=1`, is called directly by `main()` -- no pool is spawned in
    that case, see below). `tokenizer_name` selects WHICH tokenizer (see
    `src.data.tokenizer.load_named_tokenizer` -- required, no default); `tokenizer_repo`
    optionally overrides that name's own bundled default directory/Hub repo."""
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = load_named_tokenizer(
        tokenizer_name, repo_id=tokenizer_repo, revision=tokenizer_revision
    )


def _tokenize_shard_in_worker(task: dict) -> dict:
    """One shard's worth of work, runnable either inside a `ProcessPoolExecutor` worker process
    (`--num-workers > 1`) or directly in the main process (`--num-workers == 1`) -- same
    function either way, so parallel and single-process runs are byte-identical on the shards
    they share (verified by test).

    Returns only small, cheaply-picklable values (hashes/counts, never the tokenized tensors
    themselves -- those are written straight to the shard's own bin/idx files on disk inside
    `tokenize_one_shard`), so this is safe to run in a separate OS process without any large
    inter-process data transfer.
    """
    global _WORKER_TOKENIZER
    tok = _WORKER_TOKENIZER
    assert tok is not None, "_init_worker() must run before _tokenize_shard_in_worker()"
    out_prefix = task["out_prefix"]
    stats = tokenize_one_shard(task["parquet_path"], out_prefix, tok, task["max_docs"])
    bin_path, idx_path = get_bin_path(out_prefix), get_idx_path(out_prefix)
    fsync_path(bin_path)
    fsync_path(idx_path)
    return {
        "index": task["index"],
        "shard_id": task["shard_id"],
        "bin_sha256": sha256_file(bin_path),
        "idx_sha256": sha256_file(idx_path),
        "num_docs": stats["num_docs"],
        "num_tokens": stats["num_tokens"],
        "truncated": stats["truncated"],
    }


def _publish_batch(staging_dir, public_out_dir, shards_out, raw_manifest_path, tok, is_complete):
    """Build a fresh manifest.json (cumulative -- all shards done so far, `is_complete` marking
    whether every shard the raw_manifest asked for is present) and atomically publish it plus
    every shard's bin/idx. Called repeatedly (every `--publish-every` shards) so training can
    start consuming before tokenization fully finishes ("train while downloading/tokenizing")
    -- see packed_dataset.py's "Streaming / incremental publish"
    docstring section for what readers do with `is_complete` and how they discover a newly-
    published batch."""
    manifest = {
        "dtype": str(np.dtype(TOKEN_DTYPE)),
        "raw_manifest": os.path.abspath(raw_manifest_path),
        # Pinned deliberately: the bin/idx format is whatever
        # megatron-core's IndexedDatasetBuilder happens to write for this installed version --
        # if a future megatron-core upgrade changes index-format details, this field is the
        # first thing to check when old shards stop reading cleanly.
        "megatron_core_version": importlib.metadata.version("megatron-core"),
        "is_complete": is_complete,
        **manifest_fields(tok),
        # Aggregate convenience fields (the manifest records total tokens alongside tokenizer
        # identity and shard order) --
        # recomputed as a sum every publish, never carried forward/cached, so a partial
        # (is_complete=False) publish's totals are always exactly what's in `shards_out` right
        # now, not a stale count from an earlier, smaller batch.
        "total_docs": sum(s["num_docs"] for s in shards_out),
        "total_tokens": sum(s["num_tokens"] for s in shards_out),
        "shards": shards_out,
    }
    # Promote staging -> a fresh, immutable, uniquely-named build dir. Shard bin/idx files are
    # HARDLINKED (same filesystem, instant, zero extra disk) because they are genuinely
    # write-once per shard_id -- safe to share an inode across every build dir that includes
    # that shard. `manifest.json` is NOT hardlinked -- it is written as an independent file
    # directly into build_dir every publish.
    #
    # BUG CAUGHT BY OUR OWN TESTING (before this ever shipped): the first version of
    # this function did `for name in os.listdir(staging_dir): os.link(...)` unconditionally,
    # which ALSO hardlinked manifest.json. Since staging_dir/manifest.json is the SAME filename
    # reopened with "w" on every publish, and `open(path, "w")` on an existing hardlinked file
    # TRUNCATES THE SHARED INODE IN PLACE (it does not unlink+recreate), every previously
    # "published" build dir's manifest.json silently changed to match the LATEST publish the
    # moment the NEXT batch was written -- completely defeating the "old build dirs are frozen,
    # immutable snapshots" guarantee _atomic_publish() is built on. Caught by hand-verifying a
    # multi-batch incremental publish end to end before trusting the design; confirmed via
    # `stat -c '%h' manifest.json` showing nlink=4 (shared across staging + 3 build dirs) when
    # it should have been nlink=1 per build dir. `tokenize_ledger.jsonl` had the exact same
    # exposure (also reopened with "w"... actually append, but same shared-filename risk) and
    # is additionally not meant for readers at all -- excluded from publish entirely, staging
    # dir is its only home.
    build_dir = f"{public_out_dir}.build-{uuid.uuid4().hex[:8]}"
    os.makedirs(build_dir, exist_ok=True)
    for name in os.listdir(staging_dir):
        if name in ("manifest.json", "tokenize_ledger.jsonl"):
            continue  # manifest written fresh below; ledger is staging-internal, never published
        os.link(os.path.join(staging_dir, name), os.path.join(build_dir, name))

    manifest_path = os.path.join(build_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    fsync_path(manifest_path)

    _atomic_publish(build_dir, public_out_dir)
    print(
        f"published {public_out_dir} -> {os.path.basename(build_dir)} "
        f"({len(shards_out)} shards, is_complete={is_complete})",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", required=True, help="dir with *.parquet from download_shards.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--tokenizer-name", required=True, choices=sorted(TOKENIZER_CLASSES),
        help="which tokenizer, by name. Required, no default -- which tokenizer produced a corpus determines "
        "its vocab_size and its ignition-loss ceiling (ln(vocab_size)); this must never be "
        "picked silently. See src.data.tokenizer.load_named_tokenizer / TOKENIZER_CLASSES.",
    )
    ap.add_argument(
        "--tokenizer-repo", default=None,
        help="OPTIONAL override of --tokenizer-name's own bundled local directory: another "
        "local directory, or an HF Hub repo id. Omit to use that name's repo-bundled default "
        "(zero network/token dependency).",
    )
    ap.add_argument(
        "--tokenizer-revision", default=None,
        help="HF Hub revision, only meaningful when --tokenizer-repo is an actual Hub repo id "
        "(raises if given together with a local-directory --tokenizer-repo).",
    )
    ap.add_argument(
        "--max-docs-per-shard", type=int, default=None,
        help="dev/test knob to cap tokenization volume; omit for full shard (production)",
    )
    ap.add_argument(
        "--publish-every", type=int, default=1,
        help="atomically publish (see _publish_batch) after every N *originally-ordered* shards "
        "complete (a CONTIGUOUS prefix from shard_000 -- see the module docstring on why order, "
        "not completion timing, defines the corpus), so readers can start consuming before the "
        "whole run finishes. Default 1 = publish ASAP as each next-in-line shard becomes ready.",
    )
    ap.add_argument(
        "--num-workers", type=int, required=True,
        help="how many OS processes tokenize shards in parallel. No default -- this changes wall-clock behavior "
        "and CPU/memory footprint on a shared cluster, so it must be an explicit choice, never a "
        "silently-picked one (same 'semantic parameters have no default' discipline as the rest "
        "of this pipeline). 1 = the single-process code path (no pool spawned at all). "
        ">1 spawns a ProcessPoolExecutor; each "
        "worker loads its own tokenizer once (not once per shard). Regardless of --num-workers, "
        "shard output files are always named by ORIGINAL shard index (shard_NNN, never by "
        "completion order) and the manifest's 'shards' list is always built as the contiguous "
        "prefix of originally-ordered shards completed so far -- packed_dataset.py treats "
        "manifest 'shards' order as the canonical, seed-independent corpus order, so that order "
        "must never depend on which worker happens to finish which shard first.",
    )
    args = ap.parse_args()
    if args.num_workers < 1:
        raise SystemExit(f"--num-workers must be >= 1, got {args.num_workers}")

    raw_manifest_path = os.path.join(args.raw_dir, "raw_manifest.json")
    with open(raw_manifest_path) as f:
        raw_manifest = json.load(f)
    n_total = len(raw_manifest["shards"])

    tok = load_named_tokenizer(
        args.tokenizer_name, repo_id=args.tokenizer_repo, revision=args.tokenizer_revision
    )
    # Computed once up front (not per shard): each ledger entry below records exactly which
    # tokenizer.json produced it (each ledger line records tokenizer name + revision +
    # tokenizer.json sha256). `tok_fields["tokenizer_repo"]` (not `tok.repo_id` directly) is
    # what gets stored/compared below -- for a local tokenizer directory `tok.repo_id` is an
    # ABSOLUTE path (account-bearing), while `manifest_fields()` already relativizes it against
    # the repo root (same convention `_publish_batch`'s in-loop manifest uses). The ledger is
    # not always safely outside git (a smoke-data ledger can be committed), so it must never
    # carry an absolute path either.
    tok_fields = manifest_fields(tok)
    tokenizer_json_sha256 = tok_fields["tokenizer_key_file_sha256"]["tokenizer.json"]
    tokenizer_repo_safe = tok_fields["tokenizer_repo"]
    document_framing = tok_fields["document_framing"]

    # Code provenance (every other manifest-producing module records code_revision(); this
    # pipeline does too).
    # resolve_code_revision() (not the bare code_revision()) because a 100BT tokenize run is
    # exactly the long-running-job case its own docstring is about: reading live git HEAD at
    # write time answers "what does the repo say right now", not "what code produced this
    # shard", and those two answers diverge the moment anyone commits into the same clone
    # while the job is running (an observed failure, not hypothetical -- see
    # resolve_code_revision's docstring). Computed once up front and reused for every shard
    # done in THIS invocation; a resumed run started later naturally gets a different value if
    # the code moved on, which is accurate, not a bug.
    code_rev, code_rev_source = resolve_code_revision(paths=("src/data",))

    public_out_dir = os.path.abspath(args.out_dir)
    # STABLE (not random) staging directory: this is what makes tokenization resumable. Readers
    # never see it directly -- only `public_out_dir`, once atomically published (below). It is
    # deliberately mutable and persists across runs so a killed/restarted job can pick up where
    # it left off via `tokenize_ledger.jsonl`.
    staging_dir = f"{public_out_dir}.staging"
    os.makedirs(staging_dir, exist_ok=True)
    ledger_path = os.path.join(staging_dir, "tokenize_ledger.jsonl")
    ledger = read_ledger(ledger_path)

    # shards_out[i] stays None until shard i's record is known (skipped-via-ledger, or freshly
    # tokenized just now) -- filled in by ORIGINAL INDEX, never by arrival order, so a manifest
    # built from a PREFIX of this list is always "shard_000..shard_K in order", regardless of
    # which worker process happened to finish which shard first.
    shards_out: list[dict | None] = [None] * n_total
    n_skipped = 0
    pending_tasks = []
    for i, shard_info in enumerate(raw_manifest["shards"]):
        shard_id = f"shard_{i:03d}"
        out_prefix = os.path.join(staging_dir, shard_id)
        bin_path, idx_path = get_bin_path(out_prefix), get_idx_path(out_prefix)

        prior = ledger.get(shard_id)
        if (
            prior is not None
            and prior["source_raw_sha256"] == shard_info["sha256"]
            # Tokenizer identity must also match: a shard tokenized with the NousResearch
            # mirror and one tokenized with the official meta-llama repo have the same
            # source_raw_sha256 (same input parquet) but are NOT interchangeable output --
            # without this check, switching --tokenizer-repo between runs (exactly the
            # mirror-now / official-later contingency this pipeline is built for) would
            # silently keep serving mirror-tokenized shards forever.
            and prior.get("tokenizer_repo") == tokenizer_repo_safe
            and prior.get("tokenizer_revision") == tok.revision
            # (repo, revision) alone is NOT enough for a LOCAL tokenizer directory: revision
            # is always None there (it only means something for a Hub repo id), so if the
            # bundled tokenizer.json at that same path was ever swapped for a different one
            # (e.g. re-bundled at a newer HF revision, same local path), the check above would
            # see "same repo, same (null) revision" and wrongly consider the identity
            # unchanged -- silently keeping shards tokenized with the OLD file forever. Content
            # hash closes that gap regardless of repo/revision shape.
            and prior.get("tokenizer_json_sha256") == tokenizer_json_sha256
            and os.path.exists(bin_path) and os.path.exists(idx_path)
            and sha256_file(bin_path) == prior["bin_sha256"]
            and sha256_file(idx_path) == prior["idx_sha256"]
        ):
            n_skipped += 1
            shards_out[i] = prior["shard_record"]
            continue  # already tokenized and verified in a prior (possibly killed) run

        pending_tasks.append({
            "index": i,
            "shard_id": shard_id,
            "parquet_path": os.path.join(args.raw_dir, shard_info["filename"]),
            "out_prefix": out_prefix,
            "max_docs": args.max_docs_per_shard,
            "source_raw_sha256": shard_info["sha256"],
        })

    print(
        f"{n_skipped}/{n_total} shards already tokenized in a prior run; "
        f"{len(pending_tasks)} to do now with --num-workers={args.num_workers}",
        flush=True,
    )

    published_prefix_len = 0

    def _contiguous_prefix_len() -> int:
        n = 0
        while n < n_total and shards_out[n] is not None:
            n += 1
        return n

    def _maybe_publish(force_final: bool) -> None:
        nonlocal published_prefix_len
        prefix_len = _contiguous_prefix_len()
        if prefix_len == published_prefix_len:
            return  # nothing new at the front of the line yet
        if not force_final and prefix_len != n_total and (prefix_len - published_prefix_len) < args.publish_every:
            return  # not enough NEW contiguous shards yet to cross the next --publish-every boundary
        is_complete = prefix_len == n_total
        _publish_batch(
            staging_dir, public_out_dir, shards_out[:prefix_len], raw_manifest_path, tok, is_complete,
        )
        published_prefix_len = prefix_len

    def _record_result(task: dict, result: dict) -> None:
        shard_id = task["shard_id"]
        bin_path, idx_path = get_bin_path(task["out_prefix"]), get_idx_path(task["out_prefix"])
        shard_record = {
            "shard_id": shard_id,
            "source_raw_file": os.path.basename(task["parquet_path"]),
            "source_raw_sha256": task["source_raw_sha256"],
            "bin_path": os.path.basename(bin_path),
            "idx_path": os.path.basename(idx_path),
            "bin_sha256": result["bin_sha256"],
            "idx_sha256": result["idx_sha256"],
            "num_docs": result["num_docs"],
            "num_tokens": result["num_tokens"],
            "truncated_for_dev": result["truncated"],
        }
        shards_out[task["index"]] = shard_record
        entry = {
            "key": shard_id,
            "source_raw_sha256": task["source_raw_sha256"],
            "tokenizer_repo": tokenizer_repo_safe,
            "tokenizer_revision": tok.revision,
            "tokenizer_json_sha256": tokenizer_json_sha256,
            "document_framing": document_framing,
            "num_tokens": result["num_tokens"],
            "bin_sha256": result["bin_sha256"],
            "idx_sha256": result["idx_sha256"],
            "wall_time_utc": _utc_now_iso(),
            "git_commit": code_rev["commit"],
            "code_dirty": code_rev["dirty"],
            "code_diff_sha256": code_rev["diff_sha256"],
            "code_revision_source": code_rev_source,
            "shard_record": shard_record,
        }
        append_ledger(ledger_path, entry)  # durable before this shard is considered done
        ledger[shard_id] = entry
        print(
            f"[{task['index'] + 1}/{n_total}] {shard_id} done: docs={result['num_docs']} "
            f"tokens={result['num_tokens']} bin_sha256={result['bin_sha256']}",
            flush=True,
        )

    if args.num_workers == 1:
        # No pool at all -- the exact same call sequence as before --num-workers existed.
        _init_worker(args.tokenizer_name, tok.repo_id, tok.revision)
        for task in pending_tasks:
            print(
                f"[{task['index'] + 1}/{n_total}] tokenizing {task['parquet_path']} "
                f"-> {task['out_prefix']}.bin/.idx",
                flush=True,
            )
            result = _tokenize_shard_in_worker(task)
            _record_result(task, result)
            _maybe_publish(force_final=False)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(args.tokenizer_name, tok.repo_id, tok.revision),
        ) as pool:
            futures = {pool.submit(_tokenize_shard_in_worker, task): task for task in pending_tasks}
            for fut in concurrent.futures.as_completed(futures):
                task = futures[fut]
                result = fut.result()  # re-raises a worker's exception in the main process
                _record_result(task, result)
                # Publish checked after EVERY completion (not just in submission order) because
                # a later-submitted shard can finish before an earlier one; only a growing
                # CONTIGUOUS prefix ever gets published (see _maybe_publish / _contiguous_prefix_len).
                _maybe_publish(force_final=False)

    # Final publish covers any leftover contiguous shards not aligned to a --publish-every
    # boundary. Cheap to call unconditionally even if the in-loop publishing already covered
    # everything -- same content, same manifest, just republishes (harmless, atomic either way).
    _maybe_publish(force_final=True)


if __name__ == "__main__":
    main()

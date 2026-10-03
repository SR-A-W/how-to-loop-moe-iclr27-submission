#!/usr/bin/env python
"""Standalone integrity guard for a finished (or partially finished) parallel tokenize run.

Reads a run's ledger (one or more `tokenize_ledger.jsonl`-shaped JSONL files -- normally just
`tokenize_to_bin_idx.py`'s own staging ledger; `--ledger-path` can be repeated if a future
driver ever splits shard ranges across independently-ledgered processes/nodes instead of this
repo's in-process `ProcessPoolExecutor`), verifies every expected shard is present exactly
once and internally consistent, and only then re-derives `total_tokens` as the sum over
shards -- this script is what decides "safe to trust this run's output", separate from and
independent of `tokenize_to_bin_idx.py`'s own in-loop manifest writing.

Raises (never silently drops/skips/truncates) on ANY of:
  - a missing shard (some expected shard id has no ledger entry at all)
  - a duplicate ledger entry for the same shard id
  - an unexpected shard id (a ledger entry outside the expected shard_000..shard_{N-1} set)
  - a shard whose own two recorded token counts disagree (top-level ledger field vs the
    embedded shard_record -- written together by tokenize_to_bin_idx.py today, so they should
    never drift; this guard is what would catch a future refactor that let them)
  - shards tokenized under more than one distinct tokenizer identity (repo + revision +
    tokenizer.json sha256)
  - shards tokenized with more than one distinct document-framing convention (BOS/EOS
    placement, recorded per-shard by
    `tokenize_to_bin_idx.py` via `src.data.tokenizer.manifest_fields`)
  - (optional, via --expected-tokenizer-json-sha256) every shard's tokenizer.json sha256
    disagreeing with a caller-supplied known-good value -- closes the gap
    where all shards could silently agree with EACH OTHER on a wrong-but-same-shaped tokenizer

Also records per-shard code provenance (git commit / dirty /
diff sha256 / source), matching every other manifest-producing module in this project
(`src.metrics.provenance.resolve_code_revision`) -- NOT required to be uniform across shards,
since a run resumed after the code changed legitimately has different commits per shard.

Writes nothing unless every REQUIRED check passes: any failure raises and no manifest is
written.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json


def load_ledger_records(ledger_paths: list[str]) -> list[dict]:
    """One dict per JSONL line read across all `ledger_paths`, in file-then-line order, with NO
    deduplication. Deliberately NOT `_ledger_utils.read_ledger` (which keeps only the LAST
    entry per key -- the right tool for a resume lookup, the wrong tool here: this guard's
    entire purpose is to NOTICE when more than one entry exists for the same shard, which
    `read_ledger`'s last-wins semantics would silently hide)."""
    records: list[dict] = []
    for path in ledger_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def validate_shard_records(records: list[dict], expected_shard_ids: list[str]) -> list[dict]:
    """Validate `records` (one ledger entry per element, as `load_ledger_records` returns)
    against `expected_shard_ids` (the full, ordered set of shard ids this run was supposed to
    produce: normally `[f"shard_{i:03d}" for i in range(num_shards)]`).

    Returns the records reordered to match `expected_shard_ids` on success. Raises ValueError
    with a specific, actionable message on any integrity violation (see module docstring for
    the full list of checks).
    """
    ids = [r["key"] for r in records]

    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(
            f"duplicate ledger entries for {len(dupes)} shard(s): {dupes} -- "
            f"each shard must have exactly one ledger entry, not applying last-wins here "
            f"(unlike a resume lookup, a guard must not silently paper over this)"
        )

    present = set(ids)
    expected = set(expected_shard_ids)
    missing = sorted(expected - present)
    if missing:
        raise ValueError(
            f"missing ledger entries for {len(missing)} of {len(expected_shard_ids)} "
            f"expected shard(s): {missing} -- refusing to emit a manifest for an incomplete run"
        )
    extra = sorted(present - expected)
    if extra:
        raise ValueError(
            f"ledger has entries for {len(extra)} shard(s) outside the expected set: {extra} "
            f"-- expected exactly {sorted(expected_shard_ids)}"
        )

    by_id = {r["key"]: r for r in records}
    ordered = [by_id[sid] for sid in expected_shard_ids]

    for r in ordered:
        declared = r["num_tokens"]
        embedded = r["shard_record"]["num_tokens"]
        if declared != embedded:
            raise ValueError(
                f"shard {r['key']!r}: ledger's top-level num_tokens ({declared}) != its own "
                f"embedded shard_record num_tokens ({embedded}) -- these are written together "
                f"by tokenize_to_bin_idx.py and must always agree; a mismatch means the ledger "
                f"line is corrupted or was hand-edited"
            )
    return ordered


def _framing_key(framing: dict) -> tuple:
    return tuple(sorted(framing.items()))


def build_manifest(
    ordered_records: list[dict],
    *,
    megatron_core_version: str,
    raw_manifest_path: str,
    is_complete: bool,
    note: str | None = None,
    expected_tokenizer_json_sha256: str | None = None,
) -> dict:
    """Assembles the final manifest dict from already-`validate_shard_records`-checked records
    (must already be in the desired final shard order -- this function does not reorder).

    Tokenizer identity (repo + revision + tokenizer.json sha256) and document framing (BOS/EOS
    convention) must be IDENTICAL across every shard: a run that silently mixed two tokenizer
    identities or framings (e.g. an interrupted run resumed later with a different
    --tokenizer-repo) must not produce a manifest claiming one single identity for the whole
    corpus -- raises instead. Code revision is recorded per-shard in `shards[i]` (a run resumed
    days later, after the code moved on, legitimately has different shards built from different
    commits -- that is accurate, not an error, so it is NOT required to be uniform here).

    `expected_tokenizer_json_sha256`, if given, hard-verifies the run's ACTUAL tokenizer.json
    hash against a caller-supplied known-good value: the
    identity-consistency check above only proves every shard agrees with every OTHER shard, not
    that they agree with the tokenizer that was actually INTENDED (e.g. a same-shaped-but-wrong
    tokenizer.json -- same vocab_size/bos/eos, different actual vocab -- would pass silently
    otherwise). Omit to skip this extra check (matches the pre-audit behavior).
    """
    tok_identities = {
        (r["tokenizer_repo"], r["tokenizer_revision"], r["tokenizer_json_sha256"])
        for r in ordered_records
    }
    if len(tok_identities) > 1:
        raise ValueError(
            f"shards were tokenized with {len(tok_identities)} DIFFERENT tokenizer identities "
            f"(repo, revision, tokenizer.json sha256): {sorted(tok_identities)} -- a single "
            f"manifest cannot represent a corpus tokenized with more than one tokenizer"
        )
    tokenizer_repo, tokenizer_revision, tokenizer_json_sha256 = (
        next(iter(tok_identities)) if tok_identities else (None, None, None)
    )

    if expected_tokenizer_json_sha256 is not None and tokenizer_json_sha256 is not None:
        if tokenizer_json_sha256 != expected_tokenizer_json_sha256:
            raise ValueError(
                f"tokenizer.json sha256 {tokenizer_json_sha256!r} does not match the expected "
                f"value {expected_tokenizer_json_sha256!r} -- every shard AGREES with every "
                f"other shard (that check already passed), but not with the tokenizer this run "
                f"was supposed to use. Refusing to emit a manifest for the wrong tokenizer."
            )

    framings = {_framing_key(r["document_framing"]) for r in ordered_records if "document_framing" in r}
    if len(framings) > 1:
        raise ValueError(
            f"shards were tokenized with {len(framings)} DIFFERENT document-framing conventions: "
            f"{sorted(framings)} -- a single manifest cannot represent a corpus framed two "
            f"different ways (BOS/EOS placement changes where packed_dataset.py's doc "
            f"boundaries actually are)"
        )
    document_framing = dict(next(iter(framings))) if framings else None

    shards = [r["shard_record"] for r in ordered_records]
    total_tokens = sum(s["num_tokens"] for s in shards)
    total_docs = sum(s["num_docs"] for s in shards)

    # Per-shard code provenance (see docstring: legitimately non-uniform across a
    # multi-day-resumed run, so recorded per shard rather than required to be a single value).
    code_revisions_by_shard = {
        r["key"]: {
            "git_commit": r.get("git_commit"),
            "code_dirty": r.get("code_dirty"),
            "code_diff_sha256": r.get("code_diff_sha256"),
            "code_revision_source": r.get("code_revision_source"),
        }
        for r in ordered_records
        if "git_commit" in r
    }

    manifest = {
        "raw_manifest": raw_manifest_path,
        "megatron_core_version": megatron_core_version,
        "tokenizer_repo": tokenizer_repo,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_json_sha256": tokenizer_json_sha256,
        "document_framing": document_framing,
        "is_complete": is_complete,
        "n_shards": len(shards),
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "shards": shards,
    }
    if code_revisions_by_shard:
        manifest["code_revision_by_shard"] = code_revisions_by_shard
    # Optional, freeform, deliberately NOT parsed by anything -- a place to record intent that
    # `raw_manifest`'s path alone doesn't make obvious (e.g. that a run's source is
    # sample-10BT/000_00000.parquet and it is smoke-only -- a bare path to a small dev-scale
    # config could otherwise be mistaken for the start of a real production tokenize run).
    if note is not None:
        manifest["note"] = note
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ledger-path", action="append", required=True, dest="ledger_paths",
        help="tokenize_ledger.jsonl to read; repeat --ledger-path for more than one (e.g. one "
        "per independently-ledgered worker range). At least one required -- this IS the "
        "guard's entire input, no default makes sense.",
    )
    ap.add_argument(
        "--num-shards", type=int, required=True,
        help="how many shards (shard_000..shard_{N-1}) this run was supposed to produce -- "
        "required, no default; this defines what 'complete' means for this run.",
    )
    ap.add_argument("--raw-manifest", required=True, help="path to the raw_manifest.json this run tokenized")
    ap.add_argument("--out", required=True, help="path to write the validated manifest.json to")
    ap.add_argument(
        "--note", default=None,
        help="optional freeform text recorded verbatim as the manifest's top-level 'note' "
        "field -- e.g. to flag a smoke/dev-scale run so it isn't mistaken for a production "
        "one just because its raw_manifest path looks similar. Not parsed by anything.",
    )
    ap.add_argument(
        "--expected-tokenizer-json-sha256", default=None,
        help="optional hard-verify: raise unless every shard's tokenizer.json sha256 equals "
        "this value (not just each other's). An acceptance check should pass this so a "
        "same-shaped-but-wrong tokenizer.json cannot pass silently. Omit to skip.",
    )
    args = ap.parse_args()

    expected_shard_ids = [f"shard_{i:03d}" for i in range(args.num_shards)]
    records = load_ledger_records(args.ledger_paths)
    ordered = validate_shard_records(records, expected_shard_ids)
    manifest = build_manifest(
        ordered,
        megatron_core_version=importlib.metadata.version("megatron-core"),
        raw_manifest_path=args.raw_manifest,
        is_complete=True,
        note=args.note,
        expected_tokenizer_json_sha256=args.expected_tokenizer_json_sha256,
    )
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"OK: {manifest['n_shards']} shards, {manifest['total_tokens']} tokens -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

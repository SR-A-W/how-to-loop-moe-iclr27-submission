#!/usr/bin/env python
"""Re-tokenize an existing probe-corpus JSON (`src/data/probe_corpus.py`'s
40-doc or 500-doc artifacts, or any file in the same shape) with a DIFFERENT tokenizer,
producing a NEW corpus file -- never overwrites/mutates the input: the same documents, re-cut
with a new tokenizer into a new corpus file with its own CORPUS_ID (sha256 registered in
probe_corpus.py); the old corpus stays untouched.

Why this needs its own script rather than just swapping the tokenizer object passed to
`probe_corpus.build_probe_batch()`: that function tokenizes the raw `text` field on the fly,
so a DIFFERENT tokenizer already works against the existing corpus JSON with zero code change
-- the *documents* (real text, provenance, domain balance) are tokenizer-agnostic. What this
script produces instead is a FROZEN, pre-tokenized snapshot (`token_ids` per document, using
the new tokenizer, with the same BOS+doc+EOS framing convention `Llama3Tokenizer.encode_document`
uses) -- useful once a specific new tokenizer (e.g. `smollm2`) is the one actually
being measured against, so every reading is against an exact, recorded set of token ids rather
than "whatever this tokenizer produces today."

Any HF-style tokenizer loadable via `transformers.AutoTokenizer.from_pretrained` works as
`--tokenizer-repo`; nothing here is llama3-specific. Registering a new `CORPUS_ID` constant +
sha256 in `probe_corpus.py` is a separate, deliberately NOT-done-here step -- do that once the
output has been produced with the real target tokenizer.

Output shape: same top-level `{"provenance": ..., "docs": [...]}` as the input, with each doc
dict keeping every original field (`name`/`domain`/`source_index`/`sha256`/`n_chars`/`text` --
unchanged, so the new file stays diffable against the original for the parts that didn't
change) plus two new per-doc fields (`token_ids`, `num_tokens`). `provenance` keeps the
original corpus's provenance dict nested under `source_corpus_provenance`, and adds
`retokenize`: {tokenizer identity, source corpus path + sha256, timestamp, min/max token
counts across the batch}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_relative(path: str | Path) -> str:
    """Repo-root-relative path string for a file inside this repo (no account-bearing
    absolute path in anything the repo ships -- `source_corpus_path` used to record
    `os.path.abspath(corpus_path)` unconditionally, which put the account name in every
    produced corpus's provenance block). Raises if `path` is outside the repo
    rather than falling back to the absolute string -- same reasoning as
    `src.eval.probes.artifact_paths.named_checkpoint`: a silent fallback here is exactly how
    the leak this function exists to prevent would come back on the first corpus file that
    lives somewhere unanticipated."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        raise ValueError(
            f"{resolved} is outside the repo ({_REPO_ROOT}) -- refusing to record its "
            "absolute path in a corpus artifact's provenance block."
        ) from None


def _load_tokenizer(repo_id: str, revision: str | None):
    """Any HF-style tokenizer -- deliberately NOT `src.data.tokenizer.Llama3Tokenizer`
    (which hard-asserts Llama-3's specific vocab_size/bos/eos and would refuse to load
    smollm2 or anything else). BOS/EOS framing below falls back to plain `encode()` with no
    added specials if the loaded tokenizer has neither a bos nor an eos token id."""
    from transformers import AutoTokenizer

    is_local = os.path.isdir(repo_id)
    return AutoTokenizer.from_pretrained(
        repo_id, revision=revision, use_fast=True, local_files_only=is_local,
    )


def _encode_document(tokenizer, text: str) -> list[int]:
    """Mirrors `src.data.tokenizer._NamedBpeTokenizer.encode_document`'s framing convention
    exactly (kept in sync deliberately -- see that method's docstring for the full
    rationale): BOS + text + EOS when bos_id != eos_id (Llama-3's case); text + ONE trailing
    separator, no leading one, when bos_id == eos_id (SmolLM2's case: a document gets exactly
    one separator, at the end). Falls back to the tokenizer's own default special-token
    handling if it has neither a bos nor an eos id at all."""
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if bos_id is not None and eos_id is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if bos_id == eos_id:
            return ids + [eos_id]
        return [bos_id] + ids + [eos_id]
    return tokenizer.encode(text, add_special_tokens=True)


def _framing_label(tokenizer) -> str:
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if bos_id is None or eos_id is None:
        return "tokenizer default add_special_tokens=True (no separate bos AND eos id)"
    if bos_id == eos_id:
        return "text_ids + eos_id (single trailing separator, no leading one -- bos==eos)"
    return "bos_id + text_ids + eos_id"


def retokenize_corpus(corpus_path: str, tokenizer_repo: str, tokenizer_revision: str | None) -> dict:
    """Pure function (no file I/O for the output) so it's directly unit-testable: reads
    `corpus_path`, tokenizes every doc's `text` with the given tokenizer, returns the new
    corpus dict. Raises if any document tokenizes to 0 tokens (would silently produce an
    unusable probe sample -- same "raise, don't degrade" discipline as `probe_corpus.py`'s
    `build_probe_batch`)."""
    with open(corpus_path) as f:
        source = json.load(f)
    if not isinstance(source, dict) or "docs" not in source:
        raise ValueError(f"{corpus_path} is not a prepare_corpus.py-shaped artifact (missing 'docs')")

    tok = _load_tokenizer(tokenizer_repo, tokenizer_revision)
    tokenizer_is_local = os.path.isdir(tokenizer_repo)

    new_docs = []
    token_counts = []
    for d in source["docs"]:
        ids = _encode_document(tok, d["text"])
        if len(ids) == 0:
            raise ValueError(f"doc {d.get('name', '?')!r} tokenized to 0 tokens with {tokenizer_repo}")
        token_counts.append(len(ids))
        new_docs.append({**d, "token_ids": ids, "num_tokens": len(ids)})

    retokenize_meta = {
        "tokenizer_repo": tokenizer_repo,
        "tokenizer_source": "local" if tokenizer_is_local else "hf_hub",
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_class": type(tok).__name__,
        "tokenizer_vocab_size": len(tok),
        "source_corpus_path": _repo_relative(corpus_path),
        "source_corpus_sha256": _sha256_file(corpus_path),
        "retokenized_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_docs": len(new_docs),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "bos_eos_framing": _framing_label(tok),
    }

    return {
        "provenance": {
            "source_corpus_provenance": source.get("provenance", {}),
            "retokenize": retokenize_meta,
        },
        "docs": new_docs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-path", required=True, help="input probe-corpus JSON (prepare_corpus.py shape)")
    ap.add_argument("--tokenizer-repo", required=True, help="local dir or HF Hub repo id -- the NEW tokenizer")
    ap.add_argument("--tokenizer-revision", default=None)
    ap.add_argument("--out", required=True, help="output path -- must NOT equal --corpus-path")
    args = ap.parse_args()

    if os.path.abspath(args.out) == os.path.abspath(args.corpus_path):
        raise SystemExit("--out must not equal --corpus-path -- this script never overwrites its input")
    if os.path.exists(args.out):
        raise SystemExit(f"--out {args.out!r} already exists -- refusing to overwrite; pick a new path")

    result = retokenize_corpus(args.corpus_path, args.tokenizer_repo, args.tokenizer_revision)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    meta = result["provenance"]["retokenize"]
    print(
        f"OK: {meta['n_docs']} docs re-tokenized with {args.tokenizer_repo} "
        f"(min={meta['min_tokens']} max={meta['max_tokens']} tokens) -> {args.out}"
    )


if __name__ == "__main__":
    main()

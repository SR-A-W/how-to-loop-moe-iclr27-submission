"""Named tokenizer wrappers: SmolLM2 (used by the paper's models) and Llama-3.

We don't reimplement BPE. We load the actual tokenizer.json (HF PreTrainedTokenizerFast)
via transformers, which guarantees byte-identical vocab/merges to the real thing, and
hard-assert each tokenizer's expected vocab size and BOS/EOS ids.

The SmolLM2 tokenizer files are bundled in `src/data/artifacts/tokenizer/smollm2/` (see
its PROVENANCE.md), so tokenization needs no network access and no HF token.

The Llama-3 tokenizer (vocab 128,256; BOS 128000, EOS 128001; the tokenizer of LT2,
arXiv 2605.20670) is NOT bundled: its files are covered by Meta's license. To use it,
obtain the files from meta-llama/Meta-Llama-3-8B (after accepting the license) and pass
their local directory or Hub repo id explicitly as `repo_id` (e.g. `--tokenizer-repo`).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

# Relative to this file's own location, not to cwd -- correct regardless of where the caller's
# cwd is (unlike a cwd-relative path, this survives being invoked from any directory, matching
# how every other path-sensitive thing in this pipeline is deliberately made invocation-
# location-independent).
_TOKENIZER_ARTIFACTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "tokenizer")
)
EXPECTED_VOCAB_SIZE = 128256
EXPECTED_BOS_ID = 128000
EXPECTED_EOS_ID = 128001

# SmolLM2 (the ablation family's tokenizer, replacing Llama-3). Bundled the same way as llama3 -- see
# src/data/artifacts/tokenizer/smollm2/PROVENANCE.md for source/revision/sha256.
SMOLLM2_LOCAL_TOKENIZER_DIR = os.path.join(_TOKENIZER_ARTIFACTS_DIR, "smollm2")
SMOLLM2_EXPECTED_VOCAB_SIZE = 49152
# SmolLM2 (a GPT-2-style BPE tokenizer) uses ONE token, id 0 (`<|endoftext|>`), for BOTH
# roles -- unlike Llama-3's separate bos(128000)/eos(128001). Not a bug in the bundled
# files; see PROVENANCE.md's "Verified fields" section. `encode_document` below therefore
# frames a SmolLM2 document as text + ONE trailing id 0 (no leading one).
SMOLLM2_EXPECTED_BOS_ID = 0
SMOLLM2_EXPECTED_EOS_ID = 0

# Files whose sha256 we pin per-run into the shard manifest, so that a one-time diff against
# the official repo's files can prove mirror==official without re-tokenizing.
TOKENIZER_KEY_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_local_dir(repo_id: str) -> bool:
    return os.path.isdir(repo_id)


def read_tokenizer_provenance_json(repo_id: str) -> dict | None:
    """Reads `<repo_id>/provenance.json` if `repo_id` is a local directory that has one --
    `None` otherwise (a non-local repo_id, or a local dir without one).

    Machine-readable counterpart to the hand-written `PROVENANCE.md` next to it:
    `PROVENANCE.md` stays for humans, this file is what
    lets `manifest_fields()` embed the ORIGINAL HF Hub revision (e.g. SmolLM2's
    `effd688a12921b4cc83e3312b6feb579f70f9c71`) into a run's manifest without hand-parsing
    prose. The two files must agree (repo id, revision, and per-file sha256).
    """
    if not _is_local_dir(repo_id):
        return None
    path = os.path.join(repo_id, "provenance.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def tokenizer_file_sha256(repo_id: str, revision: str | None = None) -> dict:
    """sha256 of each file in TOKENIZER_KEY_FILES. For a local directory (the default, see
    module docstring), reads files directly off disk -- no network involved. For an actual HF
    repo id, fetches via huggingface_hub (from cache if present) -- same mechanism transformers
    itself uses, not reimplemented."""
    out = {}
    if _is_local_dir(repo_id):
        for filename in TOKENIZER_KEY_FILES:
            out[filename] = _sha256_file(os.path.join(repo_id, filename))
        return out
    from huggingface_hub import hf_hub_download

    for filename in TOKENIZER_KEY_FILES:
        local_path = hf_hub_download(repo_id, filename, revision=revision)
        out[filename] = _sha256_file(local_path)
    return out


@dataclass
class TokenizedDoc:
    ids: list  # python ints, BOS + text tokens + EOS


class _NamedBpeTokenizer:
    """Shared implementation behind `Llama3Tokenizer`/`SmolLM2Tokenizer`: load once via
    `transformers.AutoTokenizer`, hard-assert this specific tokenizer's expected vocab/bos/eos
    (subclasses set the three `_EXPECTED_*` class attributes), and encode documents with
    BOS/EOS framing.

    Factored out when SmolLM2 became a second,
    independently-loadable tokenizer needing the exact same shape of wrapper -- the
    per-tokenizer identity (which vocab size, which bos/eos ids, which bundled directory) is
    the only thing that differs, so it lives on the subclass, not duplicated per class.
    """

    _EXPECTED_VOCAB_SIZE: int
    _EXPECTED_BOS_ID: int
    _EXPECTED_EOS_ID: int
    _DEFAULT_REPO: str
    _RECIPE_LABEL: str  # human-readable name for the error message, e.g. "Llama-3 tiktoken"

    def __init__(self, repo_id: str | None = None, revision: str | None = None):
        from transformers import AutoTokenizer

        if repo_id is None:
            repo_id = self._DEFAULT_REPO
        if repo_id is None:
            raise ValueError(
                f"{self._RECIPE_LABEL}: no tokenizer files are bundled with this repository. "
                "Obtain them from the official source (for Llama-3: meta-llama/Meta-Llama-3-8B, "
                "after accepting Meta's license) and pass their local directory or Hub repo id "
                "explicitly (e.g. --tokenizer-repo)."
            )
        self.is_local = _is_local_dir(repo_id)
        if self.is_local and revision is not None:
            raise ValueError(
                f"repo_id={repo_id!r} is a local directory -- `revision` only makes sense for "
                f"an HF Hub repo id, got revision={revision!r}. Leave it unset for local paths."
            )

        self.repo_id = repo_id
        self.revision = revision
        # local_files_only=True for a local dir: belt-and-suspenders zero-network guarantee
        # (from_pretrained already treats an existing local directory as local automatically,
        # but this makes the "no network, ever, for this path" property explicit and enforced
        # rather than incidental).
        self._tok = AutoTokenizer.from_pretrained(
            repo_id, revision=revision, use_fast=True, local_files_only=self.is_local
        )

        # NOTE: `self._tok.vocab_size` on a fast tokenizer can be the BASE vocab size (e.g.
        # 128,000 for Llama-3's raw tiktoken BPE ranks, EXCLUDING 256 added special tokens) --
        # `len(self._tok)` is base + added, which is the number that matters here (it's the
        # dimension our model's embedding/output layer must match). Caught by this exact
        # assertion the first time this mattered (Llama-3) -- `vocab_size` alone would have
        # silently under-counted by 256 and no downstream code would have complained until a
        # >=128000 token id showed up out-of-range, which could be a very rare token late in a
        # real corpus. (SmolLM2's `vocab_size` attribute and `len()` happen to already agree --
        # see its PROVENANCE.md -- but `len()` is still what's checked, not assumed safe because
        # this particular tokenizer's numbers happen to match.)
        total_vocab_size = len(self._tok)
        if total_vocab_size != self._EXPECTED_VOCAB_SIZE:
            raise ValueError(
                f"tokenizer len={total_vocab_size} (base vocab_size={self._tok.vocab_size}), "
                f"expected {self._EXPECTED_VOCAB_SIZE} ({self._RECIPE_LABEL}). "
                f"Refusing to silently proceed."
            )
        if self._tok.bos_token_id != self._EXPECTED_BOS_ID or self._tok.eos_token_id != self._EXPECTED_EOS_ID:
            raise ValueError(
                f"tokenizer bos/eos ids = {self._tok.bos_token_id}/{self._tok.eos_token_id}, "
                f"expected {self._EXPECTED_BOS_ID}/{self._EXPECTED_EOS_ID}. Refusing to silently proceed."
            )
        self.bos_id = self._tok.bos_token_id
        self.eos_id = self._tok.eos_token_id
        self.vocab_size = total_vocab_size

    def encode_document(self, text: str) -> list:
        """Frame one document's token ids per this tokenizer's convention.

        Llama-3 (bos_id != eos_id): BOS + text tokens + EOS.

        SmolLM2 (bos_id == eos_id, both `<|endoftext|>`): text tokens + ONE trailing
        separator, NO leading one (each SmolLM2 document gets one id 0 at the end, none at the
        start). Adding the same id at BOTH ends (the original, WRONG version of this method,
        before this fix) would have put
        two separators per document instead of the one the recipe specifies -- caught before
        any real corpus other than the since-superseded `pile10k_d50_b500_s512.smollm2.json`
        (v1, replaced by v2) consumed it. The `bos_id == eos_id` check is what decides which
        branch runs -- not a per-name lookup -- so a third tokenizer with the same "shared
        BOS/EOS token" property gets the correct single-trailing-separator framing
        automatically, and one with genuinely distinct bos/eos ids gets Llama-3's framing,
        without this method needing to know tokenizer names at all.
        """
        ids = self._tok.encode(text, add_special_tokens=False)
        if self.bos_id == self.eos_id:
            return ids + [self.eos_id]
        return [self.bos_id] + ids + [self.eos_id]

    def encode_documents_batch(self, texts: list) -> list:
        """Batched variant of encode_document for throughput."""
        batch = self._tok(texts, add_special_tokens=False)["input_ids"]
        if self.bos_id == self.eos_id:
            return [ids + [self.eos_id] for ids in batch]
        return [[self.bos_id] + ids + [self.eos_id] for ids in batch]


class Llama3Tokenizer(_NamedBpeTokenizer):
    """Thin wrapper: load once, encode documents with BOS/EOS framing, hard-asserting the
    Llama-3 recipe's exact vocab/bos/eos (128,256 / 128000 / 128001).

    The files are not bundled (Meta's license), so `repo_id` is required: a local directory
    or an HF Hub repo id -- `AutoTokenizer.from_pretrained` handles both transparently.
    """

    _EXPECTED_VOCAB_SIZE = EXPECTED_VOCAB_SIZE
    _EXPECTED_BOS_ID = EXPECTED_BOS_ID
    _EXPECTED_EOS_ID = EXPECTED_EOS_ID
    _DEFAULT_REPO = None
    _RECIPE_LABEL = "Llama-3 tiktoken, vocabulary 128,256"


class SmolLM2Tokenizer(_NamedBpeTokenizer):
    """Same shape as `Llama3Tokenizer`, for the SmolLM2 tokenizer used by the ablation family.
    Hard-asserts vocab_size 49,152 and bos==eos==0 (`<|endoftext|>` -- see
    `src/data/artifacts/tokenizer/smollm2/PROVENANCE.md`'s "Verified fields" section for why
    bos and eos are the same id for this tokenizer, not an error)."""

    _EXPECTED_VOCAB_SIZE = SMOLLM2_EXPECTED_VOCAB_SIZE
    _EXPECTED_BOS_ID = SMOLLM2_EXPECTED_BOS_ID
    _EXPECTED_EOS_ID = SMOLLM2_EXPECTED_EOS_ID
    _DEFAULT_REPO = SMOLLM2_LOCAL_TOKENIZER_DIR
    _RECIPE_LABEL = "SmolLM2 tokenizer, vocabulary 49,152"


#: Name -> tokenizer class, for `load_named_tokenizer` below. The two names a caller may pick
#: between -- deliberately a closed set, not "any HF repo id", so a typo in a name is a
#: KeyError-shaped `ValueError` immediately rather than a tokenizer that loads successfully but
#: is not the one anybody meant.
TOKENIZER_CLASSES: dict[str, type] = {"llama3": Llama3Tokenizer, "smollm2": SmolLM2Tokenizer}


def load_named_tokenizer(
    name: str, *, repo_id: str | None = None, revision: str | None = None
) -> "_NamedBpeTokenizer":
    """Select a tokenizer BY NAME -- `name` is REQUIRED, no default. A silent default here is
    exactly the mistake the "semantic parameters have no defaults" rule exists to prevent:
    which tokenizer a run uses changes its vocab_size, its ignition-loss ceiling
    (`ln(vocab_size)`), and whether an existing bin/idx corpus is
    even readable against it -- getting this wrong should be impossible to do by omission.

    `name` must be a key of `TOKENIZER_CLASSES` (currently `"llama3"` or `"smollm2"`).
    `repo_id`/`revision` are optional overrides (e.g. a not-yet-bundled Hub revision); omitted,
    each name's own bundled local directory is used.
    """
    if name not in TOKENIZER_CLASSES:
        raise ValueError(f"unknown tokenizer name {name!r}; choose from {sorted(TOKENIZER_CLASSES)}")
    cls = TOKENIZER_CLASSES[name]
    kwargs: dict = {}
    if repo_id is not None:
        kwargs["repo_id"] = repo_id
    if revision is not None:
        kwargs["revision"] = revision
    return cls(**kwargs)


_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def manifest_fields(tok: "_NamedBpeTokenizer") -> dict:
    """Fields to embed in shard manifests so tokenization is traceable to an exact tokenizer.

    Includes per-file sha256 of the key tokenizer files actually used -- this is the record
    that makes the mirror-repo/bundled-file usage auditable and lets a later official-vs-mirror
    hash diff decide "same, keep the data" vs "different, re-tokenize" without guessing which
    files/revision were in play.

    `tokenizer_repo`: for a local directory, this is recorded as a REPO-ROOT-RELATIVE path
    (e.g. "src/data/artifacts/tokenizer/smollm2"), not an absolute filesystem path -- an absolute
    path is machine-specific and meaningless on anyone else's clone of this repo;
    the relative path is a stable identity since the bundled files live at a fixed location in
    version control. For an actual HF Hub repo id, recorded as-is (already portable).
    """
    if tok.is_local:
        try:
            tokenizer_repo_field = os.path.relpath(os.path.abspath(tok.repo_id), _REPO_ROOT)
        except ValueError:
            tokenizer_repo_field = tok.repo_id  # different drive on Windows etc.; fall back
    else:
        tokenizer_repo_field = tok.repo_id
    provenance = read_tokenizer_provenance_json(tok.repo_id)
    return {
        "tokenizer_repo": tokenizer_repo_field,
        "tokenizer_source": "local_bundled" if tok.is_local else "hf_hub",
        "tokenizer_revision": tok.revision,
        # The ORIGINAL HF Hub revision this bundled local directory was fetched from (e.g.
        # SmolLM2's effd688a12921b4cc83e3312b6feb579f70f9c71) -- distinct from
        # `tokenizer_revision` above, which is the revision PASSED TO `from_pretrained` right
        # now (always None for a local dir; that concept doesn't apply there). `None` when
        # there's no provenance.json.
        "tokenizer_source_revision": provenance["revision"] if provenance else None,
        "vocab_size": tok.vocab_size,
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "tokenizer_key_file_sha256": tokenizer_file_sha256(tok.repo_id, tok.revision),
        # Determined from THIS loaded tokenizer's own bos_id/eos_id, not a per-name constant
        # (the document-framing convention `encode_document` actually applies would otherwise
        # be documented only in prose, nowhere machine-readable). Mirrors `_NamedBpeTokenizer.encode_document`'s own branch exactly:
        # bos_id == eos_id -> no leading token, one trailing separator (SmolLM2's case);
        # otherwise -> BOS prepended + EOS appended (Llama-3's case). If a future tokenizer
        # class ever changes that branch, this field is computed FROM the same two ids, not
        # duplicated logic that could drift out of sync.
        "document_framing": {
            "bos_prepended": tok.bos_id != tok.eos_id,
            "eos_appended": True,
            "eos_id": tok.eos_id,
        },
    }

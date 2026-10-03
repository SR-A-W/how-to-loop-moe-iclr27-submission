"""Deterministic, resumable, seed-driven packed-sequence dataset over Megatron bin/idx shards.

Design (determinism and provenance are hard requirements):

1. Canonical corpus order is FIXED and NOT seed-dependent: shards in manifest order, documents
   within a shard in on-disk order (the order tokenize_to_bin_idx.py wrote them, which is the
   parquet row order, which is the order download_shards.py listed files in -- sorted filenames).
   This gives a single virtual concatenated token stream, sliced into non-overlapping
   `seq_len`-token windows ("packed sequences", index 0..num_sequences-1). This slicing is also
   seed-independent -- packing is a deterministic function of the corpus alone.

2. What the seed controls is the ORDER packed sequences are fed to training. We do NOT store a
   giant training-length permutation. Instead, per-epoch permutations are generated on demand:
       epoch_perm(seed, epoch) = numpy.random.default_rng(seed + epoch).permutation(num_sequences)
   This is a pure function of (seed, epoch) -- regenerating it is a few numpy calls, not an I/O
   or replay operation. Given any training step, we can compute (epoch, within_epoch_index)
   from pure arithmetic and regenerate exactly that epoch's permutation. This is what makes
   "resume/replay from an arbitrary step" O(1)-ish instead of "replay everything from step 0".

3. `locate(logical_seq_idx)` answers "which shard, which document, which byte offset does this
   packed sequence start at" -- i.e. which shard and offset the current step is consuming.
   Training loop should log `locate(...)` for the first sequence of each
   step's batch.

ASSUMPTION: "seed determines order" is implemented as document/packed-sequence
-level shuffling (standard practice, e.g. Megatron/GPT-NeoX-style), not intra-document token
shuffling (which would break language modeling). If a different granularity was intended,
this is the place to change it -- the rest of the pipeline doesn't care.

## Streaming / incremental publish ("train while downloading/tokenizing")

`tokenize_to_bin_idx.py` can now publish `manifest.json` incrementally -- every `--publish-every`
shards, not just once at the end -- with an `is_complete: bool` field marking whether every shard
tokenize_to_bin_idx.py was asked to produce is actually present yet. This lets a training run
start consuming data before the full 100BT corpus finishes tokenizing (a small smoke run only
needs the first batch).

**What this does and does NOT preserve, stated plainly**: `num_sequences` (and therefore
`epoch_permutation`) is always computed from whatever's CURRENTLY published. Two different
constructions of `PackedSeqDataset` against manifests with different shard counts get
DIFFERENT epoch-0 permutations -- `numpy.random.default_rng(seed).permutation(N)` is not a
stable prefix as N grows, it's an unrelated shuffle. So:
- **Within one fixed, unchanging manifest** (`is_complete: true`, or a `PackedSeqDataset` that
  never calls `refresh()`): the existing "same seed -> bit-identical order" guarantee holds
  exactly, unchanged.
- **Across a `refresh()` that grows the corpus** (see `on_boundary="wait"` below): order is
  reproducible only relative to that exact sequence of manifest snapshots, not against a
  fresh construction pointed at the eventual final manifest. This is a deliberate, LOUD (never
  silent -- see `CorpusBoundaryError`/log lines below) tradeoff for early/streaming runs. A
  full production run should either wait for `is_complete: true` before starting, or
  use `on_boundary="raise"` and let the trainer decide how to handle a pause, not `"wait"`.

`ensure_available(global_pos, on_boundary, ...)` is the boundary check: called by
`torchtitan_adapter.py` before serving a position beyond the currently-known `num_sequences`.
`on_boundary` is required, no default (semantic parameters have no defaults, same as `seed`/`seq_len`) --
"wait" and "raise" are both real, supported behaviors with different tradeoffs, not a fallback
one of them should silently become.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from megatron.core.datasets.indexed_dataset import IndexedDataset, get_bin_path, get_idx_path


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Location:
    shard_id: str
    doc_idx_in_shard: int
    byte_offset_in_bin: int
    token_offset_in_doc: int


class CorpusBoundaryError(Exception):
    """Raised (on_boundary="raise") when training needs a position beyond what's currently
    published, but more shards are still expected (manifest's is_complete is False). Not a bug
    -- it's the explicit, non-silent signal the streaming-publish design calls for."""


class CorpusExhaustedError(Exception):
    """Raised when training needs a position beyond num_sequences AND the manifest says
    is_complete=True -- there is no more data coming, ever, for this corpus. A real end
    condition, distinct from CorpusBoundaryError's "not yet, check back" semantics."""


class PackedSeqDataset:
    """seed and seq_len are required -- no defaults (semantic parameters have no defaults:
    silently defaulting seq_len would be exactly the kind of bug
    that ships a training run on the wrong context length)."""

    def __init__(self, tokenized_dir: str, seq_len: int, seed: int, verify_sha256: bool = True):
        self.tokenized_dir = tokenized_dir
        self.seq_len = seq_len
        self.seed = seed
        self.verify_sha256 = verify_sha256

        self._datasets: dict[str, IndexedDataset] = {}
        # cumulative token offsets, flattened across all shards, one entry per document,
        # entry i = token offset at which document i begins in the virtual corpus. Plain lists
        # (not numpy arrays) so `refresh()` can append cheaply -- bisect works fine on a list.
        self._doc_shard_id: list = []
        self._doc_idx_in_shard: list = []
        self._doc_cum_start: list = []
        self._loaded_shard_ids: set = set()
        self.total_tokens = 0
        self.manifest = {}
        self.is_complete = True
        # single-slot cache for epoch_permutation(); see its docstring
        self._perm_cache_key: tuple | None = None
        self._perm_cache: np.ndarray | None = None

        self.refresh()  # initial load shares all logic with later incremental refreshes
        if self.num_sequences == 0:
            raise ValueError(
                f"corpus has only {self.total_tokens} tokens, fewer than seq_len={seq_len}; "
                "cannot form a single packed sequence"
            )

    def refresh(self) -> bool:
        """Re-read manifest.json and load any shards published since the last load/refresh.
        Returns True if the corpus grew (new shards appeared). Cheap on a no-op call (just a
        small JSON parse); only re-hashes/opens shards not already loaded -- previously-loaded
        shards are immutable once published (atomic-publish never rewrites an existing shard),
        so re-verifying them on every refresh would be wasted I/O at 100BT scale."""
        with open(os.path.join(self.tokenized_dir, "manifest.json")) as f:
            manifest = json.load(f)
        self.manifest = manifest
        self.is_complete = manifest.get("is_complete", True)  # legacy manifests: assume complete

        new_shards = [s for s in manifest["shards"] if s["shard_id"] not in self._loaded_shard_ids]
        if not new_shards:
            return False

        for shard in new_shards:
            shard_id = shard["shard_id"]
            prefix = os.path.join(self.tokenized_dir, shard_id)
            if self.verify_sha256:
                actual_bin = sha256_file(get_bin_path(prefix))
                actual_idx = sha256_file(get_idx_path(prefix))
                if actual_bin != shard["bin_sha256"] or actual_idx != shard["idx_sha256"]:
                    raise ValueError(
                        f"sha256 mismatch for shard {shard_id}: manifest says "
                        f"bin={shard['bin_sha256']} idx={shard['idx_sha256']}, got "
                        f"bin={actual_bin} idx={actual_idx}. Refusing to train on unverified data."
                    )
            ds = IndexedDataset(prefix)
            self._datasets[shard_id] = ds
            lengths = ds.sequence_lengths  # one entry per document (see tokenize_to_bin_idx.py)
            for i, length in enumerate(lengths):
                self._doc_shard_id.append(shard_id)
                self._doc_idx_in_shard.append(i)
                self._doc_cum_start.append(self.total_tokens)
                self.total_tokens += int(length)
            self._loaded_shard_ids.add(shard_id)

        self.num_sequences = self.total_tokens // self.seq_len
        return True

    def ensure_available(self, global_pos: int, on_boundary: str, poll_interval_s: float = 30.0) -> None:
        """Block or raise until `global_pos` is servable by the currently-published corpus.
        `on_boundary` has no default -- see module docstring "Streaming / incremental publish"
        for why "wait" and "raise" are both real supported choices, not a fallback pair.

        Only meaningful while the corpus might still grow (is_complete=False) AND global_pos
        would otherwise need to wrap into a nominal "epoch 1" -- i.e. this refuses to let a
        streaming, not-yet-complete corpus be treated as if a real second pass had started."""
        if on_boundary not in ("wait", "raise"):
            raise ValueError(f'on_boundary must be "wait" or "raise", got {on_boundary!r}')
        while global_pos >= self.num_sequences:
            if self.is_complete:
                raise CorpusExhaustedError(
                    f"global_pos={global_pos} >= num_sequences={self.num_sequences} and "
                    f"{self.tokenized_dir}/manifest.json says is_complete=True (no more shards "
                    f"are coming) -- this is a real end-of-corpus condition."
                )
            if on_boundary == "raise":
                raise CorpusBoundaryError(
                    f"global_pos={global_pos} >= currently-published num_sequences="
                    f"{self.num_sequences}, and the corpus at {self.tokenized_dir} is not yet "
                    f"complete (more tokenize batches pending). on_boundary='raise' means: stop "
                    f"here and let the caller decide (retry later, checkpoint, or restart with "
                    f"on_boundary='wait')."
                )
            print(
                f"[PackedSeqDataset] waiting for more published data: need global_pos="
                f"{global_pos}, have num_sequences={self.num_sequences}; sleeping "
                f"{poll_interval_s}s then re-checking {self.tokenized_dir}/manifest.json ...",
                flush=True,
            )
            time.sleep(poll_interval_s)
            grew = self.refresh()
            if grew:
                print(
                    f"[PackedSeqDataset] corpus grew to num_sequences={self.num_sequences} "
                    f"(WARNING: epoch-0 permutation for this run is now computed against the "
                    f"NEW corpus size -- order will NOT bit-for-bit replay against a fresh "
                    f"construction pointed at the eventual final manifest; see module docstring "
                    f"'Streaming / incremental publish'. Expected/accepted for early runs, NOT "
                    f"for anything requiring full-run reproducibility.)",
                    flush=True,
                )

    # ---- packing (seed-independent) ----

    def _doc_for_token(self, token_pos: int) -> int:
        """Index of the document containing global corpus token position `token_pos`."""
        return bisect.bisect_right(self._doc_cum_start, token_pos) - 1

    def locate(self, logical_seq_idx: int) -> Location:
        if not 0 <= logical_seq_idx < self.num_sequences:
            raise IndexError(logical_seq_idx)
        token_start = logical_seq_idx * self.seq_len
        doc_i = self._doc_for_token(token_start)
        shard_id = self._doc_shard_id[doc_i]
        doc_idx_in_shard = self._doc_idx_in_shard[doc_i]
        token_offset_in_doc = token_start - int(self._doc_cum_start[doc_i])
        ds = self._datasets[shard_id]
        seq_pointer, _length, _mode = ds.index[doc_idx_in_shard]
        itemsize = ds.index.dtype_size if hasattr(ds.index, "dtype_size") else np.dtype(np.int32).itemsize
        byte_offset = int(seq_pointer) + token_offset_in_doc * itemsize
        return Location(shard_id, doc_idx_in_shard, byte_offset, token_offset_in_doc)

    def get_sequence(self, logical_seq_idx: int) -> np.ndarray:
        """Return the `seq_len` token ids for packed sequence `logical_seq_idx` (canonical,
        seed-independent order). May span multiple documents and multiple shards."""
        if not 0 <= logical_seq_idx < self.num_sequences:
            raise IndexError(logical_seq_idx)
        token_start = logical_seq_idx * self.seq_len
        token_end = token_start + self.seq_len
        out = np.empty(self.seq_len, dtype=np.int64)
        filled = 0
        pos = token_start
        doc_i = self._doc_for_token(pos)
        while filled < self.seq_len:
            shard_id = self._doc_shard_id[doc_i]
            doc_idx_in_shard = self._doc_idx_in_shard[doc_i]
            doc_start = int(self._doc_cum_start[doc_i])
            ds = self._datasets[shard_id]
            doc_len = int(ds.sequence_lengths[doc_idx_in_shard])
            in_doc_offset = pos - doc_start
            take = min(doc_len - in_doc_offset, self.seq_len - filled)
            chunk = ds.get(doc_idx_in_shard, offset=in_doc_offset, length=take)
            out[filled : filled + take] = np.asarray(chunk)
            filled += take
            pos += take
            doc_i += 1
        return out

    # ---- ordering (seed-dependent, O(1)-ish resumable) ----

    def epoch_permutation(self, epoch: int) -> np.ndarray:
        """Deterministic pure function of (self.seed, epoch). Same seed -> same array, always,
        including across processes/restarts -- two passes with the same seed give bit-identical order.

        Single-slot cached on (epoch, num_sequences): the adapter calls this once per *sample*
        (`torchtitan_adapter._seq_idx_for_global_pos`) just to read one index out of it, and at
        production scale (num_sequences ~2.4e7) rebuilding a 191 MB permutation per sample is
        what the dataloader spends its time on. Only one epoch is cached, and a rebuild is a
        pure recomputation, so the result is bit-identical either way -- caching changes speed,
        never order. `num_sequences` is part of the key because `refresh()` can grow the corpus
        (see "Streaming / incremental publish" above), which changes the permutation for an
        epoch already in the slot.

        The returned array is the cached object, shared with every other caller: it must be
        treated as read-only and never modified in place.
        """
        key = (epoch, self.num_sequences)
        if self._perm_cache_key != key:
            rng = np.random.default_rng(self.seed + epoch)
            self._perm_cache = rng.permutation(self.num_sequences)
            self._perm_cache_key = key
        return self._perm_cache

    def training_order_index(self, step: int, batch_size: int) -> np.ndarray:
        """The logical_seq_idx values consumed at optimizer `step` (0-indexed), for a run with
        constant `batch_size` sequences/step. Pure function of (seed, step, batch_size, corpus
        size) -- callable for ANY step without iterating prior steps, which is what makes
        replay-from-arbitrary-step possible."""
        global_start = step * batch_size
        global_end = global_start + batch_size
        idxs = np.empty(batch_size, dtype=np.int64)
        pos = global_start
        out_i = 0
        while pos < global_end:
            epoch, within = divmod(pos, self.num_sequences)
            perm = self.epoch_permutation(epoch)
            take = min(self.num_sequences - within, global_end - pos)
            idxs[out_i : out_i + take] = perm[within : within + take]
            pos += take
            out_i += take
        return idxs

    def get_batch(self, step: int, batch_size: int) -> np.ndarray:
        """Shape (batch_size, seq_len) int64 token ids for optimizer step `step`."""
        idxs = self.training_order_index(step, batch_size)
        return np.stack([self.get_sequence(int(i)) for i in idxs], axis=0)

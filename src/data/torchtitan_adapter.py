"""torchtitan-compatible wrapper around PackedSeqDataset.

torchtitan's dataloader is a
pluggable `BaseDataLoader.Config` component; its concrete `ParallelAwareDataloader` wraps any
`torch.utils.data.IterableDataset` that also implements
`torch.distributed.checkpoint.stateful.Stateful` (state_dict/load_state_dict), and handles
DP-rank slicing + StatefulDataLoader/DCP checkpoint plumbing itself. This module supplies just
that: an IterableDataset+Stateful adapter. It does NOT import torchtitan (keeps this directory
framework-agnostic) -- only depends on core torch.

Sharding across DP ranks: each rank consumes global sample positions
`global_sample_pos ≡ dp_rank (mod dp_world_size)`, advancing by `dp_world_size` per yield. This
is the standard interleaved-IterableDataset pattern -- ranks' local batches recombine into the
correct globally-ordered batch without any cross-rank communication.

`state_dict()` includes both the minimal resumption state (seed, global_sample_pos) AND the
human-readable provenance fields (shard_id, shard_sha256,
offset_in_shard) via `PackedSeqDataset.locate()` -- the former is what correctness depends on,
the latter is what a training log or a debugging run wants to see.

Batch shape: `__iter__` yields `(input_dict, label_tensor)` 2-tuples, per torchtitan v0.2.2's
actual trainer contract (verified against torchtitan source) -- see
the docstring on `__iter__` for why `label` must NOT be a key inside the dict.

`on_boundary` ("train while downloading/tokenizing"): required, no
default. Every position is checked against `PackedSeqDataset.ensure_available()` before being
served -- see `packed_dataset.py`'s "Streaming / incremental publish" docstring section for the
full semantics and the determinism tradeoff `on_boundary="wait"` accepts. Short version:
`"raise"` fails fast when training catches up to the tokenize pipeline (recommended for a
full production run -- let the trainer decide how to pause, don't block a whole job silently);
`"wait"` blocks and polls (recommended only for small early/toy runs like the 216M open-a-few-
shards run, where the corpus streaming in is expected and full historical reproducibility
isn't the point yet).
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator

import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from src.data.packed_dataset import PackedSeqDataset


class PackedSeqIterableDataset(IterableDataset, Stateful):
    """No defaults on any semantic parameter (seed/seq_len/dp_rank/dp_world_size/on_boundary) --
    semantic parameters have no defaults; a silently-defaulted dp_world_size=1 would train
    every rank on the identical data stream and nobody would notice until eval looked odd, and a
    silently-defaulted on_boundary would either hang training forever with no explanation or
    fail every run against an incomplete corpus with no explanation -- both need to be a choice
    someone made on purpose."""

    def __init__(
        self,
        tokenized_dir: str,
        seq_len: int,
        seed: int,
        dp_rank: int,
        dp_world_size: int,
        on_boundary: str,
        verify_sha256: bool = True,
        poll_interval_s: float = 30.0,
    ):
        if not (0 <= dp_rank < dp_world_size):
            raise ValueError(f"dp_rank={dp_rank} must be in [0, dp_world_size={dp_world_size})")
        if on_boundary not in ("wait", "raise"):
            raise ValueError(f'on_boundary must be "wait" or "raise", got {on_boundary!r}')
        self._ds = PackedSeqDataset(tokenized_dir, seq_len=seq_len, seed=seed, verify_sha256=verify_sha256)
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.seed = seed
        self.on_boundary = on_boundary
        self.poll_interval_s = poll_interval_s
        # Resumable cursor: next global sample position (across ALL ranks) this rank will emit.
        # Initialized to this rank's first position; advances by dp_world_size each __next__.
        self._global_sample_pos = dp_rank

    def _seq_idx_for_global_pos(self, global_pos: int) -> int:
        epoch, within = divmod(global_pos, self._ds.num_sequences)
        return int(self._ds.epoch_permutation(epoch)[within])

    def __iter__(self) -> Iterator[tuple[dict, torch.Tensor]]:
        """Yields `(input_dict, label_tensor)` -- a 2-tuple, NOT a single dict.

        This exact shape is dictated by torchtitan v0.2.2's trainer (`train.py:403`,
        confirmed against real torchtitan source):
            input_dict, labels = batch
            inputs = input_dict["input"]
            extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        If `label` were left inside the dict instead of being the tuple's 2nd element, it
        would silently land in `extra_inputs` and get forwarded as a model kwarg -- our
        model's `**kwargs` would swallow it, `labels` would stay None, loss would be None,
        and training would *look* like it's running with no error while producing zero
        gradient signal. Keeping the two apart is not a style choice, it's the difference
        between training and silently not training.
        """
        while True:
            self._ds.ensure_available(
                self._global_sample_pos, on_boundary=self.on_boundary, poll_interval_s=self.poll_interval_s
            )
            logical_idx = self._seq_idx_for_global_pos(self._global_sample_pos)
            tokens = self._ds.get_sequence(logical_idx)
            # Advance BEFORE yielding, not after: a generator only resumes execution past
            # `yield` on the *next* call to next(), so incrementing after yield means
            # state_dict() called right after a next() call would still report the position
            # that was just consumed, not the position to resume from -- and load_state_dict()
            # would then re-yield a duplicate sample. Advancing first keeps
            # `self._global_sample_pos` always equal to "next position to emit", which is
            # exactly what state_dict()/load_state_dict() need it to mean.
            self._global_sample_pos += self.dp_world_size
            input_ids = torch.from_numpy(tokens[:-1].copy())
            labels = torch.from_numpy(tokens[1:].copy())
            yield {"input": input_ids}, labels

    def state_dict(self) -> dict[str, Any]:
        # last position actually emitted by this rank == next position to resume from, minus
        # one stride (see __iter__ comment for why `_global_sample_pos` already points past it).
        last_emitted_pos = self._global_sample_pos - self.dp_world_size
        loc_fields = {}
        if last_emitted_pos >= 0:
            logical_idx = self._seq_idx_for_global_pos(last_emitted_pos)
            loc = self._ds.locate(logical_idx)
            loc_fields = {
                "shard_id": loc.shard_id,
                "shard_sha256": self._shard_sha256(loc.shard_id),
                "offset_in_shard_bytes": loc.byte_offset_in_bin,
                "doc_idx_in_shard": loc.doc_idx_in_shard,
            }
        return {
            "seed": self.seed,
            "dp_rank": self.dp_rank,
            "dp_world_size": self.dp_world_size,
            "global_sample_idx": self._global_sample_pos,  # next position to resume from
            "epoch": self._global_sample_pos // self._ds.num_sequences,
            **loc_fields,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict["seed"] != self.seed:
            raise ValueError(
                f"checkpoint seed={state_dict['seed']} != current run seed={self.seed}; "
                "resuming with a different seed would silently change data order. Refusing."
            )
        if state_dict["dp_world_size"] != self.dp_world_size:
            raise ValueError(
                f"checkpoint dp_world_size={state_dict['dp_world_size']} != current "
                f"dp_world_size={self.dp_world_size}. Changing world size mid-run needs an "
                "explicit re-sharding decision, not a silent resume."
            )
        self._global_sample_pos = state_dict["global_sample_idx"]

    def _shard_sha256(self, shard_id: str) -> str:
        for shard in self._ds.manifest["shards"]:
            if shard["shard_id"] == shard_id:
                return shard["bin_sha256"]
        return ""

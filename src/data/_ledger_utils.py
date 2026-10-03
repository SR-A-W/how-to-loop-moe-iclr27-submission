"""Shared append-only resume-ledger primitives (long jobs write their progress in segments).

Used identically by `download_shards.py` (per-raw-shard) and `tokenize_to_bin_idx.py`
(per-tokenized-shard) -- same pattern, same three functions, factored out once it was needed
twice rather than duplicated (at 100BT scale both the download and the tokenize phase are
long enough to get interrupted by scheduler walltime/preemption, and both need to
resume without redoing already-completed shards).

A ledger is a JSONL file: one JSON object per line, appended (and fsynced) immediately after a
unit of work completes. Keyed by an arbitrary caller-chosen string key (a filename for
downloads, a shard_id for tokenization) -- if a key appears more than once (e.g. redone after a
hash mismatch on a prior run), the LAST line for that key wins, matching normal append-log
semantics.
"""
from __future__ import annotations

import json
import os


def fsync_path(path: str) -> None:
    """fsync a file OR directory by path -- forces its on-disk state durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_ledger(ledger_path: str) -> dict:
    """key -> most-recent entry recorded for that key. Empty dict if the ledger doesn't exist
    yet (first run)."""
    entries = {}
    if not os.path.exists(ledger_path):
        return entries
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries[entry["key"]] = entry
    return entries


def append_ledger(ledger_path: str, entry: dict) -> None:
    """Append one entry (must contain a "key" field) and fsync before returning -- the entry is
    durable on disk before the caller moves on to the next unit of work, which is the whole
    point: a kill right after this call still counts this unit as done on restart."""
    assert "key" in entry, 'ledger entries must have a "key" field'
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())

"""Shared helpers for the held-out validation-loss slice builders
(`build_fineweb_edu_heldout.py`, `build_domain_heldout.py`) -- factored out once needed by both
rather than duplicated, same reasoning as `_ledger_utils.py`.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys


def repo_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def hf_token() -> str | None:
    """The cached Hugging Face token if one exists, else None: every source these
    builders read is a public dataset, so no login is needed."""
    path = os.path.expanduser("~/.cache/huggingface/token")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read().strip()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True
        ).strip()
    except Exception:
        return "unknown"


class DownloadTimeout(Exception):
    pass


def download_with_timeout(
    repo_id: str, rel_path: str, token: str, local_dir: str, *,
    repo_type: str = "dataset", timeout_s: int = 300, retries: int = 3,
) -> str:
    """`hf_hub_download`, bounded.

    Observed failure: a bare `hf_hub_download` call hung with zero output for ~4 hours
    (`ps` showed uninterruptible IO wait, `Dl`) -- `hf_hub_download` takes no timeout argument
    of its own, so a stalled connection blocks forever rather than erroring. `signal.alarm`
    bounds the WHOLE call (DNS + connect + the actual transfer), raises `DownloadTimeout`
    inside it, and this retries a few times before giving up loudly -- silently hanging again
    is exactly the failure mode this exists to end, so the last retry's exception propagates
    rather than being swallowed.
    """
    import signal

    from huggingface_hub import hf_hub_download

    def _on_alarm(signum, frame):
        raise DownloadTimeout(f"{rel_path} did not complete within {timeout_s}s")

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(timeout_s)
        try:
            return hf_hub_download(
                repo_id, rel_path, repo_type=repo_type, token=token, local_dir=local_dir,
            )
        except DownloadTimeout as e:
            last_exc = e
            print(f"  [timeout] {rel_path} attempt {attempt}/{retries}: {e}", file=sys.stderr, flush=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    raise RuntimeError(f"{rel_path}: gave up after {retries} attempts") from last_exc


def write_bin_idx(token_id_lists: list, out_dir: str, shard_id: str = "shard_000") -> dict:
    """Write `<out_dir>/<shard_id>.bin/.idx` from a list of token-id lists (one per document),
    return ONE entry for manifest["shards"] -- `packed_dataset.PackedSeqDataset.refresh()` reads
    exactly this shape (`shard_id`/`bin_sha256`/`idx_sha256`, prefix = `<tokenized_dir>/
    <shard_id>`), the same schema `tokenize_to_bin_idx.py` writes -- every held-out set built
    with this helper is a drop-in `--val-tokenized-dir` for
    `src/eval/run_eval.py::compute_fineweb_edu_val_loss`, not a new format eval special-cases.
    """
    import numpy as np
    import torch
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder, get_bin_path, get_idx_path

    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, shard_id)
    # np.int32, matching tokenize_to_bin_idx.py's TOKEN_DTYPE -- same bin/idx byte format as
    # the rest of the pipeline, not a format this held-out set invents on its own.
    builder = IndexedDatasetBuilder(get_bin_path(out_prefix), dtype=np.int32, multimodal=False)
    num_tokens = 0
    for ids in token_id_lists:
        builder.add_document(torch.tensor(ids, dtype=torch.int64), [len(ids)])
        num_tokens += len(ids)
    builder.finalize(get_idx_path(out_prefix))
    return {
        "shard_id": shard_id,
        "bin_sha256": sha256_file(get_bin_path(out_prefix)),
        "idx_sha256": sha256_file(get_idx_path(out_prefix)),
        "num_docs": len(token_id_lists),
        "num_tokens": num_tokens,
    }

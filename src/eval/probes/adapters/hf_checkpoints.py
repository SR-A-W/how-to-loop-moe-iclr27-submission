"""Streaming fetch/delete of OLMoE intermediate checkpoints for the collapse
evolution curve ("download -> probe -> delete", peak
disk disclosed per step; point selection follows this project's own
trajectory schedule shape, `src.pretrain.checkpointing.schedule.trajectory_steps`).

This module does NOT run any metric. It only (1) turns the schedule into the
list of HF branch names that exist for the repo, raising if a scheduled step
has no branch (never silently substituting a neighbour), (2) downloads one
revision into the HF cache, and (3) deletes one revision from the cache,
printing the bytes freed. Every semantic argument is required -- no defaults.

Usage:
    python -m src.eval.probes.adapters.hf_checkpoints list   --repo allenai/OLMoE-1B-7B-0924 --total-steps 1220000 --first-step 5000 --uniform-every 250000
    python -m src.eval.probes.adapters.hf_checkpoints fetch  --repo allenai/OLMoE-1B-7B-0924 --revision step5000-tokens20B
    python -m src.eval.probes.adapters.hf_checkpoints delete --repo allenai/OLMoE-1B-7B-0924 --revision step5000-tokens20B
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from src.pretrain.checkpointing.schedule import trajectory_steps

_STEP_BRANCH = re.compile(r"^step(\d+)-tokens\d+B$")
_WEIGHT_PATTERNS = ["*.json", "*.safetensors"]


def scheduled_revisions(repo: str, *, total_steps: int, first_step: int, uniform_every: int) -> list[str]:
    """Branch names for every scheduled step. Raises if any scheduled step
    (other than `total_steps` itself, which maps to the final branch) has no
    `step<N>-tokens<T>B` branch -- a missing point must be visible, not patched."""
    branches = {}
    for ref in HfApi().list_repo_refs(repo).branches:
        m = _STEP_BRANCH.match(ref.name)
        if m:
            branches[int(m.group(1))] = ref.name
    steps = trajectory_steps(total_steps, first_step=first_step, uniform_every=uniform_every)
    missing = [s for s in steps if s not in branches]
    if missing:
        raise ValueError(f"{repo}: no step branch for scheduled steps {missing}; available max step {max(branches)}")
    return [branches[s] for s in steps]


def _revision_dir(repo: str, revision: str) -> Path:
    repo_dir = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    ref = repo_dir / "refs" / revision
    if not ref.exists():
        raise FileNotFoundError(f"{revision!r} of {repo} is not in the cache ({ref} missing)")
    return repo_dir / "snapshots" / ref.read_text().strip()


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) + sum(
        p.resolve().stat().st_size for p in path.rglob("*") if p.is_symlink()
    )


def fetch(repo: str, revision: str) -> Path:
    path = Path(snapshot_download(repo, revision=revision, allow_patterns=_WEIGHT_PATTERNS))
    print(f"fetched {repo}@{revision} -> {path} ({_dir_bytes(path) / 1e9:.2f} GB)")
    return path


def delete(repo: str, revision: str) -> int:
    """Remove the snapshot dir and the blobs only this revision points to."""
    snap = _revision_dir(repo, revision)
    repo_dir = snap.parent.parent
    freed = 0
    for p in snap.rglob("*"):
        if p.is_symlink():
            blob = p.resolve()
            others = [s for s in (repo_dir / "snapshots").iterdir() if s != snap]
            shared = any((o / p.relative_to(snap)).exists() and (o / p.relative_to(snap)).resolve() == blob for o in others)
            if not shared and blob.exists():
                freed += blob.stat().st_size
                blob.unlink()
    shutil.rmtree(snap)
    (repo_dir / "refs" / revision).unlink()
    print(f"deleted {repo}@{revision}: freed {freed / 1e9:.2f} GB")
    return freed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--repo", required=True)
    p.add_argument("--total-steps", type=int, required=True)
    p.add_argument("--first-step", type=int, required=True)
    p.add_argument("--uniform-every", type=int, required=True)
    for name in ("fetch", "delete"):
        q = sub.add_parser(name)
        q.add_argument("--repo", required=True)
        q.add_argument("--revision", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for r in scheduled_revisions(a.repo, total_steps=a.total_steps, first_step=a.first_step, uniform_every=a.uniform_every):
            print(r)
    elif a.cmd == "fetch":
        fetch(a.repo, a.revision)
    else:
        delete(a.repo, a.revision)


if __name__ == "__main__":
    main()

"""Writing and reading checkpoints.

Two kinds (disk-constrained):

  * `save_trajectory` -- bf16 weights in HF `from_pretrained` layout, no
    optimizer. Kept for the whole run. This is also exactly the format the
    diagnostics pipeline consumes, so the scientific artefact and the
    acceptance artefact are the same file.
  * `save_resume` / `load_resume` -- full state (weights, optimizer, RNG, data
    position) for restart, rolled to the most recent few.

The completeness rule
---------------------
Every checkpoint directory is finished by writing its manifest **last**, after
the payload is flushed and fsynced. The manifest's presence is therefore the
completeness marker, and `is_complete()` is the only sanctioned way to ask.

This exists because of how failure actually looks here: the disk quota is shared
and nearly full, and a write that runs out of space does not raise cleanly at a
convenient moment -- it leaves a directory of plausible-looking, truncated
tensors. A checkpoint that is half-written and *looks* loadable is worse than no
checkpoint, because it can silently become the point a run resumes from, or a
data point in the trajectory analysis.

Every write is additionally (a) retried with progressive backoff and (b) staged
through a temp file and renamed, so a retry can never observe its own partial
output.

Known gap: single-process only
------------------------------
There is no rank coordination and no NCCL fault handling here -- this module has
no `torch.distributed` concept at all. LT2's checkpointing has "5 retries +
NCCL resilience"; the retry half is implemented, the distributed half is not,
because multi-rank saving is only reachable through the torchtitan path (which
brings its own DCP writer) and has not been exercised on real hardware yet.
Stating the gap rather than implying coverage: a rank-0-only save under FSDP2
would silently write shards that cannot be reassembled.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "MANIFEST_NAME",
    "RETRY_DELAYS",
    "capture_rng_state",
    "restore_rng_state",
    "is_complete",
    "save_trajectory",
    "save_resume",
    "load_resume",
    "prune_resume",
]

MANIFEST_NAME = "manifest.json"
"""Written last; its presence means the checkpoint is complete."""

_RESUME_STATE_NAME = "resume_state.pt"


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG stream that affects training.

    A resumed run must be statistically indistinguishable
    from an uninterrupted one. Missing any one of these streams breaks that in a
    way no loss curve reveals immediately -- dropout masks or data shuffling
    quietly diverge, and the two runs stop being comparable without ever looking
    broken.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a snapshot from `capture_rng_state`.

    Raises on a missing stream rather than restoring partially: a silently
    half-restored RNG is the exact failure this is meant to prevent.
    """
    for required in ("python", "torch"):
        if required not in state:
            raise ValueError(
                f"RNG snapshot is missing the {required!r} stream; refusing to "
                "restore partially (a partial restore breaks resume equivalence "
                "without any visible symptom)."
            )
    random.setstate(state["python"])
    torch.set_rng_state(_as_uint8_cpu(state["torch"]))
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_uint8_cpu(s) for s in state["cuda"]])


def _as_uint8_cpu(t: Any) -> torch.Tensor:
    """torch RNG states must be uint8 CPU tensors; a round-trip can change dtype."""
    tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t)
    return tensor.to(device="cpu", dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def is_complete(path: str | Path) -> bool:
    """True iff the directory holds a finished checkpoint.

    Checks for the manifest, which is written last. Do not substitute "the
    directory exists" or "the weights file exists" -- both are true of a
    checkpoint that ran out of disk halfway.
    """
    manifest = Path(path) / MANIFEST_NAME
    if not manifest.is_file():
        return False
    try:
        with manifest.open() as fh:
            json.load(fh)
    except (json.JSONDecodeError, OSError):
        # A truncated manifest means we died *while writing the marker*.
        return False
    return True


def _fsync_dir(path: Path) -> None:
    """Force directory metadata out, so the manifest's existence is durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


RETRY_DELAYS = (5, 10, 20, 30, 60)
"""Backoff between checkpoint save attempts, in seconds. Five retries.

Copied in spirit from LT2's `lingua/checkpoint.py`, which is worth imitating:
they wrote progressive-retry and NCCL-resilience code
that the paper never mentions, which is evidence they actually lost checkpoints
in production. On a shared, nearly-full filesystem the same classes of transient
failure (ENOSPC that clears, NFS hiccups, a collective that times out) are
exactly what turns a 3-day run into a lost 3-day run.
"""


def _with_retries(operation: Callable[[], None], *, what: str) -> None:
    """Run a save operation, retrying with progressive backoff.

    Retries only the *write*. Callers are responsible for making each attempt
    idempotent -- here that means writing to a temp path and renaming, so a
    half-written attempt never becomes the visible state.

    A distributed-aware variant (rank coordination, NCCL fault handling) is not
    implemented: this module has no `torch.distributed` concept yet, and
    pretending otherwise would be worse than the honest gap. See the note in the
    module docstring.
    """
    last: Exception | None = None
    for attempt, delay in enumerate((*RETRY_DELAYS, None), start=1):
        try:
            operation()
            return
        # Only I/O is retried. A TypeError or ValueError here is a bug in the
        # calling code, and retrying it six times with backoff just delays the
        # traceback by two minutes while printing five misleading "retrying"
        # lines that look like an infrastructure problem. Bugs should fail fast;
        # only ENOSPC/EIO/NFS-class failures are worth waiting out.
        except OSError as exc:
            last = exc
            if delay is None:
                break
            print(
                f"[checkpoint] {what} failed (attempt {attempt}/{len(RETRY_DELAYS) + 1}): "
                f"{exc!r}; retrying in {delay}s"
            )
            time.sleep(delay)
    raise RuntimeError(
        f"{what} failed after {len(RETRY_DELAYS) + 1} attempts; last error: {last!r}"
    ) from last


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Write via a temp file in the same directory, then rename into place.

    `os.replace` is atomic within a filesystem, so a reader sees either the
    previous contents or the new ones -- never a truncated file. Opening the
    target directly would truncate it immediately, leaving a window in which the
    manifest still claims the checkpoint is complete while the payload is
    half-written. That window matters here because resume checkpoints are written
    repeatedly to the *same* path.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_json_fsynced(path: Path, payload: dict[str, Any]) -> None:
    def _write(target: Path) -> None:
        with target.open("w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())

    _atomic_write(path, _write)


# ---------------------------------------------------------------------------
# Trajectory checkpoints
# ---------------------------------------------------------------------------


#: Files copied into every trajectory checkpoint so it can be loaded without
#: this repository. Order matters only for readability.
_BUNDLED_MODULES = (
    ("src.model.modeling_loop_lm", "modeling_loop_lm.py"),
    ("src.model.loop_trace", "loop_trace.py"),
)

MODEL_VARIANT = "looped-moe"
"""Matches the seedvar `model_variant` vocabulary.

The diagnostics collector derives the *expected* number of loop steps from
`model_variant` rather than from `num_stacks` -- deliberately, because seedvar's
non-looped quadrants also carry `num_stacks=2` and reading K off the config would
hand `base`/`moe` a wrong K=2 expectation. Omitting this field would leave the
collector with no way to state its expectation, so it must be written out.
"""


def _make_self_describing(config: Any) -> None:
    """Add the fields HuggingFace needs to load this checkpoint cold.

    Without `architectures` and `auto_map`, `AutoModelForCausalLM.from_pretrained`
    cannot construct our custom class, and the premise that "the diagnostics
    pipeline ingests new checkpoints with zero changes" silently does not hold. The published seedvar checkpoints carry exactly these fields;
    ours were missing them.
    """
    config.architectures = ["LoopMoEForCausalLM"]
    config.model_variant = MODEL_VARIANT
    config.auto_map = {
        "AutoConfig": "modeling_loop_lm.LoopMoEConfig",
        "AutoModelForCausalLM": "modeling_loop_lm.LoopMoEForCausalLM",
    }


def _bundle_modeling_code(directory: Path) -> None:
    """Copy the modeling source into the checkpoint.

    A `trust_remote_code` checkpoint has to be self-contained: the consumer is a
    different repository that has never heard of `pretrain`. `modeling_loop_lm.py`
    falls back to importing `loop_trace` as a sibling for this reason, so both
    files travel together.
    """
    import importlib

    for module_name, filename in _BUNDLED_MODULES:
        source = Path(importlib.import_module(module_name).__file__)
        shutil.copyfile(source, directory / filename)


def save_trajectory(
    model: Any,
    path: str | Path,
    *,
    manifest_fields: dict[str, Any],
    state_dict: dict[str, Any] | None = None,
    preserve_dtype_keys: set[str] | None = None,
) -> Path:
    """Write bf16 weights in HF layout, then the manifest.

    `model` is the HF-facing module (`LoopMoEForCausalLM`), so the result loads
    with `from_pretrained` and the diagnostics pipeline ingests it unchanged.

    `manifest_fields` is passed through to `src.pretrain.manifest.write_manifest`,
    which enforces its own TIER-1 non-empty contract. Nothing is defaulted here:
    fields like `data_position` are only knowable by the training loop, and
    inventing a placeholder would defeat the contract.

    `kind` is set here rather than accepted from the caller: this function only
    ever writes trajectory checkpoints, so letting a caller label the output
    something else could only ever mislabel it. The manifest contract rejects
    `kind="trajectory"` together with `rng_state_saved=True`, which catches a
    resume checkpoint accidentally written through this path -- a mistake that
    would otherwise blow the disk budget ~7x per checkpoint in silence.
    """
    from src.pretrain.manifest import write_manifest

    if "kind" in manifest_fields:
        raise ValueError(
            "do not pass `kind` to save_trajectory; it is always 'trajectory'."
        )

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

    # Cast a CPU copy to bf16; the live model is untouched.
    # `state_dict` is passed in when the model is sharded: under FSDP2 the
    # parameters are DTensors spread across ranks, and only a caller who has
    # already gathered them (rank 0, via
    # `get_model_state_dict(..., full_state_dict=True)`) holds the real tensors.
    # Calling `model.state_dict()` there would silently persist shards.
    source = state_dict if state_dict is not None else model.state_dict()

    # Weights go to bf16; **buffers keep their dtype**. The RoPE rotation table
    # is the case that matters: it is computed in float64 and stored float32
    # precisely because at seq_len 4096 the largest angle is ~4096 rad, where
    # low precision costs several decimal digits. Round-tripping it through
    # bf16 degrades the rotation matrices' orthonormality from ~1e-7 to ~5.5e-3
    # -- four orders of magnitude -- so a reloaded checkpoint would compute
    # measurably different positional encodings than the run that produced it,
    # silently shifting every diagnostic taken on that checkpoint.
    preserve = preserve_dtype_keys
    if preserve is None:
        named_buffers = getattr(model, "named_buffers", None)
        preserve = {name for name, _ in named_buffers()} if callable(named_buffers) else set()

    state_dict = {
        k: v.detach().to(device="cpu")
        if k in preserve
        else v.detach().to(device="cpu", dtype=torch.bfloat16)
        for k, v in source.items()
    }

    def _write_payload() -> None:
        _make_self_describing(model.config)
        model.config.save_pretrained(directory)
        _bundle_modeling_code(directory)

        def _write(target: Path) -> None:
            with target.open("wb") as fh:
                torch.save(state_dict, fh)
                fh.flush()
                os.fsync(fh.fileno())

        _atomic_write(directory / "pytorch_model.bin", _write)
        # Payload must be durable *before* the marker that claims it is.
        _fsync_dir(directory)

    (directory / MANIFEST_NAME).unlink(missing_ok=True)
    _with_retries(_write_payload, what=f"save_trajectory payload at {directory}")

    _with_retries(
        lambda: write_manifest(directory / MANIFEST_NAME, kind="trajectory", **manifest_fields),
        what=f"save_trajectory manifest at {directory}",
    )
    _fsync_dir(directory)
    return directory


# ---------------------------------------------------------------------------
# Resume checkpoints
# ---------------------------------------------------------------------------


def save_resume(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    step: int,
    dataloader_state: dict[str, Any],
    manifest_fields: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write everything needed to continue training bit-comparably.

    Contents: model weights, optimizer state, all four RNG streams, the step
    counter, and the dataloader position. Anything omitted here is something a
    resumed run would silently get wrong.

    The manifest goes through the same `write_manifest` TIER-1 contract as
    trajectory checkpoints, with `kind="resume"`. An earlier version hand-rolled
    a three-field manifest here, which meant the very checkpoints most in need of
    provenance -- the ones you restart a multi-day run from -- carried no
    `git_commit`, `torch_version` or `compute_dtype`, and the contract test never
    exercised this path at all.

    Written to a temp path and renamed, because this function is called
    repeatedly on the *same* directory. Truncating the payload in place would
    leave a window where the old manifest still vouches for a half-written file.
    """
    from src.pretrain.manifest import write_manifest

    if "kind" in manifest_fields:
        raise ValueError("do not pass `kind` to save_resume; it is always 'resume'.")
    if manifest_fields.get("step") != step:
        raise ValueError(
            f"manifest_fields['step']={manifest_fields.get('step')!r} disagrees with "
            f"step={step}. The manifest must describe the checkpoint it sits next to; "
            "a mismatch would make the trajectory unreadable after the fact."
        )
    if not manifest_fields.get("rng_state_saved", False):
        raise ValueError(
            "save_resume writes the RNG streams, so manifest_fields must declare "
            "rng_state_saved=True. A resume checkpoint claiming otherwise cannot "
            "promise replay-identical restart."
        )

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(),
        "dataloader": dataloader_state,
        "extra": extra or {},
    }

    def _write_payload() -> None:
        def _write(target: Path) -> None:
            with target.open("wb") as fh:
                torch.save(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())

        _atomic_write(directory / _RESUME_STATE_NAME, _write)
        _fsync_dir(directory)

    # The manifest is removed first so that a crash mid-write cannot leave the
    # previous (now stale) manifest vouching for a payload that is being replaced.
    (directory / MANIFEST_NAME).unlink(missing_ok=True)
    _fsync_dir(directory)

    _with_retries(_write_payload, what=f"save_resume payload at {directory}")

    _with_retries(
        lambda: write_manifest(directory / MANIFEST_NAME, kind="resume", **manifest_fields),
        what=f"save_resume manifest at {directory}",
    )
    _fsync_dir(directory)
    return directory


def load_resume(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a resume checkpoint in place.

    Refuses to load an incomplete checkpoint. Resuming from a truncated one is
    the failure mode the manifest marker exists to prevent, so this is a raise
    and not a warning.

    Returns the bookkeeping (`step`, `dataloader`, `extra`) for the caller to
    reinstall into the trainer and dataloader.
    """
    directory = Path(path)
    if not is_complete(directory):
        raise RuntimeError(
            f"{directory} is not a complete checkpoint (no valid {MANIFEST_NAME}). "
            "It was probably truncated by a failed or out-of-disk write; refusing "
            "to resume from it."
        )

    payload = torch.load(directory / _RESUME_STATE_NAME, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        restore_rng_state(payload["rng"])
    return {
        "step": payload["step"],
        "dataloader": payload["dataloader"],
        "extra": payload["extra"],
    }


def prune_resume(root: str | Path, *, keep: int) -> list[Path]:
    """Delete all but the `keep` newest *complete* resume checkpoints.

    Ordering is by the step recorded in the manifest, not by mtime: mtime lies
    after a filesystem copy or a restore from backup.

    Incomplete directories are removed regardless of recency -- they cannot be
    resumed from, and leaving them wastes the quota that caused the problem.
    Returns the paths removed.
    """
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")

    root = Path(root)
    if not root.is_dir():
        return []

    complete: list[tuple[int, Path]] = []
    removed: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not is_complete(child):
            shutil.rmtree(child)
            removed.append(child)
            continue
        with (child / MANIFEST_NAME).open() as fh:
            complete.append((json.load(fh)["step"], child))

    complete.sort(key=lambda pair: pair[0])
    for _, path in complete[:-keep] if keep < len(complete) else []:
        shutil.rmtree(path)
        removed.append(path)
    return removed

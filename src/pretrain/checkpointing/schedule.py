"""When to checkpoint, and what it will cost on disk.

Pure functions -- no torch, no trainer. They are the part of the checkpoint
policy that has to be reasoned about *before* a run starts, because the disk
budget is a hard constraint rather than a target: the shared quota was ~1.6 TB,
of which ~500 GB was available for these runs.

Two checkpoint kinds, with different economics:

  * **trajectory** -- bf16 weights only, HF-loadable, kept for the whole run.
    These are the scientific product: the published seedvar models shipped no
    intermediate checkpoints, and "when does the defect form" is the question
    this project exists to answer. ~1.2 GB each at 0.6B.
  * **resume** -- weights + optimizer + RNG, rolled to the most recent 1-2.
    Pure operational insurance; worth ~3x a trajectory point each, which is why
    they are not kept.
"""

from __future__ import annotations

__all__ = [
    "trajectory_steps",
    "estimate_disk_bytes",
    "BYTES_PER_PARAM_BF16",
    "BYTES_PER_PARAM_RESUME",
]

BYTES_PER_PARAM_BF16 = 2
"""Trajectory checkpoints store bf16 weights only."""

BYTES_PER_PARAM_RESUME = 2 + 4 + 4 + 4
"""Resume checkpoints: bf16 weights + fp32 master-ish optimizer state.

Adam keeps two fp32 moments per parameter regardless of the compute dtype, and
DCP writes an fp32 copy of the parameters alongside them. 14 bytes/param is the
conservative figure used for budgeting; it is deliberately not tuned down,
because underestimating this is how a run dies at 80% completion.
"""


def trajectory_steps(
    total_steps: int,
    *,
    first_step: int,
    uniform_every: int,
) -> list[int]:
    """Steps at which to write a trajectory checkpoint.

    Log-spaced early, uniform later. The early density is not a
    stylistic preference: the defect-formation window is expected to be early,
    and OLMoE's router-saturation analysis -- the closest published analogue to
    what this project intends -- depended on exactly those early points. A purely
    uniform schedule would spend most of its budget on the boring tail.

    Args:
        total_steps: length of the run.
        first_step: the first checkpoint, and the base of the doubling
            (e.g. 1000 -> 1000, 2000, 4000, 8000, ...).
        uniform_every: once doubling would exceed this stride, switch to
            uniform spacing at this interval.

    Returns the sorted steps, always including `total_steps` (the final model).

    No defaults: the schedule determines both the disk bill and which questions
    the trajectory can answer, so it is stated explicitly per run.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if first_step < 1:
        raise ValueError(f"first_step must be >= 1, got {first_step}")
    if uniform_every < 1:
        raise ValueError(f"uniform_every must be >= 1, got {uniform_every}")

    steps: list[int] = []
    step = first_step
    while step < total_steps and step < uniform_every:
        steps.append(step)
        step *= 2

    # Uniform tail. Start at the first multiple of `uniform_every` that is past
    # the log phase, so the two phases cannot emit a duplicate.
    tail = uniform_every
    while tail < total_steps:
        if not steps or tail > steps[-1]:
            steps.append(tail)
        tail += uniform_every

    steps.append(total_steps)
    return sorted(set(steps))


def estimate_disk_bytes(
    *,
    nparams: int,
    n_trajectory_points: int,
    n_resume_kept: int,
) -> dict[str, int]:
    """Total disk for one run, broken down by kind.

    Call this before launching and compare against the budget: hitting the quota
    mid-write does not fail cleanly, it leaves a half-written checkpoint that looks real.
    Cheaper to refuse to start.
    """
    if nparams < 1:
        raise ValueError(f"nparams must be >= 1, got {nparams}")
    if n_trajectory_points < 0 or n_resume_kept < 0:
        raise ValueError("checkpoint counts must be non-negative")

    trajectory = nparams * BYTES_PER_PARAM_BF16 * n_trajectory_points
    resume = nparams * BYTES_PER_PARAM_RESUME * n_resume_kept
    return {
        "trajectory_bytes": trajectory,
        "resume_bytes": resume,
        "total_bytes": trajectory + resume,
    }

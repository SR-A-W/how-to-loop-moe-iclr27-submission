"""Eval task-set selection for the SFT tiers (`sft_probe.py` + `sft_formal.py`).

Reuses `src.eval.tasks.LT2_ZERO_SHOT_TASKS` (the verified
paper-name -> lm-eval-harness-registered-name mapping, `src/eval/tasks.py`)
rather than redefining task names here -- a second list of the same names would
be a second place for the mapping to drift out of sync with lm-eval's actual
task registry (exactly the class of mistake that module's own docstring warns
about).

Two task sets, selected via the caller-required `task_set` string (no default.
An earlier boolean `--eval-include-optional` flag only ever toggled between a
3-task and a 5-task subset and NEVER reached the full 11-task LT2 suite, even
though `sft_formal.py`'s docs assumed it did -- a plausible-looking but
silently wrong configuration that must be caught before it reaches a real
cluster run, not after):

- `"probe"`: the fast 3-task subset (`arc_easy`/`piqa`/`hellaswag`) -- a probe
  SFT pass runs on a token budget in the tens-of-millions range
  or is a formal run's routine post-hoc health check; running the full 11-task suite (thousands of
  forward passes, some tasks with generation) would dwarf either use case in
  wall time. This is "did this checkpoint show any life", not exhaustive
  benchmark coverage -- that's what "full" is for.
- `"full"`: the complete 11-task LT2 zero-shot suite (`LT2_ZERO_SHOT_TASKS`) --
  needed when a benchmark comparison must actually have statistical power
  (use the full 11-task suite rather than the 3-task subset).

The previous 5-task tier (3 required + `copa`/`openbookqa` "optional") is
retired, not kept alongside the new two-value selector: no real invocation in
this project's history ever actually passed the old flag's true branch (only
`--no-eval-include-optional` ever appears in any committed command or
receipt), so there is no live behavior to preserve, and keeping a third,
unreachable-by-CLI tier around would just be dead code the next reader has to
puzzle over. If a middle tier is ever needed again, add it back deliberately
with its own name -- do not resurrect it silently under one of these two.
"""

from __future__ import annotations

from src.eval.tasks import LT2_ZERO_SHOT_TASKS

_KNOWN = set(LT2_ZERO_SHOT_TASKS)

# Fastest of the 11 to evaluate, and each is sensitive to whether basic
# language understanding / commonsense reasoning moved at all after SFT --
# exactly the axis a probe-tier or routine post-hoc check is looking at.
PROBE_TASKS: tuple[str, ...] = ("arc_easy", "piqa", "hellaswag")

for _name in PROBE_TASKS:
    if _name not in _KNOWN:
        raise ValueError(
            f"{_name!r} is not in src.eval.tasks.LT2_ZERO_SHOT_TASKS "
            f"({sorted(_KNOWN)}) -- this module's task subset must stay a "
            "subset of the verified mapping, never an independent name."
        )

EVAL_TASK_SETS: dict[str, tuple[str, ...]] = {
    "probe": PROBE_TASKS,
    "full": LT2_ZERO_SHOT_TASKS,
}


def resolve_eval_tasks(task_set: str) -> tuple[str, ...]:
    """The lm-eval task tuple to pass to `src.eval.run_eval`'s `--tasks`.

    `task_set`: no default -- a caller must state explicitly which of the two
    sets it wants (`"probe"` or `"full"`), not inherit a silent choice that
    could drift the reported task set between runs without anyone deciding to
    change it (same rationale as every other required-no-default parameter in
    this project, e.g. `sft_formal.py`'s `--save-steps`/`world_size`).
    """
    try:
        return EVAL_TASK_SETS[task_set]
    except KeyError:
        raise ValueError(
            f"task_set must be one of {sorted(EVAL_TASK_SETS)}, got {task_set!r}"
        ) from None

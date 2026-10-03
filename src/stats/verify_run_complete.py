"""Completion acceptance for a finished pretraining run: are all the trajectory
checkpoints there, and is each one whole?

## What this answers, and what it deliberately does not

Acceptance criterion 3 on the round-2 comparison runs is "16 trajectory
archives complete, token count reconciled". `summarize_run.py` already owns the
token reconciliation (three independent numbers, raising on disagreement), so
this tool CALLS it rather than restating it -- a second implementation of the
same arithmetic is how two artifacts drift apart.

What is not covered anywhere else, and is this tool's actual content, is
completeness:

  * every step the run's own recipe scheduled has a directory -- checked
    against `RunPlan.trajectory_checkpoints`, the schedule the run declared,
    not against a hard-coded 16;
  * each directory's manifest is present, parses, and carries every TIER-1
    field non-empty;
  * each manifest's architecture matches the run's registered architecture,
    field by field;
  * each manifest's `step` agrees with the directory it sits in.

## Why "the directory is there" is not the check

A checkpoint directory appears as soon as its first weight file is created and
is finished minutes later, when the manifest lands. Counting directories would
accept a run whose last checkpoint was still being written when the job died --
and the last checkpoint is the one everything downstream loads.

## Output

A report, printed and optionally written as JSON. Every failure is listed with
the checkpoint it belongs to; the tool does not stop at the first one, because
"which checkpoints are bad" is the question being asked.

    python -m src.stats.verify_run_complete --run-dir <dir> --run-name <name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataclasses import dataclass

from src.model.config_registry import build
from src.pretrain.manifest import MANIFEST_TIER1
from src.metrics.provenance import SOURCE_JOB_START_ENV
from src.stats.summarize_run import checkpoint_root

SOURCE_TRUSTED = SOURCE_JOB_START_ENV
"""The only source whose commit answers "what code ran". See
`_check_code_provenance`."""


@dataclass(frozen=True)
class ProvenanceWaiver:
    """A human record standing in for provenance the artifacts cannot carry.

    The three round-2 runs wrote `live_head` manifests for the segments
    launched before `CODE_REVISION_JSON` existed. What code those segments ran
    is recorded only in the job's own start-up log, on the cluster, and cannot
    be recovered from our artifacts at all. A waiver is how that external
    evidence enters the check.

    It is deliberately not a flag. `--provenance-waiver /path/to/file` requires
    somebody to have written down the segment-start commits and why they are
    acceptable, and the file must NAME THE RUN it waives -- otherwise one
    waiver would silently excuse every run it was pointed at, which is how a
    documented exception turns into a blanket one.
    """

    path: Path
    text: str

    @property
    def summary(self) -> str:
        """First non-empty, non-comment line -- enough to print in a report."""
        for line in self.text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line if len(line) <= 200 else line[:197] + "..."
        return "(waiver file has no readable content line)"


def load_waiver(path: Path, run_name: str) -> ProvenanceWaiver:
    """Read a waiver and refuse it unless it names `run_name`."""
    text = Path(path).read_text()
    if run_name not in text:
        raise ValueError(
            f"waiver {path} does not mention run {run_name!r}. A waiver must name the run "
            "it excuses -- one that does not would silently cover any run it is pointed "
            "at, which is the opposite of a recorded exception."
        )
    return ProvenanceWaiver(path=Path(path), text=text)

#: Architecture fields compared against the run's registered config. These are
#: the ones that identify WHICH experiment a checkpoint belongs to; a mismatch
#: means the artifacts and the run plan disagree about what was trained.
ARCHITECTURE_FIELDS = (
    "num_layers_in_stack",
    "num_stacks",
    "num_experts",
    "d_model",
    "num_active",
    "lb_loss_factor",
    "lz_loss_factor",
)


def _run_plan(run_name: str):
    from src.pretrain.recipe import RUN_LIST

    for run in RUN_LIST:
        if run.name == run_name:
            return run
    raise ValueError(
        f"run_name={run_name!r} is not in src.pretrain.recipe.RUN_LIST -- without the run's "
        "declared schedule there is no expected checkpoint list to verify against, and "
        "counting whatever happens to be on disk would accept any number of them."
    )


def check_checkpoints(
    run_dir: Path,
    run_name: str,
    *,
    expected_commit: str | None = None,
    waiver: ProvenanceWaiver | None = None,
) -> dict[str, Any]:
    """Per-checkpoint completeness against the run's declared schedule.

    `expected_commit`, when given, is the commit the task order specified. The
    run's manifests must all name it. Omitted, the manifests must still agree
    with each other -- "every checkpoint came from one code version" is checkable
    without knowing which version was intended.
    """
    run = _run_plan(run_name)
    expected = list(run.trajectory_checkpoints)
    # Through the plan: an ablation run is built by name from the registry, and
    # `build(arch, tier)` raises on those names.
    config = run.build_config().to_dict()

    found: list[int] = []
    problems: list[dict[str, str]] = []
    #: step -> (commit, code_revision_source), for the cross-checkpoint checks.
    seen_provenance: dict[int, tuple[Any, Any]] = {}
    resume_intervals: dict[int, Any] = {}

    # One definition of where a run's manifests live, shared with summarize_run.
    root = checkpoint_root(run_dir)

    for step in expected:
        # The training side writes 8-digit step directories; toy runs use 6.
        candidates = [root / f"step_{step:08d}", root / f"step_{step:06d}"]
        directory = next((c for c in candidates if c.is_dir()), None)
        if directory is None:
            problems.append({"step": str(step), "problem": "no checkpoint directory"})
            continue

        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            problems.append(
                {
                    "step": str(step),
                    "problem": "directory exists but no manifest.json -- the checkpoint was "
                    "never finished (the manifest is written last, as the completeness marker)",
                }
            )
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            problems.append({"step": str(step), "problem": f"manifest does not parse: {exc}"})
            continue

        found.append(step)

        missing = [f for f in MANIFEST_TIER1 if f not in manifest or manifest[f] in (None, "", [], {})]
        if missing:
            problems.append(
                {"step": str(step), "problem": f"TIER-1 field(s) missing or empty: {', '.join(missing)}"}
            )
        if manifest.get("step") != step:
            problems.append(
                {
                    "step": str(step),
                    "problem": f"manifest says step={manifest.get('step')!r} but sits in {directory.name}",
                }
            )
        seen_provenance[step] = (
            manifest.get("git_commit"),
            manifest.get("code_revision_source"),
        )
        # Recorded per checkpoint, not per run, and deliberately NOT reduced to
        # a single value here: one run legitimately carries two intervals when
        # it is resubmitted with a different one mid-run (e.g. 2000 before, 500
        # after). Collapsing that to one number would hide exactly the
        # thing this field was added to make visible.
        resume_intervals[step] = manifest.get("resume_checkpoint_interval_steps")
        model_config = manifest.get("model_config") or {}
        for field in ARCHITECTURE_FIELDS:
            if field not in model_config:
                problems.append({"step": str(step), "problem": f"model_config has no {field!r}"})
            elif model_config[field] != config[field]:
                problems.append(
                    {
                        "step": str(step),
                        "problem": (
                            f"model_config.{field}={model_config[field]!r} but the registered "
                            f"run declares {config[field]!r}"
                        ),
                    }
                )

    problems.extend(_check_code_provenance(seen_provenance, expected_commit, waiver))

    return {
        "run_name": run_name,
        "expected_steps": expected,
        "expected_count": len(expected),
        "found_steps": found,
        "found_count": len(found),
        "missing_steps": [s for s in expected if s not in found],
        "unexpected_steps": sorted(
            int(d.name.split("_")[1])
            for d in root.glob("step_*")
            if d.is_dir() and int(d.name.split("_")[1]) not in expected
        )
        if root.is_dir()
        else [],
        "resume_checkpoint_intervals": resume_intervals,
        "problems": problems,
        # A waived item is recorded in the report but does not fail the run --
        # that is what waiving means. Everything else does.
        "complete": (
            not [p for p in problems if p.get("severity") != "waived"]
            and len(found) == len(expected)
        ),
    }


def _check_code_provenance(
    seen: dict[int, tuple[Any, Any]],
    expected_commit: str | None,
    waiver: "ProvenanceWaiver | None" = None,
) -> list[dict[str, str]]:
    """Did every checkpoint come from one code version, and do we believe it?

    The two sources have to be judged differently; conflating them is a
    real defect.

    **`job_start_env`**: the commit was resolved when the job started, so it
    names the code the process actually loaded. Across such manifests, the set
    of commits IS evidence: more than one means more than one code version ran.

    **`live_head`**: the commit was read from the repository at the moment each
    checkpoint was written, while the Python modules were loaded once at job
    start and never reloaded. The set size then says nothing in either
    direction. A run on ONE code version shows several commits if anyone pushed
    while it ran; a run across TWO versions shows one commit if the repository
    happened to look the same at both writing moments. So these manifests are
    reported as untrusted, and no conclusion about version count is drawn from
    them -- stating one on top of a record already judged unreliable would be
    an overreach. The authority for those segments is the job's
    own start-up log, which records HEAD at launch and is not in our artifacts.

    A run may legitimately hold both kinds: the three round-2 runs write
    `live_head` for the segments launched before the fix and `job_start_env`
    after, and that mixture is expected, not a new fault.
    """
    problems: list[dict[str, str]] = []
    if not seen:
        return problems

    live = sorted(step for step, (_, source) in seen.items() if source == "live_head")
    trusted = {step: commit for step, (commit, source) in seen.items()
               if source == SOURCE_TRUSTED}

    if live:
        if waiver is not None:
            problems.append({
                "step": ", ".join(str(s) for s in live),
                "problem": (
                    f"{len(live)} checkpoint(s) are code_revision_source='live_head' -- "
                    f"MANUALLY WAIVED by {waiver.path.name}. Basis on record: "
                    f"{waiver.summary}"
                ),
                "severity": "waived",
            })
        else:
            problems.append({
                "step": ", ".join(str(s) for s in live),
                "problem": (
                    f"{len(live)} checkpoint(s) recorded code_revision_source='live_head': "
                    "the commit was read when the checkpoint was written, not when the job "
                    "started, so it names whatever HEAD happened to be at that moment. "
                    "These manifests cannot establish which code ran -- neither that it was "
                    "one version nor that it was several. The evidence for those segments is "
                    "the job's start-up log (it prints HEAD at launch); supply it via "
                    "--provenance-waiver, or export CODE_REVISION_JSON at job start so "
                    "future segments answer for themselves"
                ),
            })

    missing_source = sorted(step for step, (_, source) in seen.items() if source is None)
    if missing_source:
        if waiver is not None:
            # Same treatment as live_head, and for the same reason: what code
            # those segments ran is external evidence the artifacts cannot hold,
            # so a waiver is the only way it can enter. A check that cannot be
            # satisfied by the evidence that answers it is not a check, it is a
            # dead end.
            problems.append({
                "step": ", ".join(str(s) for s in missing_source),
                "problem": (
                    f"{len(missing_source)} checkpoint(s) have no code_revision_source "
                    f"field (written before it existed) -- MANUALLY WAIVED by "
                    f"{waiver.path.name}. Basis on record: {waiver.summary}"
                ),
                "severity": "waived",
            })
        else:
            problems.append({
                "step": ", ".join(str(s) for s in missing_source),
                "problem": (
                    "no code_revision_source field -- written before the field existed, so "
                    "the vintage of git_commit is unknown. Not a failure of the run, but "
                    "its provenance cannot be confirmed"
                ),
            })

    # Only trustworthy manifests support a conclusion about version count.
    commits = set(trusted.values())
    if len(commits) > 1:
        by_commit: dict[Any, list[int]] = {}
        for step, commit in sorted(trusted.items()):
            by_commit.setdefault(commit, []).append(step)
        detail = "; ".join(
            f"{str(c)[:12] if c else c!r} at steps {steps}" for c, steps in by_commit.items()
        )
        problems.append({
            "step": "all",
            "problem": (
                f"checkpoints resolved at job start name {len(commits)} different commits -- "
                f"the run did not execute one code version end to end: {detail}"
            ),
        })
    elif commits and expected_commit is not None:
        only = next(iter(commits))
        if only != expected_commit:
            problems.append({
                "step": "all",
                "problem": (
                    f"checkpoints resolved at job start name commit {str(only)[:12]}, but "
                    f"the task order specified {expected_commit[:12]}"
                ),
            })

    return problems


def format_report(report: dict[str, Any], reconciliation: dict[str, Any] | None, note: str | None) -> str:
    lines = [
        f"Run completion check -- {report['run_name']}",
        "=" * 60,
        f"trajectory checkpoints : {report['found_count']} of {report['expected_count']} complete",
    ]
    if report["missing_steps"]:
        lines.append(f"  missing steps        : {report['missing_steps']}")
    if report["unexpected_steps"]:
        lines.append(f"  unscheduled steps    : {report['unexpected_steps']}")

    # Printed always, including when every checkpoint agrees: the interesting
    # case (a run that changed interval mid-flight) is indistinguishable from
    # the ordinary one unless the ordinary one is shown too.
    intervals = report.get("resume_checkpoint_intervals") or {}
    if intervals:
        distinct = sorted({v for v in intervals.values() if v is not None})
        unrecorded = sorted(k for k, v in intervals.items() if v is None)
        if distinct:
            shown = ", ".join(str(v) for v in distinct)
            lines.append(f"resume ckpt interval   : {shown} steps")
            if len(distinct) > 1:
                # Not a problem: the training cluster retunes it against preemption and a
                # resubmission legitimately changes it partway through a run.
                by_value: dict[Any, list[int]] = {}
                for step, value in sorted(intervals.items()):
                    if value is not None:
                        by_value.setdefault(value, []).append(step)
                for value in sorted(by_value):
                    steps = by_value[value]
                    lines.append(
                        f"  {value} steps          : checkpoints {steps[0]}..{steps[-1]}"
                        f" ({len(steps)})"
                    )
        if unrecorded:
            # Written as "not recorded", never as the recipe default: a
            # manifest from before this field existed says nothing about what
            # its job used, and printing the constant there would invent an
            # answer.
            lines.append(
                f"  not recorded         : {len(unrecorded)} checkpoint(s) predate the field"
            )

    if reconciliation is not None:
        lines.append(
            f"token reconciliation   : declared {reconciliation['declared_tokens']:,} == "
            f"expected {reconciliation['expected_tokens_from_recipe']:,} (delta "
            f"{reconciliation['delta']})"
        )
    elif note:
        lines.append(f"token reconciliation   : NOT RUN -- {note}")

    if report["problems"]:
        lines.append("")
        failing = [p for p in report["problems"] if p.get("severity") != "waived"]
        waived = [p for p in report["problems"] if p.get("severity") == "waived"]
        if failing:
            lines.append(f"problems ({len(failing)}):")
            for p in failing:
                lines.append(f"  step {p['step']}: {p['problem']}")
        if waived:
            lines.append(f"waived ({len(waived)}):")
            for p in waived:
                lines.append(f"  step {p['step']}: {p['problem']}")

    lines.append("")
    verdict = "COMPLETE" if report["complete"] and reconciliation is not None else "NOT COMPLETE"
    lines.append(f"verdict: {verdict}")
    if report["complete"] and reconciliation is None:
        lines.append(
            "  (checkpoints are complete, but the verdict is NOT COMPLETE because token "
            "reconciliation did not run -- an unrun check is never a pass)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Verify a finished run's trajectory checkpoints.")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--run-name", required=True, help="must be a name in src.pretrain.recipe.RUN_LIST")
    ap.add_argument("--train-series", type=Path, default=None,
                    help="train_series.<run>.v1.jsonl; without it token reconciliation cannot run")
    ap.add_argument(
        "--expected-commit", default=None,
        help=(
            "the training-path commit the task order specified (40-hex). Every "
            "manifest must name it. Omit to check only that the manifests agree "
            "with each other"
        ),
    )
    ap.add_argument(
        "--provenance-waiver", type=Path, default=None,
        help=(
            "a file recording the segment-start commits and why the run's "
            "'live_head' manifests are acceptable. Downgrades them to waived "
            "instead of failing; the file must name the run"
        ),
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    waiver = (
        load_waiver(args.provenance_waiver, args.run_name)
        if args.provenance_waiver is not None
        else None
    )
    report = check_checkpoints(
        args.run_dir, args.run_name, expected_commit=args.expected_commit, waiver=waiver
    )

    reconciliation, note = None, None
    if args.train_series is None:
        note = "--train-series was not given"
    else:
        from src.stats.summarize_run import (
            _read_train_series,
            _read_trajectory_checkpoints,
            lookup_recipe_tokens_per_step,
            reconcile_tokens,
        )

        try:
            reconciliation = reconcile_tokens(
                _read_trajectory_checkpoints(args.run_dir),
                lookup_recipe_tokens_per_step(args.run_name),
                _read_train_series(args.train_series),
            )
        except ValueError as exc:
            note = str(exc)

    text = format_report(report, reconciliation, note)
    print(text)
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"checkpoints": report, "reconciliation": reconciliation, "note": note}, indent=2)
        )
    raise SystemExit(0 if report["complete"] and reconciliation is not None else 1)


if __name__ == "__main__":
    main()

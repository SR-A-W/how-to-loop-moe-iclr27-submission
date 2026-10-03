"""Per-run statistics summary.

## Inputs

An earlier design read `train_metrics.jsonl` straight out of a run
directory, but real runs' `dump_folder`s live on the training cluster, not
on the node this tool runs on. The tool therefore reads the artifact the
training cluster streams back instead: `train_series.<run_name>.v1.jsonl`
(downsampled every ~100 steps, each row carrying a real `wall_time_utc`). This tool
reads THAT plus trajectory checkpoint manifests (`trajectory/step_*/
manifest.json`) plus the run's own declared recipe (`src.pretrain.recipe.
RUN_LIST`, looked up by `run_name`).

## Token reconciliation is a three-way, hard-gate check

An earlier one-legged version (derive an "expected" token count from
`train_metrics.jsonl`'s own row count, i.e. circular against the same
manifest it was checking) is gone. The reconciliation is now three
genuinely independent numbers, and RAISES (message carries the numbers) the
moment any two disagree -- fail loudly rather than silently: a stats
file that disagrees with its own inputs must not exist on disk:

1. **`declared_tokens`** -- the latest trajectory checkpoint's own
   `tokens_consumed` (the training loop's bookkeeping).
2. **`expected_tokens_from_recipe`** -- `step * tokens_per_step`, where
   `tokens_per_step` comes from `src.pretrain.recipe.RUN_LIST[run_name].recipe.
   tokens_per_step` (`global_batch_seqs * seq_len`) -- the run's DECLARED
   hyperparameters, independent of anything the training loop wrote to disk.
   Any run whose `run_name` is not in `RUN_LIST` cannot be reconciled this
   way and this function raises rather than skipping the check.
3. **`train_series` must confirm the checkpoint's progress** -- its last
   row's `step` must be within one `sampled_every_steps` interval of the
   checkpoint's step (downsampling means an exact match isn't expected, but
   a train_series file that stops far short of the checkpoint's claimed
   step is a real inconsistency, not sampling noise).

A `reconciliation.ok` field would be structurally always `true` in any file
that exists (a failing reconciliation raises before any file is written) --
so it would carry zero information while looking like a passed check. `ok` is
NOT a field here; the three numbers (`declared_tokens`,
`expected_tokens_from_recipe`, `delta`) are, and their mere presence in a
written file already means they reconciled -- documented here, not implied.

Reconciliation requires a LOCAL trajectory
checkpoint manifest for leg 1 (`declared_tokens`). Real production runs'
`dump_folder`s live on the training cluster, not this node, so that manifest is
routinely just not present here -- the common case, not an error. When no
checkpoint is found, `tokens.reconciliation` is `null` with
`tokens.reconciliation_status = "unavailable"`, and the REST of the file
(loss/grad_norm/throughput, all derivable from `train_series` alone) is
still computed and written. The hard-raise behavior above still applies
whenever a checkpoint manifest IS found and disagrees.

## Throughput: real wall-clock deltas, segmented around resumes

`train_series` rows carry `wall_time_utc` per row. The row's own
`tokens_per_sec` field is NOT "the run's throughput": that field is each
training PROCESS's own, PER-GPU, framework-self-reported rate; using it as a run-wide
number silently under-reports by however many GPUs the run used. Kept, but
renamed to say what it is: `tokens_per_sec_per_gpu_median`.

The run-wide `token_per_s`/`step_per_s` are instead derived from consecutive
rows' real `wall_time_utc` deltas (`step_per_s`) times the recipe's declared
`tokens_per_step` (`token_per_s`) -- see `_wall_time_segments()`. Real
resumed runs re-log a region of steps after a resume (`step` AND
`wall_time_utc` both regress at the seam -- confirmed on 3 of the first 5
real `train_series` files this tool was validated against); diffing across
that seam would produce a nonsense rate, so rows are split into maximal
monotonic segments first and deltas are never taken across a split.
`basis="unavailable"` (not a fabricated rate) when no segment has 2+ rows.
The earlier mtime-based proxy (dev-node checkpoint directories get copied/
rsynced after the fact, flattening mtimes -- verified, not assumed) is gone
entirely; it is not one of `basis`'s two possible values
(`"train_series_wall_time"` / `"unavailable"`).

## Code provenance: `producer.code_revision`

Every artifact this project's tooling produces records ITS OWN generating
code's revision under the same key path, `producer.code_revision`, using
`src.metrics.provenance.code_revision()`'s exact `{commit, dirty,
diff_sha256, describe}` shape verbatim -- no per-tool renaming, so no
two-name translation layer exists anywhere. `checkpoint_code_revision` is a DIFFERENT, unrelated
field: the TRAINING code's revision at the checkpoint being summarized
(read from that checkpoint's own manifest fields), not this script's.

## data_mix is null for now -- and says so, distinctly from grad_norm's null

This project currently has exactly one data source (FineWeb-Edu); nothing
to compute a "mix ratio" against yet. `data_mix` is `null` with a sibling
`data_mix_status: "not_applicable"`, because a bare `null` doesn't distinguish "this concept doesn't apply yet"
(data_mix) from "this WAS applicable but nothing was recorded" (grad_norm's
`logged_count: 0`, a real sparse-data gap). Conflating the two would make a
future multi-source run's genuinely-missing mix data look identical to
today's not-yet-a-concept case.

## Deduplication before ANY stat is computed

Real train_series files from runs with resumes contain duplicate step
values (143-192 per run from 5-6 resume rewinds each, as measured).
`_dedupe_by_step()` keeps only the LAST occurrence (file order) of each step
-- the one that continued into the weights that actually survived -- before
computing loss/grad_norm/throughput/reconciliation. `steps.resume_rewinds`
(count of regression points in the raw file) and `steps.duplicate_steps`
(rows dropped by dedup) are both reported, so a reader can see the file had
resumes without needing to re-derive it from the raw data themselves.

## tokens_per_step_source: which of two possible origins the number is

`tokens.tokens_per_step` can come from either the run's DECLARED recipe
(`src.pretrain.recipe.RUN_LIST`, when no local checkpoint manifest exists) or
the checkpoint manifest itself (when one does, the manifest is authoritative:
it is what the training loop that produced the surviving
weights actually used, not merely what was declared). `tokens_per_step_
source` (`"recipe"` / `"manifest"`) says which, so a reader never has to
guess or assume.

## No absolute filesystem paths in the output

A produced artifact must never carry an absolute path (which may contain an
account name) -- those
get pasted into reports/task tickets/messages to collaborators, none of
which scrub an account name out the way a hosted page's `scrub()` can.
Every caller names which root `run_dir` sits under (`run_dir_root_name`,
e.g. `"gpu_smoke_root"`) and that root's absolute path
(`run_dir_root_path`, kept only in the caller's own local config, never
written out); the output stores the root's NAME plus `run_dir`'s path
RELATIVE to it (`run_dir_root`/`run_dir_rel`).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.metrics.provenance import code_revision

SCHEMA_VERSION = "1.1"
"""1.1 adds `input_window` -- see the payload comment where it is built."""
GENERATOR_NAME = "src/stats/summarize_run.py"
GENERATOR_VERSION = "0.2.0"


def _read_train_series(path: Path) -> list[dict[str, Any]]:
    """Every row of `train_series.<run_name>.v1.jsonl`, in FILE order.

    This function deliberately does NOT raise on a `step` regression, even
    though a consumer that pages through the series would treat a
    regression as a defective file. That view doesn't transfer to this
    tool: real `train_series` files have real
    step regressions from resumed training overlapping the region right
    before a resume point -- an EXPECTED artifact of resumed real runs, not
    file corruption. The monitoring page's job (flag it visibly) and this
    tool's job (produce correct aggregate numbers around it) are different
    problems; `_dedupe_by_step()` (below) is where regressions/duplicates
    are actually resolved before any stat is computed, not here.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found -- refusing to summarize a run with no train_series file.")
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            if rec.get("kind") != "train_series":
                raise ValueError(f"{path}:{lineno} has kind={rec.get('kind')!r}, expected 'train_series'.")
            rows.append(rec)
    if not rows:
        raise ValueError(f"{path} has no rows -- nothing to summarize.")
    return rows


def _dedupe_by_step(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    """Resolve resume-caused step regressions/duplicates BEFORE any stat is
    computed (real runs with resumes have 143-192 duplicate steps): keep only the LAST
    occurrence (file order) of each `step` value, then sort by step.

    Why last, not first: a resumed run re-logs a region of steps starting
    from its resume point; the LATER occurrence of a given step is the one
    that actually continued into the weights that survived (the earlier,
    pre-resume occurrence describes a training trajectory that was
    discarded). Keeping the first occurrence would silently summarize a
    training run that, in the surviving weights, never happened.

    Returns `(deduped_rows, resume_rewinds, duplicate_steps)`:
    - `resume_rewinds`: count of regression points in the ORIGINAL
      (pre-dedup) sequence -- indices where `step` strictly DECREASED from
      the row before it. A row repeating the same step as its predecessor
      is not counted here (that is what `duplicate_steps` already measures;
      counting it in both would count the same fact twice) -- matches
      `src.monitoring.build_page`'s own regression count (the two must
      agree on exactly this boundary). This is a count of REWIND EVENTS (how many
      times training resumed into an already-seen step region), not of
      individual affected rows.
    - `duplicate_steps`: rows dropped by dedup (`len(rows) - len(deduped)`)
      -- how many individual step values appeared more than once. (Example
      from real data: one run goes 2672 -> 2480 rows, i.e. 192 duplicate
      steps from 5 rewind events.)
    """
    resume_rewinds = 0
    prev_step: int | None = None
    for row in rows:
        if prev_step is not None and row["step"] < prev_step:
            resume_rewinds += 1
        prev_step = row["step"]

    last_by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        last_by_step[row["step"]] = row
    deduped = [last_by_step[step] for step in sorted(last_by_step)]
    duplicate_steps = len(rows) - len(deduped)
    return deduped, resume_rewinds, duplicate_steps


#: The two places a run's manifests are found, in the order they are tried.
#:
#: `trajectory/step_*/` is the training job's own output directory -- what
#: exists on the cluster while a run is alive. `step_*/` is the shape the
#: reflux writes into `stats/manifests/<run>/`, which is what this repository
#: actually holds once a run is over and its output directory is gone.
#:
#: Supporting both avoids having to build a symlink tree that impersonates a
#: run directory just to read reflux manifests.
MANIFEST_LAYOUTS = ("trajectory/step_*/manifest.json", "step_*/manifest.json")


def checkpoint_root(run_dir: Path) -> Path:
    """The directory whose `step_*` children are this run's checkpoints.

    `run_dir/trajectory` for a training output directory, `run_dir` itself for a
    reflux directory such as `stats/manifests/<run>/`. The training layout wins
    when both are present, so a run directory that happens to hold top-level
    `step_*` entries is never reinterpreted.

    This exists so that "where a run's manifests live" has ONE definition,
    shared with `verify_run_complete`. Two definitions of one fact do not stay
    equal; a second copy that misses the reflux layout would report every
    checkpoint missing for the directories the repository actually stores.
    """
    trajectory = run_dir / "trajectory"
    return trajectory if any(trajectory.glob("step_*")) else run_dir


def _read_trajectory_checkpoints(run_dir: Path) -> list[tuple[int, dict[str, Any], Path]]:
    """`(step, manifest_dict, manifest_path)` for every `kind="trajectory"`
    manifest under `run_dir`, sorted by step.

    Accepts either layout in `MANIFEST_LAYOUTS`; the first that matches wins,
    so a real run directory is never reinterpreted as a reflux directory.
    """
    pattern = next(
        (p for p in MANIFEST_LAYOUTS if any(run_dir.glob(p))), MANIFEST_LAYOUTS[0]
    )
    out: list[tuple[int, dict[str, Any], Path]] = []
    for manifest_path in sorted(run_dir.glob(pattern)):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("kind") != "trajectory":
            continue
        out.append((manifest["step"], manifest, manifest_path))
    out.sort(key=lambda t: t[0])
    return out


def lookup_recipe_tokens_per_step(run_name: str) -> int:
    """The run's DECLARED tokens/step (`global_batch_seqs * seq_len`), from
    `src.pretrain.recipe.RUN_LIST` -- independent of anything a checkpoint
    manifest says about itself. Raises if `run_name` isn't a registered run:
    a stats file cannot independently reconcile tokens for a run this
    project doesn't know the recipe of, and a silent skip would defeat the
    whole point of the check."""
    from src.pretrain.recipe import RUN_LIST

    for run in RUN_LIST:
        if run.name == run_name:
            return run.recipe.tokens_per_step
    raise ValueError(
        f"run_name={run_name!r} is not in src.pretrain.recipe.RUN_LIST -- cannot independently "
        "derive tokens/step from the declared recipe, so token reconciliation cannot proceed."
    )


def reconcile_tokens(
    checkpoints: list[tuple[int, dict[str, Any], Path]],
    tokens_per_step_from_recipe: int,
    train_series_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Three-way token reconciliation -- see module docstring. Raises
    `ValueError` (message carries the numbers) on any disagreement. The
    returned dict has NO `ok` field: a dict this function successfully returns already means
    every check passed, so an `ok` field inside it would be structurally
    always `True` and carry zero information.

    Callers: this function itself still requires `checkpoints` to be
    non-empty (raises if not) -- but `summarize_run()` only CALLS it when a
    checkpoint manifest was actually found locally. Real
    production runs' `dump_folder`s live on the training cluster, not this node, so a
    checkpoint manifest is frequently just not available here. That makes reconciliation `null`/`"unavailable"` for this run
    (not computable, not a failure), not a hard raise -- the rest of the
    stats file (loss, grad_norm, throughput) still gets written.
    """
    if not checkpoints:
        raise ValueError("no trajectory checkpoints found -- nothing to reconcile tokens against.")
    if not train_series_rows:
        raise ValueError("train_series has no rows -- nothing to confirm checkpoint progress against.")

    last_step, last_manifest, last_path = checkpoints[-1]
    declared_tokens = last_manifest["tokens_consumed"]
    expected_tokens = tokens_per_step_from_recipe * last_step
    delta = declared_tokens - expected_tokens
    if delta != 0:
        raise ValueError(
            f"token reconciliation failed: {last_path} declares {declared_tokens} tokens_consumed "
            f"at step {last_step}, but the declared recipe implies {tokens_per_step_from_recipe} "
            f"tokens/step -> expected {expected_tokens}. delta={delta} (positive means the "
            "manifest claims MORE tokens than the recipe's own hyperparameters account for)."
        )

    last_series_row = train_series_rows[-1]
    series_last_step = last_series_row["step"]
    sampled_every_steps = last_series_row["sampled_every_steps"]
    shortfall = last_step - series_last_step
    if shortfall > sampled_every_steps:
        raise ValueError(
            f"train_series does not confirm checkpoint progress: {last_path} declares step="
            f"{last_step}, but train_series's last row is only at step={series_last_step} "
            f"(sampled_every_steps={sampled_every_steps}) -- a gap of {shortfall} steps, more "
            "than one sampling interval can explain. Either the checkpoint is ahead of what was "
            "ever logged, or train_series stopped updating."
        )

    return {
        "declared_tokens": declared_tokens,
        "tokens_per_step_from_recipe": tokens_per_step_from_recipe,
        "expected_tokens_from_recipe": expected_tokens,
        "delta": delta,
        "train_series_last_step": series_last_step,
    }


def _series_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """First/last/min over `field` across ALL rows that have it non-null.
    `train_series` is downsampled (every `sampled_every_steps` steps), so
    "first"/"last" are the first/last SAMPLED point, not necessarily the
    run's true first/last training step -- that is the nature of a
    downsampled series, not a bug in this function."""
    values = [r[field] for r in rows if r.get(field) is not None]
    if not values:
        return {"first": None, "last": None, "min": None}
    return {"first": values[0], "last": values[-1], "min": min(values)}


def _grad_norm_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """`grad_norm` is a sparse optional field in train_series (contract:
    "null when not recorded, never 0") -- `mean`/`min`/`max` are `None` (not 0 or
    omitted) when no row carries it, so a reader can't mistake "never
    logged" for "logged as zero"."""
    values = [r["grad_norm"] for r in rows if r.get("grad_norm") is not None]
    if not values:
        return {"logged_count": 0, "mean": None, "min": None, "max": None}
    return {
        "logged_count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def _wall_time_segments(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split rows (in file order) into maximal runs where BOTH `wall_time_utc`
    and `step` strictly increase. Real resumed
    training runs re-log a region of steps after a resume, which regresses
    both `step` and `wall_time_utc` at the seam -- diffing straight across
    that seam would produce a nonsense (possibly negative or wildly
    inflated) rate. Splitting there and never diffing across the split is
    the fix; each segment is internally consistent and only within-segment
    deltas are used for rates."""
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prev_t: float | None = None
    prev_step: int | None = None
    for row in rows:
        if row.get("wall_time_utc") is None:
            continue
        t = _parse_iso(row["wall_time_utc"])
        if prev_t is not None and (t <= prev_t or row["step"] <= prev_step):
            if len(current) >= 2:
                segments.append(current)
            current = []
        current.append(row)
        prev_t, prev_step = t, row["step"]
    if len(current) >= 2:
        segments.append(current)
    return segments


def _throughput(rows: list[dict[str, Any]], tokens_per_step_from_recipe: int) -> dict[str, Any]:
    """Run-wide `token_per_s`/`step_per_s` from `wall_time_utc` deltas within
    monotonic segments (see `_wall_time_segments()`), NOT from train_series's
    own `tokens_per_sec` field.

    `tokens_per_sec` is each training
    process's PER-GPU, framework-self-reported rate -- using it directly as
    "the run's throughput" silently under-reports a multi-GPU run by however
    many GPUs it used. `token_per_s` here is instead `step_per_s *
    tokens_per_step_from_recipe` (the run-wide rate, from real wall-clock
    deltas x the recipe's declared tokens/step). The raw per-row field is
    kept as `tokens_per_sec_per_gpu_median`, named to say what it actually
    is so nobody reaches for it expecting a run-wide number.

    `basis="unavailable"` (not a fabricated rate) when no segment has 2+ rows
    -- i.e. every row's wall clock was null or came from a single-sample
    "segment" with nothing to diff against.
    """
    segments = _wall_time_segments(rows)
    step_rates: list[float] = []
    for segment in segments:
        for r0, r1 in zip(segment, segment[1:]):
            dt = _parse_iso(r1["wall_time_utc"]) - _parse_iso(r0["wall_time_utc"])
            step_rates.append((r1["step"] - r0["step"]) / dt)

    per_gpu_values = [r["tokens_per_sec"] for r in rows if r.get("tokens_per_sec") is not None]
    per_gpu_median = statistics.median(per_gpu_values) if per_gpu_values else None

    if not step_rates:
        return {
            "basis": "unavailable",
            "token_per_s": None,
            "step_per_s": None,
            "tokens_per_sec_per_gpu_median": per_gpu_median,
            "note": (
                "no usable wall_time_utc pair within a monotonic segment -- see "
                "_wall_time_segments(); tokens_per_sec_per_gpu_median (if not null) is still "
                "train_series's own per-row field, a PER-GPU rate, not a run-wide one."
            ),
        }

    token_rates = [r * tokens_per_step_from_recipe for r in step_rates]
    return {
        "basis": "train_series_wall_time",
        "token_per_s": {"median": statistics.median(token_rates), "tail": token_rates[-1]},
        "step_per_s": {"median": statistics.median(step_rates), "tail": step_rates[-1]},
        "tokens_per_sec_per_gpu_median": per_gpu_median,
        "note": (
            "token_per_s/step_per_s: wall_time_utc deltas within monotonic segments (resume "
            "regions split, never diffed across) x the recipe's declared tokens_per_step -- a "
            "run-wide rate. tokens_per_sec_per_gpu_median is train_series's own reported "
            "tokens_per_sec median: a PER-GPU framework-self-reported rate, NOT run-wide "
            "throughput -- do not sum or compare it directly against token_per_s."
        ),
    }


def _parse_iso(ts: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _checkpoint_code_revision(manifest: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The TRAINING code's revision at the checkpoint being summarized, in
    the SAME `{commit, dirty, diff_sha256, describe}` shape as `producer.
    code_revision`: one concept must not carry two sets of field names in
    the same file, or every consumer needs two parsing branches. `describe` is always `null` here: `write_manifest`'s
    contract never carried a `git describe` field, so there is nothing to
    put there, not a gap in this function.

    Returns `(None, "unavailable")` when the checkpoint's manifest predates
    the `code_dirty`/`code_diff_sha256` rollout (older manifests only have
    `git_commit`). This is deliberate, not an oversight: in the unified
    shape, `dirty=None`/`diff_sha256=None` MEANS "the working tree was
    clean" (`provenance.code_revision()`'s own documented semantics) -- so a
    genuinely UNKNOWN dirty state must never be represented as `None`
    inside that shape, or a reader would misread "we don't know" as "it was
    clean".
    """
    commit = manifest.get("git_commit")
    dirty = manifest.get("code_dirty")
    diff_sha256 = manifest.get("code_diff_sha256")
    if dirty is None:
        return None, "unavailable"
    return {"commit": commit, "dirty": dirty, "diff_sha256": diff_sha256, "describe": None}, "ok"


def _input_window(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What data this summary was computed from, as opposed to when it was run.

    `generated_at` answers "when was this file produced", which is not the same
    question as "what does it describe" -- and the two drift apart by however
    long the reflux lags (86-88 minutes as measured on real files). Without
    this field every number can be self-consistent while nothing in the file
    says which slice of the run it covered.

    `source_generated_at` is the train_series file's OWN last-row timestamp, not
    ours: it dates the input, so a stale input is visible even when this summary
    was regenerated seconds ago. `source_sha256` makes "same input" checkable
    rather than inferred -- two summaries that disagree can be told apart into
    "different data" and "different code" without guessing.

    `first_step`/`last_step` mirror `steps.first_sampled_step`/`last_sampled_step`
    deliberately: both are read from the same `rows` here, in one place, so they
    cannot diverge. They are repeated inside `input_window` so a consumer
    checking provenance has the whole window in one object rather than half of
    it here and half of it elsewhere.
    """
    import hashlib

    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        digest = None
    return {
        "first_step": rows[0]["step"] if rows else None,
        "last_step": rows[-1]["step"] if rows else None,
        "source_generated_at": rows[-1].get("generated_at") if rows else None,
        "source_sha256": digest,
    }


def summarize_run(
    run_dir: str | Path,
    *,
    train_series_path: str | Path,
    run_dir_root_name: str,
    run_dir_root_path: str | Path,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Build the `stats.<run_name>.v1.json` payload for one run.

    `run_dir_root_name`/`run_dir_root_path`: this project's rule is that no
    produced artifact writes an absolute (possibly account-name-bearing)
    filesystem path -- those get pasted into reports/task tickets/messages
    to collaborators, none of which scrub them the way a hosted page can.
    Every caller must therefore name which root `run_dir` sits under (e.g.
    `"gpu_smoke_root"` -> `/abs/path/to/_gpu_smoke`); this function stores
    only the root's NAME and `run_dir`'s path relative to it
    (`run_dir_root`/`run_dir_rel`) -- the root-name-to-absolute-path mapping
    itself stays in local, unproduced config, never in the artifact. Raises
    if `run_dir` is not actually under `run_dir_root_path` (a wrong mapping
    is a caller bug, not something to silently paper over).

    Raises (does not return a partial dict) if a LOCAL trajectory checkpoint
    manifest was found but its tokens disagree with the other two
    reconciliation legs -- see `reconcile_tokens()`. When no local checkpoint
    manifest exists at all (the common case for a real run), reconciliation
    is `null`/`"unavailable"` instead of a raise, and the rest of the file is
    still written. Every other gap (no grad_norm logged, throughput
    unavailable) is likewise an explicit null/empty field, never silently
    omitted.
    """
    run_dir = Path(run_dir).resolve()
    run_dir_root_path = Path(run_dir_root_path).resolve()
    train_series_path = Path(train_series_path)

    try:
        run_dir_rel = run_dir.relative_to(run_dir_root_path)
    except ValueError as exc:
        raise ValueError(
            f"run_dir={run_dir} is not under run_dir_root_path={run_dir_root_path} "
            f"(claimed root name {run_dir_root_name!r}) -- refusing to record a wrong mapping."
        ) from exc

    raw_series_rows = _read_train_series(train_series_path)
    # Resolve resume rewinds/duplicates BEFORE computing anything -- see
    # _dedupe_by_step()'s docstring. Every downstream computation (loss,
    # grad_norm, throughput, reconciliation) uses the DEDUPED series.
    series_rows, resume_rewinds, duplicate_steps = _dedupe_by_step(raw_series_rows)
    if run_name is None:
        run_name = series_rows[0]["run_name"]

    # Always attempted, independent of whether a checkpoint manifest is
    # available locally -- this is a declared-hyperparameter lookup, not
    # something read off disk from the run's own output.
    tokens_per_step_from_recipe = lookup_recipe_tokens_per_step(run_name)

    trajectory_checkpoints = _read_trajectory_checkpoints(run_dir)
    if trajectory_checkpoints:
        # Checkpoints found locally -> reconciliation is a hard gate again
        # (raises on disagreement, same as before). A checkpoint manifest,
        # once reconciled, is treated as the AUTHORITATIVE source for
        # tokens_per_step: it's what the training
        # loop that produced the weights actually used, whereas the recipe
        # is what was DECLARED -- reconciliation having already proven the
        # two agree, "manifest" is still the more authoritative label.
        reconciliation = reconcile_tokens(trajectory_checkpoints, tokens_per_step_from_recipe, series_rows)
        reconciliation_status = "ok"
        tokens_per_step = reconciliation["declared_tokens"] // trajectory_checkpoints[-1][0]
        tokens_per_step_source = "manifest"
        last_traj_manifest = trajectory_checkpoints[-1][1]
        checkpoint_code_revision, checkpoint_code_revision_status = _checkpoint_code_revision(last_traj_manifest)
    else:
        # Real production runs' dump_folders live on the training cluster,
        # not this node -- no local checkpoint manifest is the
        # COMMON case for a real run summarized here, not an error. There is
        # nothing to reconcile tokens against, so the field is explicitly
        # "unavailable" rather than the whole file refusing to exist.
        reconciliation = None
        reconciliation_status = "unavailable"
        tokens_per_step = tokens_per_step_from_recipe
        tokens_per_step_source = "recipe"
        checkpoint_code_revision, checkpoint_code_revision_status = None, "unavailable"

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "run_stats",
        "run_name": run_name,
        "run_dir_root": run_dir_root_name,
        "run_dir_rel": str(run_dir_rel),
        "generated_at": _now_iso(),
        # WHEN this file was produced. `input_window` below says WHAT it
        # describes; they are different questions and the reflux lag makes the
        # answers differ by over an hour in practice.
        "input_window": _input_window(train_series_path, series_rows),
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
        "producer": {"code_revision": code_revision()},
        "steps": {
            "sampled_row_count": len(series_rows),
            "first_sampled_step": series_rows[0]["step"],
            "last_sampled_step": series_rows[-1]["step"],
            "sampled_every_steps": series_rows[-1]["sampled_every_steps"],
            # Counted against the RAW (pre-dedup) train_series -- see
            # _dedupe_by_step(). resume_rewinds: how many times training
            # resumed into an already-logged step region. duplicate_steps:
            # how many individual rows that produced (dropped by dedup).
            "resume_rewinds": resume_rewinds,
            "duplicate_steps": duplicate_steps,
        },
        "tokens": {
            # Duplicated from reconciliation's own value on purpose (a copy
            # that can be cross-checked is safe -- a value
            # stored in two places that must always agree is a free consistency
            # check, not redundancy for its own sake; src/monitoring/validate_stats.py
            # enforces the two stay equal). tokens_per_step_source says which
            # of the two possible origins (see below) it actually is here.
            "tokens_per_step": tokens_per_step,
            "tokens_per_step_source": tokens_per_step_source,
            "_tokens_per_step_source_note": (
                "'manifest': a local trajectory checkpoint was found and reconciled -- its own "
                "declared_tokens/step is authoritative (what the training loop that produced the "
                "surviving weights actually used). 'recipe': no local checkpoint manifest -- the "
                "only available source is src.pretrain.recipe.RUN_LIST's DECLARED hyperparameters "
                "(global_batch_seqs x seq_len), not verified against anything the run itself wrote."
            ),
            "reconciliation": reconciliation,
            "reconciliation_status": reconciliation_status,
            "_reconciliation_note": (
                "no 'ok' field: when reconciliation is not null, this file only exists because "
                "all three reconciliation checks already passed (a disagreement raises before "
                "any write) -- its presence on disk IS the passing result. reconciliation is "
                "null with reconciliation_status='unavailable' when no trajectory checkpoint "
                "manifest was found locally (the common case for a real run: its dump_folder "
                "lives on the training cluster, not the node this tool ran on) -- that is a coverage gap, not a "
                "failed check, and does not block the rest of this file from being written."
            ),
        },
        "loss": _series_stats(series_rows, "loss"),
        "grad_norm": _grad_norm_stats(series_rows),
        "throughput": _throughput(series_rows, tokens_per_step),
        "checkpoints": {
            "trajectory": {
                "count": len(trajectory_checkpoints),
                "steps": [s for s, _, _ in trajectory_checkpoints],
            },
        },
        "checkpoint_code_revision": checkpoint_code_revision,
        "checkpoint_code_revision_status": checkpoint_code_revision_status,
        "data_mix": None,
        "data_mix_status": "not_applicable",
    }


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_stats(
    run_dir: str | Path,
    *,
    train_series_path: str | Path,
    run_dir_root_name: str,
    run_dir_root_path: str | Path,
    run_name: str | None = None,
    out: str | Path | None = None,
) -> Path:
    """Compute and atomically write `stats.<run_name>.v1.json` (temp file +
    rename -- same convention as `src/eval/run_eval.py` -- a reader must
    never see a truncated file)."""
    payload = summarize_run(
        run_dir,
        train_series_path=train_series_path,
        run_dir_root_name=run_dir_root_name,
        run_dir_root_path=run_dir_root_path,
        run_name=run_name,
    )
    out_path = Path(out) if out is not None else Path(run_dir) / f"stats.{payload['run_name']}.v1.json"
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(out_path)
    return out_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir", required=True,
        help=(
            "directory holding the run's manifests, in either layout: a training "
            "output directory (trajectory/step_*/manifest.json) or a reflux "
            "directory such as stats/manifests/<run>/ (step_*/manifest.json)"
        ),
    )
    ap.add_argument(
        "--train-series", required=True,
        help="Path to train_series.<run_name>.v1.jsonl (the per-run training series, one row per ~100 steps)",
    )
    ap.add_argument(
        "--run-dir-root-name", required=True,
        help='A name for the root run_dir sits under (e.g. "gpu_smoke_root") -- stored in the '
             "output instead of any absolute path (no account-name-bearing absolute paths in "
             "produced artifacts).",
    )
    ap.add_argument(
        "--run-dir-root-path", required=True,
        help="The absolute path --run-dir-root-name names. NOT written to the output file.",
    )
    ap.add_argument("--run-name", default=None, help="Default: the train_series file's own run_name field")
    ap.add_argument("--out", default=None, help="Default: <run-dir>/stats.<run_name>.v1.json")
    args = ap.parse_args(argv)

    out_path = write_stats(
        args.run_dir,
        train_series_path=args.train_series,
        run_dir_root_name=args.run_dir_root_name,
        run_dir_root_path=args.run_dir_root_path,
        run_name=args.run_name,
        out=args.out,
    )
    print(f"[summarize_run] wrote {out_path}")


if __name__ == "__main__":
    main()

"""Checkpoint manifest contract.

Adapts the THINKING behind the earlier diagnostics code base's `collect/schema.py`'s
`TRACE_META_CONTRACT` (TIER-1 machine-checked / TIER-2 human-read split,
guarded by a test that must fail before it can pass -- a check that has
never failed proves nothing) to what a TRAINING checkpoint needs to record, not what an offline
diagnostic trace needs. The two field sets are deliberately NOT the same:
`TraceMeta` carries collection-layer facts (capture points, activation
tensors, affine-probe results) this project's training loop has no use for;
this module only carries what a checkpoint needs -- step/token accounting,
dtype provenance, data position, RNG-replay guarantee, and a pointer to the
offline-only perturbation probe (the expensive tier never runs in-loop).

## `kind`: two checkpoint species, one contract

The disk budget splits checkpoints into two species that this manifest must
tell apart:

- `"trajectory"` -- bf16 weights only, HF format, kept for the ENTIRE run
  (the format the diagnostics pipeline ingests). Deliberately carries
  **no RNG state** -- storing it would mean it is no longer "just the
  weights", and trajectory checkpoints are never resumed from.
- `"resume"` -- weights + optimizer + RNG family + data position, kept for
  only the last 1-2 (rolling). RNG state is mandatory here -- this is the
  only species the replay-identical-resume guarantee applies to.

`rng_state_saved=False` on a `"trajectory"` manifest is not a documented gap
-- it is the CORRECT, honest value for that species. The original
unconditional "`rng_state_saved` must be `True`" rule (see git history) was
right for what existed when it was written, but became wrong for
trajectory checkpoints once they existed as a category; `write_manifest`
now enforces the rule PER KIND instead of dropping it, with a reverse check
in the other direction too (a `"trajectory"` manifest claiming
`rng_state_saved=True` is *itself* raised on -- that combination almost
certainly means a resume checkpoint got mislabeled as a trajectory one,
which would silently blow the disk budget by ~7x: resume checkpoints are
~14 bytes/param, trajectory ones ~2).

## `resume_checkpoint_interval_steps`

How often the job that wrote this checkpoint was writing **resumable**
checkpoints, in steps, read off the live torchtitan config.

It exists because the interval is not a property of the recipe. It is retuned
against how hard the cluster is preempting (for example 2000 -> 500), and the
change used to be applied by editing the generated launch command. That left
nothing on disk recording what any given job actually used:
`recipe.RESUME_CHECKPOINT_INTERVAL_STEPS` describes the *suggestion*, and
reading it back would answer a question nobody asked.

Recorded **per checkpoint, not per run**, and that is deliberate: a run
resubmitted mid-flight legitimately has checkpoints at 2000 before the switch
and 500 after. A single per-run value could only be right by discarding one of
them.

**Optional, null when absent.** Every manifest written before this date
predates the field; back-filling the recipe constant would make those
checkpoints assert an interval nobody verified. `verify_run_complete` prints
such checkpoints as "not recorded" rather than as the default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The only two valid values for `kind` -- see the module docstring's
# "two checkpoint species" section for what distinguishes them.
MANIFEST_KINDS: tuple[str, ...] = ("trajectory", "resume", "coldstart")
"""What a checkpoint is for.

`coldstart` is the state BEFORE the first optimizer step: the shared
initialisation two clusters start the same configuration from, instead of each
re-deriving it from a seed. It is a distinct kind rather than a `resume` at step 0
because the questions differ -- a resume point is asked "where was the run", a
cold start is asked "which seed and which code made these weights", and only the
second is answerable for a checkpoint that has never consumed a token.

Readers that record `manifest["kind"]` verbatim (the offline probes) need no
change; anything that branches on the value should treat an unknown kind as
unknown rather than as one of the two it knows."""

# TIER-1: every field here must be present AND non-empty in a written
# manifest, or `write_manifest` raises (see the per-field checks below --
# "present" and "non-empty" are two different guarantees, both enforced).
# `rng_state_saved` is checked against `kind`, not against a fixed "must be
# True" rule -- see the per-field loop below and the module docstring.
MANIFEST_TIER1: tuple[str, ...] = (
    "step",
    "kind",
    "tokens_consumed",
    "git_commit",
    "compute_dtype",
    "storage_dtype",
    "autocast",
    "torch_version",
    "transformers_version",
    "model_config",
    "data_position",
    "rng_state_saved",
    "perturbation_probe",
)


def write_manifest(
    path: str | Path,
    *,
    step: int,
    kind: str,
    tokens_consumed: int,
    git_commit: str,
    compute_dtype: str,
    storage_dtype: str,
    autocast: dict[str, Any],
    torch_version: str,
    transformers_version: str,
    model_config: dict[str, Any],
    data_position: dict[str, Any],
    rng_state_saved: bool,
    perturbation_probe: dict[str, Any],
    learning_rate: float | None = None,
    lr_schedule: dict | None = None,
    max_seq_len: int | None = None,
    epochs_completed: float | None = None,
    code_dirty: bool | None = None,
    code_diff_sha256: str | None = None,
    run_name: str | None = None,
    code_revision_source: str | None = None,
    resume_checkpoint_interval_steps: int | None = None,
    activation_checkpoint_mode: str | None = None,
    tokenizer_name: str | None = None,
    arch: str | None = None,
    seed: int | None = None,
    training_steps: int | None = None,
    steps_overridden: bool | None = None,
) -> None:
    """Validate the TIER-1 contract, then write `path` as JSON.

    Every argument is a required keyword (no defaults: semantic parameters
    never get one). A missing argument is a `TypeError` from
    Python itself (call-site bug, caught immediately); a present-but-empty
    value is a `ValueError` raised HERE (a manifest that "has the field" but
    the field carries no information is the failure mode this contract
    exists to catch -- see `schema.py`'s own note that a contract test locks
    "field present and non-empty", not just existence).

    Field semantics:
        kind: `"trajectory"` or `"resume"` -- see the module docstring's
            "two checkpoint species" section. Any other value raises
            immediately (not silently treated as one of the two).
        compute_dtype: dtype the model forward actually computed in (from
            model parameters), e.g. `"bfloat16"`.
        storage_dtype: dtype the checkpoint itself is written in on disk --
            NOT assumed equal to `compute_dtype` (optimizer state is
            routinely fp32 under bf16 compute; conflating the two produced
            a real false-DISAGREE incident in that code base).
        autocast: `{"enabled": bool, "dtype": str | None}` -- explicit even
            when autocast is off (`{"enabled": False, "dtype": None}`), not
            omitted, so "we don't know" and "it's off" are never confused.
        data_position: which data shard/offset training had consumed up to
            this checkpoint (determinism requirement -- must be enough
            to resume the exact reading position).
        rng_state_saved: checked AGAINST `kind`, not against a fixed rule
            (see module docstring):
              - `kind="resume"` and `rng_state_saved=False` raises. Resume
                requires the full RNG family (python/numpy/torch/cuda) for
                replay-identical resume; a resume manifest admitting it
                wasn't saved is a broken resume promise, caught here rather
                than at the (much later, much more expensive) resume attempt.
              - `kind="trajectory"` and `rng_state_saved=True` ALSO raises
                (the reverse check) -- a trajectory checkpoint is weights
                only by definition; one claiming to carry RNG state most
                likely means a resume checkpoint was mislabeled, which would
                silently blow the disk budget (~7x per param) since trajectory
                checkpoints
                are kept for the entire run.
              - `kind="trajectory"` and `rng_state_saved=False` is the
                correct, expected value -- no error.
        perturbation_probe: pointer to the offline-only expensive-tier probe
            -- `{"status": "not_run", "offline_script": ...,
            "input_ids_sha8": ...}` when not yet executed. This field
            existing and being non-empty does NOT mean the probe ran; it
            means the manifest honestly says whether it did (`status` is
            part of TIER-2 semantics this project doesn't machine-check the
            *value* of, only that the pointer itself is present).
        learning_rate: the actual learning rate this run used, or `None` if
            not applicable/tracked for this call site (needed for
            `src.posttrain.sft_formal`'s era-1/era-2 comparison, where the
            checkpoint's OWN directory name is not a reliable record of
            which lr actually ran -- see that module's docstring). Written
            explicitly (including when `None`) rather than omitted, same
            "explicit even when off" convention `autocast` already uses --
            deliberately NOT in `MANIFEST_TIER1`: unlike every other field
            here, a single scalar lr is not meaningful for every checkpoint
            kind this contract covers (e.g. `src.posttrain.sft_probe`'s probe
            tier does not pass it), so this is the one genuinely optional
            field rather than a "no defaults" violation.
        lr_schedule: the shape of the learning-rate schedule this run used
            -- `{"warmup_steps", "decay_ratio", "decay_type", "min_lr_factor",
            "peak_lr"}` -- or `None` for a call site where it does not apply.
            Added with the move from cosine to warmup-stable-decay, because
            `learning_rate` alone cannot distinguish the two: a
            checkpoint at step 30,000 of a constant-LR stretch and one at
            step 30,000 of a cosine decay can record the same scalar and have
            trained under entirely different schedules. Read off the LIVE
            `job_config.lr_scheduler`, never from the recipe constant, for the
            same reason `resume_checkpoint_interval_steps` is: the manifest has
            to describe the job that ran, not the recipe someone meant to run.

            **Absent in manifests written before this field existed**, and
            readers must treat it that way: those runs were cosine, but this field
            says so only where it is present. Optional for the same reason as
            `learning_rate`/`max_seq_len` -- not every checkpoint kind this
            contract covers has one -- so it is deliberately NOT in
            `MANIFEST_TIER1`, and adding it there would make every existing
            manifest fail validation.
        max_seq_len: the actual max sequence length this run trained/
            evaluated with, or `None` if not applicable/tracked for this
            call site (previously this value only existed buried inside `model_config["context_length"]`,
            which is the checkpoint's architectural limit, not necessarily
            what this specific run actually used -- e.g. `sft_formal.py`
            currently derives one from the other, but a future call site
            need not). Same "genuinely optional, not a `MANIFEST_TIER1`
            no-defaults violation" rationale as `learning_rate` -- not every
            checkpoint kind this contract covers has a single meaningful
            sequence length (e.g. a pretraining "resume" checkpoint's
            manifest has no SFT-style run to describe).
        epochs_completed: how many full passes over the training corpus this
            run actually completed by the time this manifest was written
            (e.g. `trainer.state.epoch` after `trainer.train()` returns --
            the ACTUAL value, not the recipe's intended epoch count), or
            `None` if not applicable/tracked. `step: 0` in formal-SFT manifests (single-pass, not
            step-numbered like pretraining) previously left epoch progress
            with no field to record at all. Same optional-field rationale as
            `learning_rate`/`max_seq_len`.
        code_dirty: whether the working tree had uncommitted changes (tracked
            or untracked) at the moment this manifest's `git_commit` was
            read, or `None` if not tracked/applicable for this call site
            (`git_commit` alone answers "what's checked out", not "did it
            actually match what ran" -- a debug edit left uncommitted makes
            the two disagree with no way to tell). Written explicitly
            (including when `None`) -- same "genuinely optional, not a
            `MANIFEST_TIER1` no-defaults violation" rationale as
            `learning_rate`: pair with `src.metrics.provenance.
            code_revision()`, whose `dirty` field this is meant to carry.
        code_revision_source: `provenance.SOURCE_JOB_START_ENV` or
            `SOURCE_LIVE_HEAD` -- see the field's comment below. `None` when
            the writer did not record it; `git_commit` is then of unknown
            vintage, which is exactly the ambiguity this field removes.
        run_name: the run's own name from `src.pretrain.recipe.RUN_LIST` (e.g.
            `noloop_moe_1p3b_lb0p02`), taken from the launch argument, NOT from
            the output directory's name -- a checkpoint moved or uploaded
            elsewhere keeps its identity. `None` (written as null) when not
            supplied; never defaulted to the architecture name.
        code_diff_sha256: sha256 fingerprint of the uncommitted diff (see
            `code_revision()`'s own docstring for exactly what it hashes),
            or `None` when `code_dirty` is `False`/`None`/untracked for this
            call site. Same optional-field rationale as `code_dirty`.
        tokenizer_name: which of `src.data.tokenizer.TOKENIZER_CLASSES`
            (`"llama3"`/`"smollm2"`) this checkpoint's fine-tuning/eval run
            actually loaded, or `None` if not applicable/tracked for this
            call site (`src.posttrain.sft_probe`/`sft_formal` require the
            caller to name a tokenizer explicitly rather than silently
            defaulting to Llama-3 -- this field is where that choice becomes
            visible in the artifact instead of only in the CLI invocation
            that produced it). Same optional-field rationale as
            `learning_rate`: not every checkpoint kind this contract covers
            has a single meaningful tokenizer choice (e.g. a pretraining
            checkpoint's own manifest predates this field and has no SFT-
            style run to describe).
    """
    if kind not in MANIFEST_KINDS:
        raise ValueError(f"kind must be one of {MANIFEST_KINDS}, got {kind!r}.")

    fields: dict[str, Any] = {
        "step": step,
        "kind": kind,
        "tokens_consumed": tokens_consumed,
        "git_commit": git_commit,
        "compute_dtype": compute_dtype,
        "storage_dtype": storage_dtype,
        "autocast": autocast,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "model_config": model_config,
        "data_position": data_position,
        "rng_state_saved": rng_state_saved,
        "perturbation_probe": perturbation_probe,
        "learning_rate": learning_rate,
        "lr_schedule": lr_schedule,
        "max_seq_len": max_seq_len,
        "epochs_completed": epochs_completed,
        "code_dirty": code_dirty,
        "code_diff_sha256": code_diff_sha256,
        # Which RUN produced this checkpoint, not which architecture. Written
        # as null when not supplied (this file's convention for optional
        # fields) rather than filled in with the architecture name: two runs
        # can share an architecture and differ only in a hyperparameter
        # (noloop_moe_1p3b and noloop_moe_1p3b_lb0p02 differ only in
        # lb_loss_factor), so an architecture name here would look like an
        # answer while being ambiguous. null says "not recorded"; a wrong name
        # says something false.
        "run_name": run_name,
        # How often THIS run was writing resumable checkpoints, in steps, read
        # off the live torchtitan config rather than from
        # `recipe.RESUME_CHECKPOINT_INTERVAL_STEPS`. The constant is a default
        # suggestion; the real value is retuned against cluster preemption
        # (e.g. 2000 -> 500) and used to be changed by editing the
        # generated command, which left nothing on disk saying what the job
        # actually did. One run's checkpoints can therefore legitimately carry
        # different values before and after a resubmission -- that is the point
        # of recording it per checkpoint rather than per run.
        #
        # Optional, null when not supplied: older manifests predate the
        # field, and back-filling a guess would make
        # those checkpoints claim an interval nobody verified.
        "resume_checkpoint_interval_steps": resume_checkpoint_interval_steps,
        # Which activation-checkpointing mode the job actually ran with. Read off
        # the live config, never from `recipe.ac_mode_for_depth`: the rule states
        # what a run SHOULD use, and a manifest that records the rule rather than
        # the job cannot answer the one question it is asked -- what this
        # checkpoint was produced by. "selective" on a D<=2 architecture means no
        # layer was wrapped at all, which is exactly the state that OOM'd an earlier
        # run, so this field is how that failure becomes visible in the
        # artifact instead of only in a memory graph.
        #
        # Optional, null when not supplied: manifests written before this field
        # existed have no value, and back-filling the rule's answer would make
        # them claim a mode nobody verified.
        "activation_checkpoint_mode": activation_checkpoint_mode,
        # Which tokenizer this run actually loaded ("llama3"/"smollm2"), or null
        # when not supplied -- see this field's docstring above for why it is
        # optional rather than a MANIFEST_TIER1 no-defaults violation.
        "tokenizer_name": tokenizer_name,
        # Which configuration produced this checkpoint. `run_name` is the run's
        # identity and `arch` is what it was built from: for the DVF family they
        # differ (`dvf_a_0p6b` is `dvf_a` at a tier), for the ablation family they
        # are the same name by construction. Recorded because a checkpoint that
        # has left its directory is otherwise a set of shapes -- `model_config`
        # carries d_model and the layer counts, but nothing says which entry of
        # the registry those came from.
        "arch": arch,
        # Model initialization seed (torchtitan's live debug.seed), separate
        # from the data-order seed recorded in data_position. None means it was
        # not explicitly configured or was not recorded by an older writer.
        "seed": seed,
        # How long the run was launched for, and whether that matched its
        # registered plan. The LR schedule is derived from the step count, so a
        # shortened run differs from the full one at every step -- not only at
        # the end -- and nothing else in the manifest reveals it.
        "training_steps": training_steps,
        "steps_overridden": steps_overridden,
        # Where `git_commit`/`code_dirty`/`code_diff_sha256` came from:
        # "job_start_env" (resolved when the job started) or "live_head" (read
        # from the repository at the moment this manifest was written).
        # They answer different questions, and only the first is reliable while
        # a job runs against a clone anyone may commit to -- one earlier run wrote
        # sixteen checkpoints under "live_head" and none of them named the
        # commit it was launched from. Recorded so a reader can tell which
        # question a given manifest answered instead of assuming.
        "code_revision_source": code_revision_source,
    }

    for field in MANIFEST_TIER1:
        value = fields[field]
        if field == "kind":
            continue  # already validated above
        if field == "rng_state_saved":
            if kind == "resume" and value is not True:
                raise ValueError(
                    f"rng_state_saved must be True when kind='resume' -- got {value!r}. "
                    "A resume checkpoint manifest that admits its RNG state wasn't saved "
                    "cannot promise replay-identical resume; fix RNG saving before "
                    "writing this manifest, do not write a manifest that documents the gap."
                )
            if kind == "coldstart" and value is not False:
                raise ValueError(
                    f"rng_state_saved must be False when kind='coldstart' -- got {value!r}. "
                    "A cold-start checkpoint is the state before the first step; there is no "
                    "RNG stream to resume, and claiming one invites a reader to treat it as a "
                    "resume point."
                )
            if kind == "trajectory" and value is not False:
                raise ValueError(
                    f"rng_state_saved must be False when kind='trajectory' -- got {value!r}. "
                    "A trajectory checkpoint is weights-only by definition; one claiming to "
                    "carry RNG state most likely means a resume checkpoint was mislabeled as "
                    "a trajectory one, which would silently blow the disk budget (trajectory "
                    "checkpoints are kept for the entire run at ~7x smaller size)."
                )
            continue
        if value is None or value == "" or value == {} or value == []:
            raise ValueError(
                f"MANIFEST_TIER1 field {field!r} is present but empty ({value!r}). "
                "TIER-1 guarantees 'present AND non-empty' -- an empty value defeats "
                "the contract while looking like it's satisfied."
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=2, sort_keys=True))

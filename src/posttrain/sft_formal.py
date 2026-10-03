"""Formal instruction-tuning launcher: `checkpoint -> full-corpus SFT (SmolTalk
`smol-smoltalk`, the ENTIRE train split, not the probe's token-budget prefix)
-> eval`.

This is the caller of `src/posttrain/data.py::load_full_dataset` and the
launcher for the formal tier: without it, the three formal `-instruct` runs
could not be submitted through any existing script.

## Why a separate module, not a `src/posttrain/sft_probe.py` CLI flag

`src/posttrain/sft_probe.py`'s whole design point (its own module docstring:
"The probe recipe ... is fixed, not a CLI knob") is that the probe's SFT
recipe (`PROBE_RECIPE`) is fixed and NOT CLI-overridable, so every probe run
is comparable to every other. The formal tier needs its OWN fixed recipe
(`FORMAL_RECIPE` below) that is deliberately DIFFERENT from `PROBE_RECIPE`
(lr 3e-4 vs 2e-5, 2 epochs vs 1, warmup 3% vs 0). Adding
a `--lr`/`--epochs` override flag to `sft_probe.py` to reach these values
would violate that script's own "not a CLI knob" invariant; keeping the two
scripts separate means the probe stays exactly what it already is
while the formal launcher gets its
own fixed recipe under the exact same "recipe is fixed, not a CLI knob"
rule -- for an even stronger reason here: the three formal runs' hyperparameters
must be exactly identical, a hard cross-run comparability requirement, not a guideline,
so nothing about the recipe may vary between invocations by accident.

`--batch-size` and `--dtype` ARE required CLI arguments here (not part of
the fixed recipe) because both are open, per-checkpoint-independent
decisions: batch size depends on the multi-GPU dry-run's memory
measurement, and dtype has no long-standing prior default to preserve
the way `sft_probe.py`'s does (that script's `--dtype` defaults to float32
specifically to preserve its earlier behavior) -- this is
a brand-new script, so there is nothing to preserve, and bf16 vs float32 is
exactly the kind of consequential choice that must be a required parameter
with no default.

`--lr`: also a required CLI argument, NOT part of `FORMAL_RECIPE`
-- added when a rerun at a different learning rate (lr=2e-5) had no code
path for that value, so `FORMAL_RECIPE["lr"]=3e-4`
would have silently produced another 3e-4 copy under the new run's directory
name. `lr` is deliberately removed from `FORMAL_RECIPE` (not left there
alongside a CLI override) to avoid a two-source-of-truth state where the
dict says one number and the CLI says another with no way to tell which
one actually ran -- the CLI value is now the single source, and the
manifest records whatever it actually was (see `run_formal_sft`'s
docstring) so a saved checkpoint is self-describing regardless of which
lr produced it. This is a narrower, formal-tier-only decision: the
reasoning above for why `sft_probe.py` still does NOT get a
`--lr`/`--epochs` override remains valid for the probe tier -- that
script's fixed `PROBE_RECIPE` is untouched by this change.

Shares its checkpoint-loading, DDP main-process gating, and eval-shelling
plumbing with `sft_probe.py` (imported, not re-derived) so the two scripts
cannot silently drift apart on those concerns.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from src.pretrain.checkpointing.store import is_complete, save_trajectory
from src.data.tokenizer import TOKENIZER_CLASSES, load_named_tokenizer
from src.metrics.provenance import code_revision
from src.posttrain.chat_template import MINIMAL_CHAT_TEMPLATE
from src.posttrain.data import load_full_dataset
from src.posttrain.sft_probe import (
    INSTRUCTION_PROBE_RESULTS_NAME,
    _release_cuda_memory_before_eval,
    _resolve_is_main_process,
    build_shifted_labels_sft_trainer_class,
    run_instruction_probes,
    run_lm_eval,
)

# Fixed formal-tier SFT recipe (reviewed hyperparameters). Not CLI-exposed --
# see this module's docstring for why. Deliberately
# DIFFERENT from `src.posttrain.sft_probe.PROBE_RECIPE` (that recipe is for a
# tens-of-millions-token diagnostic probe, this one is for training on the
# full ~392M-token `smol-smoltalk` corpus for 2 epochs, per a measured
# token count).
FORMAL_RECIPE = {
    # "lr" deliberately REMOVED from this dict -- it is now a required
    # `--lr` CLI argument (see build_formal_sft_config/run_formal_sft/main
    # below and this module's docstring). Keeping it here too, alongside a
    # CLI override, would create a dict-says-3e-4 / CLI-says-2e-5 dual-source
    # state with no way to tell which one a given run actually used; the CLI
    # value is the single source now, and it is written into the manifest so
    # every produced checkpoint self-describes its own lr.
    "epochs": 2,
    # `warmup_ratio` itself isn't a `TrainingArguments`/`SFTConfig` field on
    # this project's pinned `transformers==5.15.0` (same finding
    # `src/posttrain/sft_probe.py`'s `build_sft_config` docstring already made
    # for the probe) -- `build_formal_sft_config` below converts this ratio
    # to an absolute `warmup_steps` count itself, once it knows the total
    # step count (which depends on `--batch-size`, a CLI arg, not known here).
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    # Both of these previously had no explicit setting at all -- meaning they
    # silently inherited HF `TrainingArguments`' own defaults (linear decay,
    # max_grad_norm=1.0) rather than being a deliberate project decision.
    # Both are now explicit: cosine to match pretraining's own schedule shape; grad-clip norm stays
    # numerically 1.0 (also matching pretraining) but is now written down
    # instead of silently inherited.
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
}

# Fixed process count for TRL's `SFTTrainer._prepare_dataset` map calls
# (Tokenizing/Truncating/Building labels/Dropping fully masked examples --
# measured at 12:45 total on the full 460,341-example corpus, on top of the
# ~55-minute counting-loop cost the token-stats cache in data.py now
# removes). A FIXED integer, NOT `os.cpu_count()` or anything else read
# from the machine: the requirement that the three formal runs be exactly
# identical outranks squeezing out the
# last bit of parallelism, and a value that changes with whatever node
# happens to run a given submission would make the corpus-processing
# step (order of tokenized rows before `shuffle`, which happens BEFORE
# this and is unaffected, but pool-scheduling nondeterminism inside
# `datasets.Dataset.map(num_proc=N)` itself is a separate, real risk)
# silently vary between the three formal runs, or between two attempts of
# the SAME run after a preemption-triggered restart on a different node.
# 8 is a conservative, always-available constant on both the development
# machine and the training cluster's GPU nodes (neither is a single-core
# box), enough to meaningfully parallelize the four maps without risking
# oversubscription on a shared node. Actual wall-clock speedup is to be
# measured on real GPU hardware -- not verified on the development machine.
DATASET_NUM_PROC = 8


def _get_last_checkpoint(folder: str) -> str | None:
    """Thin wrapper around `transformers.trainer_utils.get_last_checkpoint`,
    kept lazy-imported (this module's existing style: `trl`/`transformers`
    are optional at module-import time, see the other local imports below)
    but still a MODULE-LEVEL name -- so tests can `monkeypatch.setattr(
    sft_formal_module, "_get_last_checkpoint", ...)` without needing
    `transformers` installed just to import this file.

    Resume support: `run_formal_sft`'s call site below only decides WHETHER
    to resume (does a `checkpoint-*` directory already exist under the
    scratch dir); it does NOT itself constitute proof that a real resume of
    `LoopMoEForCausalLM` + the `_ShiftedLabelsSFTTrainer` subclass through
    HF's checkpoint save/resume machinery produces bit-identical
    continuation. That proof ("run N steps -> kill -> resubmit -> confirm
    training resumes from step N, not step 0") must be done on real GPUs.
    """
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(folder)


def _assert_not_data_parallel_wrapped(model) -> None:
    """See the call site's comment and `src.posttrain.sft_probe._refuse_silent_
    dataparallel`'s docstring for the full failure mode.
    Raises rather than a bare `assert` for the same "must survive `python -O`"
    reason `src.posttrain.sft_probe._resolve_is_main_process` already documents."""
    if isinstance(model, torch.nn.DataParallel):
        raise RuntimeError(
            "trainer wrapped the model in nn.DataParallel despite this script's multi-process "
            "(torchrun) launch expectation -- this hits the same device-mismatch crash described "
            "in src/posttrain/sft_probe.py (_refuse_silent_dataparallel). "
            "Check that RANK/LOCAL_RANK env vars are actually set (i.e. this was launched via "
            "torchrun, not run directly as a plain single process with multiple GPUs visible)."
        )


def build_formal_sft_config(
    *, out_dir: Path, seed: int, max_seq_len: int, batch_size: int, num_examples: int,
    gradient_accumulation_steps: int, save_steps: int, world_size: int, lr: float,
    max_steps: int | None = None,
):
    """Same three hard-coded non-default `SFTConfig` settings as
    `src.posttrain.sft_probe.build_sft_config` (`loss_type="nll"` / `packing=False`
    / `gradient_checkpointing=False`) -- see that function's docstring for the
    full architectural rationale (this project's `LoopMoEForCausalLM` never
    returns `last_hidden_state`, never uses `attention_mask`, does not support
    gradient checkpointing; none of that is dtype- or recipe-dependent, so the
    same three settings apply unchanged here).

    `warmup_steps` is computed from `FORMAL_RECIPE["warmup_ratio"]` here
    (rather than being a fixed number like the probe's `PROBE_RECIPE`) because
    the formal tier's step count depends on `batch_size`, which is a required
    CLI arg -- so the absolute step count
    cannot be baked into a fixed constant the way the probe's `warmup_steps=0`
    could be. `total_steps` uses `math.ceil` (steps-per-epoch rounds UP, not
    down) matching the same "get the boundary arithmetic right, not
    approximately right" rule `src/pretrain/recipe.py`'s own docstring
    insists on for its own step-count math -- an off-by-one here would shift
    every learning-rate value in the schedule, exactly the class of "no error,
    just silently wrong" bug that rule exists to catch.

    `world_size`: `num_examples` is the FULL, unsharded corpus count (every
    `torchrun` process loads the entire `smol-smoltalk` split independently --
    see `src/posttrain/data.py::load_full_dataset`), but the Trainer's actual
    optimizer-step count under DDP is computed over each process's SHARD
    (`DistributedSampler` splits the dataset ~evenly across `world_size`
    processes, and all processes step in lockstep). Dividing by `world_size`
    here makes `total_steps`/`warmup_steps` match what the Trainer will
    ACTUALLY run; omitting this division (an earlier bug) makes
    `total_steps` `world_size`-times too large, so 3% of it becomes 3% *
    `world_size` of the REAL total -- e.g. 24% under 8 processes -- a "no
    error, just a silently different (and unintended) learning-rate schedule
    shape" bug, not a crash. `world_size=1` (single-process, e.g. the probe
    tier's usage pattern, or a 1-process dry-run) reduces to
    the previous, already-verified formula exactly.

    `save_steps`: required, no
    default -- how often to checkpoint depends on real step-timing, which
    must be measured; there is no safe number to guess here (see this
    module's `main()` --save-steps help text). `save_total_limit=2` is NOT a
    parameter (fixed at 2, hard-coded below) -- unbounded checkpoint retention
    would fill the disk on a multi-day run (1.3B bf16 weights + optimizer
    state is ~8-18GB/checkpoint), and 2 is enough to always have one fully-written fallback
    if a kill lands mid-write of the newest one.

    `lr`: required, no default -- see this module's docstring and `FORMAL_RECIPE`'s
    comment for why it moved out of the fixed recipe dict into a required
    CLI-sourced parameter (runs pass 3e-4 or 2e-5; the recipe
    dict is not consulted for this one field anymore, avoiding a two-source
    state where the dict and the actual run could silently disagree).

    `assistant_only_loss=True` (assistant-only masking policy): mirrors `src.posttrain.sft_probe.build_sft_config`'s own
    setting -- see that function's docstring for the full rationale
    (`MINIMAL_CHAT_TEMPLATE`'s `{% generation %}` markers, why TRL fails
    loudly rather than silently reverting to full-sequence loss if they're
    ever removed). Both call sites must carry this identically: it is a
    semantic parameter for the three formal runs, and the requirement that
    they be exactly identical applies to this exactly as it does to
    lr/epochs/warmup.
    """
    from trl import SFTConfig

    # `total_steps` here means OPTIMIZER steps -- gradient accumulation means
    # several micro-batches (each `batch_size` examples) get summed before one
    # optimizer step happens, so the step count the warmup schedule actually
    # operates on divides by `gradient_accumulation_steps` too, not just
    # `batch_size`. Getting this wrong doesn't error -- it just makes the
    # warmup phase silently too long or too short relative to the intended
    # 3% ratio (same "no error, just silently wrong" class of bug this
    # function's docstring already warns about for the ceil() choice).
    #
    # `examples_per_process`: see this function's docstring's `world_size`
    # section -- `num_examples` is unsharded, the Trainer's real step count
    # runs over each process's ~1/world_size share of it.
    examples_per_process = math.ceil(num_examples / world_size)
    micro_batches_per_epoch = math.ceil(examples_per_process / batch_size)
    optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / gradient_accumulation_steps)
    total_steps = optimizer_steps_per_epoch * FORMAL_RECIPE["epochs"]
    warmup_steps = round(FORMAL_RECIPE["warmup_ratio"] * total_steps)

    return SFTConfig(
        output_dir=str(out_dir),
        seed=seed,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=FORMAL_RECIPE["epochs"],
        # `max_steps`: dry-run only -- None (default)
        # preserves the real formal-run behavior (stop after
        # FORMAL_RECIPE["epochs"] full epochs over the corpus). HF
        # `TrainingArguments` treats any positive `max_steps` as an override
        # that stops training early regardless of `num_train_epochs` --
        # exactly what a short dry-run needs (a few/a few hundred steps, not
        # 2 epochs over 392M tokens). Passing `None`/`-1` through
        # unconditionally (rather than omitting the kwarg) keeps this one
        # branch instead of two near-duplicate SFTConfig() call sites.
        max_steps=max_steps if max_steps is not None else -1,
        warmup_steps=warmup_steps,
        weight_decay=FORMAL_RECIPE["weight_decay"],
        lr_scheduler_type=FORMAL_RECIPE["lr_scheduler_type"],
        max_grad_norm=FORMAL_RECIPE["max_grad_norm"],
        report_to=[],
        logging_steps=1,
        # --- save_strategy="no" would mean zero intermediate checkpoints --
        # one external SIGTERM would lose the ENTIRE multi-day run's progress,
        # 100%, regardless of how close to done it was. save_total_limit
        # fixed at 2, see docstring above.
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        # --- max_length must be passed explicitly: when it was once
        # accidentally dropped, TRL's own default (max_length=1024)
        # silently truncated (keep_start) every sequence at 1024 tokens --
        # under assistant-only masking, the supervised (assistant) tokens
        # live at the END of the sequence, so this silently deleted most of
        # the actual training signal on any example longer than 1024 tokens
        # (corpus mean is 851.6 tokens/example, but longer multi-turn
        # examples exist and are exactly the ones this would gut).
        max_length=max_seq_len,
        use_cpu=not torch.cuda.is_available(),
        gradient_checkpointing=False,
        loss_type="nll",
        packing=False,
        # --- assistant-only masking policy, see docstring above ---
        assistant_only_loss=True,
        # --- TRL-map parallelism, see DATASET_NUM_PROC above ---
        dataset_num_proc=DATASET_NUM_PROC,
    )


def run_formal_sft(
    *, checkpoint: Path, out_dir: Path, seed: int, batch_size: int, gradient_accumulation_steps: int,
    dtype: torch.dtype, save_steps: int, lr: float, tokenizer_name: str, max_steps: int | None = None,
    recompute_token_stats: bool = False,
) -> dict[str, Any]:
    """`tokenizer_name`: no default -- required, one of `src.data.tokenizer.TOKENIZER_CLASSES`
    (see `src.posttrain.sft_probe.run_sft`'s docstring for the full rationale, shared verbatim here).

    Full-corpus counterpart to `src.posttrain.sft_probe.run_sft` -- same
    checkpoint-validation / DDP-main-process-gating / manifest-writing shape,
    but `load_full_dataset` (entire corpus) instead of `load_probe_dataset`
    (token-budget prefix), and `FORMAL_RECIPE` instead of `PROBE_RECIPE`.

    `save_steps`: see `build_formal_sft_config`'s docstring -- required, no default.

    `lr`: required, no default -- see `build_formal_sft_config`'s docstring. Also written
    verbatim into the manifest as `learning_rate` (a
    produced checkpoint must self-describe its own lr, since run
    directory names alone are not a reliable record of what actually ran).

    `world_size` is read from the `WORLD_SIZE` env var here, not
    accepted as a parameter -- `torchrun` sets it on every process it
    launches (same convention `src.posttrain.sft_probe._is_main_process` already
    uses for `RANK`); a plain single-process invocation (no `torchrun`) has
    no `WORLD_SIZE` set, and `"1"` is the correct value for that case (no
    sharding), not a fallback that hides a real multi-process launch.

    `recompute_token_stats` (superseding an earlier `compute_
    token_stats`/main-process-only attempt at this problem, which was
    shown not to actually save wall-clock -- see `load_full_dataset`'s
    docstring): default `False` means every invocation, on every rank,
    looks up the committed `posttrain/token_stats_cache.json` for the
    full-corpus token count instead of recomputing it (a cache miss raises,
    it does not silently fall back to the ~55-minute pass). Pass `True`
    (the `--recompute-token-stats` CLI flag) ONLY to deliberately
    recompute and refresh the cache file after a real data/tokenizer/
    template change -- never for a routine formal-tier submission.
    """
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not is_complete(checkpoint):
        raise ValueError(
            f"{checkpoint} has no valid manifest.json -- refusing to fine-tune an "
            "incomplete/unverified checkpoint (same discipline as "
            "src/eval/run_eval.py's `_read_manifest` and src/posttrain/sft_probe.py's run_sft)."
        )

    # Read BEFORE training, not after (same reasoning as
    # src/posttrain/sft_probe.py's run_sft):
    # `code_revision()` raises if git is unusable, e.g. a transient lock from
    # another process's concurrent `git add`/`commit` in a shared working
    # tree. This is the FORMAL (production-tier, multi-epoch, multi-GPU)
    # launcher -- discarding a completed run's result to a post-hoc git
    # failure would be even more costly here than in the probe tier.
    rev = code_revision()

    torch.manual_seed(seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    tokenizer_repo = load_named_tokenizer(tokenizer_name).repo_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
    tokenizer.chat_template = MINIMAL_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, dtype=dtype)
    # Read off the checkpoint's own config, not a CLI flag -- same rationale
    # as the probe: it is a property of the checkpoint, not an independent choice.
    max_seq_len = model.config.context_length

    examples, data_stats = load_full_dataset(
        seed=seed, tokenizer=tokenizer, recompute_token_stats=recompute_token_stats
    )
    train_dataset = Dataset.from_list(examples)

    scratch_dir = out_dir / "_trainer_scratch"
    sft_config = build_formal_sft_config(
        out_dir=scratch_dir, seed=seed, max_seq_len=max_seq_len,
        batch_size=batch_size, num_examples=len(examples),
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps, world_size=world_size, lr=lr, max_steps=max_steps,
    )
    # ShiftedLabelsSFTTrainer, not plain SFTTrainer -- the same single fix
    # point sft_probe.py uses (see
    # src.posttrain.sft_probe._shift_labels_for_next_token_loss's docstring).
    trainer_cls = build_shifted_labels_sft_trainer_class()
    trainer = trainer_cls(model=model, args=sft_config, train_dataset=train_dataset, processing_class=tokenizer)
    # Defensive check, not expected to ever fire: this script is meant to be
    # launched via torchrun (one process per GPU), which HF Trainer recognizes
    # and does NOT wrap in nn.DataParallel. `sft_probe.py` (single-process by
    # design) hit the DP-wrapping path and crashed on this project's
    # hand-indexed embedding -- this assertion exists so that IF this script is ever
    # accidentally launched as a single process with multiple GPUs visible
    # (e.g. someone runs it directly instead of through torchrun), it fails
    # loudly here, at the one point that actually matters, rather than
    # silently training on DP-corrupted gradients or crashing deep inside
    # `trainer.train()` with a less legible stack trace.
    _assert_not_data_parallel_wrapped(trainer.model)
    # Resume from the last checkpoint under
    # `scratch_dir` if one exists (a prior invocation against this SAME
    # `out_dir` got killed mid-run and this is a resubmission) -- `None` when
    # no checkpoint dir exists yet (the scratch dir itself may not even exist
    # on a genuinely first-ever submission, and `get_last_checkpoint` errors
    # on a missing directory rather than returning `None`, hence the
    # existence check first). This is the CODE-LEVEL half of the fix; the
    # real-hardware "kill mid-run -> resubmit -> confirm it actually resumed"
    # proof must be done on GPUs (see `_get_last_checkpoint`'s docstring).
    resume_checkpoint = _get_last_checkpoint(str(scratch_dir)) if scratch_dir.exists() else None
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    manifest_fields = dict(
        step=0,  # formal SFT is single-pass over N epochs, not step-numbered like pretraining
        tokens_consumed=data_stats["num_tokens_selected"],
        git_commit=rev["commit"],
        code_dirty=rev["dirty"],
        code_diff_sha256=rev["diff_sha256"],
        tokenizer_name=tokenizer_name,
        # The actually-used lr, written verbatim so a saved checkpoint
        # self-describes it (a run's directory name is not, by itself, a
        # reliable record of what lr actually ran).
        learning_rate=lr,
        # Make the manifest self-describing
        # for these two fields instead of leaving them only derivable from
        # `model_config["context_length"]` (architectural limit, not
        # necessarily what THIS run used) or absent entirely (`step` is
        # always 0 for formal SFT -- see the comment on that field below --
        # so epoch progress previously had nowhere to be recorded).
        max_seq_len=max_seq_len,
        epochs_completed=trainer.state.epoch,
        compute_dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
        autocast={"enabled": False},
        torch_version=torch.__version__,
        transformers_version=__import__("transformers").__version__,
        model_config=model.config.to_dict(),
        data_position={"formal_sft_source": data_stats, "source_checkpoint": str(checkpoint)},
        perturbation_probe={"runs_offline": True},
        storage_dtype="bfloat16",
        rng_state_saved=False,
    )
    # DDP main-process gating: shared with the probe (`src.posttrain.sft_probe.
    # _resolve_is_main_process`) -- see that function's docstring for why this
    # raises instead of asserting, and why it cross-checks against
    # `trainer.is_world_process_zero()` rather than trusting the env var alone.
    is_main_process = _resolve_is_main_process(trainer)
    if is_main_process:
        save_trajectory(model, out_dir, manifest_fields=manifest_fields)

    # A real formal-tier run trains for 2 epochs over ~392M tokens -- by the
    # time it ends, this process's CUDA caching allocator has grown to
    # occupy nearly the whole GPU (measured: 139.03 GiB held on a
    # 139.80 GiB card after just 300 steps). `main()` shells out to
    # `run_lm_eval` in a SEPARATE process on the SAME GPU right after this
    # function returns -- without releasing first, that subprocess fails to
    # allocate its own copy of the model (`253.50 MiB is free`), and the
    # run ends looking like a training failure when training actually
    # succeeded. See `src.posttrain.sft_probe._release_cuda_memory_before_eval`'s
    # docstring for why the `del` must happen here (this function's own
    # scope holds the only Python references to `trainer`/`model`) and why
    # it's guarded for a CUDA-less environment.
    del trainer, model
    _release_cuda_memory_before_eval()

    return {
        "train_loss": train_result.training_loss,
        "data_stats": data_stats,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "warmup_steps": sft_config.warmup_steps,
        "is_main_process": is_main_process,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Input pretraining trajectory checkpoint (HF format)")
    ap.add_argument("--out", required=True, help="Output directory for the -instruct checkpoint + eval JSONs")
    ap.add_argument("--seed", type=int, required=True, help="Data shuffle + torch seed (reproducibility)")
    ap.add_argument(
        "--tokenizer-name", dest="tokenizer_name", choices=sorted(TOKENIZER_CLASSES), required=True,
        help="Which tokenizer this checkpoint expects -- see src.data.tokenizer.TOKENIZER_CLASSES. "
             "No default: same rationale as src/posttrain/sft_probe.py's --tokenizer-name.",
    )
    ap.add_argument(
        "--dtype", choices=("float32", "bfloat16"), required=True,
        help="No default -- this is the formal tier, bf16 is the intended precision. "
             "float32 is only for CPU debugging/dry-runs; must be explicit "
             "either way, unlike src/posttrain/sft_probe.py's --dtype (which defaults to float32 "
             "to preserve that script's already-in-flight behavior).",
    )
    ap.add_argument(
        "--batch-size", type=int, required=True,
        help="Per-device train batch size. No default -- set it from a multi-GPU dry-run's "
             "memory measurement; do not guess a number here.",
    )
    ap.add_argument(
        "--gradient-accumulation-steps", type=int, required=True,
        help="Micro-batches accumulated per optimizer step. No default -- like --batch-size, "
             "set it from a dry-run (target effective batch size / feasible per-device batch "
             "size).",
    )
    ap.add_argument(
        "--lr", type=float, required=True,
        help="Learning rate. No default: a hardcoded value would let a run with a different "
             "intended lr silently train with the old one. The value is also written into the "
             "checkpoint's manifest (learning_rate field) so the produced weights self-describe "
             "which lr actually ran, not just their directory names.",
    )
    ap.add_argument(
        "--save-steps", type=int, required=True,
        help="Checkpoint every N optimizer steps (a formal run with save_strategy='no' loses "
             "100%% of its progress to a single external kill, regardless of how close to done "
             "it was). No default -- the right number depends on real step timing, measured in "
             "a dry-run; do not guess a number here. "
             "save_total_limit is fixed at 2 (not a CLI flag), see build_formal_sft_config's docstring.",
    )
    ap.add_argument(
        "--eval-python", required=True,
        help="Path to the `looptrain-eval` env's python interpreter (same rationale as "
             "src/posttrain/sft_probe.py's --eval-python: lm_eval is not a dependency of this env).",
    )
    ap.add_argument(
        "--eval-task-set", dest="eval_task_set", choices=("probe", "full"), required=True,
        help="Which lm-eval task set to run post-SFT: 'probe' = fast 3-task subset "
             "(arc_easy/piqa/hellaswag), 'full' = complete 11-task LT2 suite (needed for an "
             "architecture comparison with statistical power). No default -- replaces the "
             "retired --eval-include-optional/--no-eval-include-optional flag (that boolean "
             "never actually reached the full 11-task suite despite looking like it should, see "
             "src/posttrain/eval_subset.py's docstring).",
    )
    ap.add_argument("--eval-limit", dest="eval_limit", type=float, default=None,
                     help="Per-task lm-eval example cap (unset = full eval for the chosen task set)")
    ap.add_argument(
        "--max-steps", dest="max_steps", type=int, default=None,
        help="Dry-run ONLY -- stop after this many optimizer steps regardless of "
             "FORMAL_RECIPE's epoch count. Unset (default) = a real formal run (full "
             "FORMAL_RECIPE['epochs'] over the whole corpus). Never set this for an actual "
             "-instruct production run.",
    )
    ap.add_argument(
        "--recompute-token-stats", dest="recompute_token_stats", action="store_true",
        help="Recompute the full-corpus token count and refresh "
             "posttrain/token_stats_cache.json, instead of reading "
             "the committed cache (the default). This pays the ~55-minute per-row tokenize pass -- "
             "only pass this deliberately, after a real dataset/tokenizer/chat-template change, to "
             "re-seed the cache (then commit the updated cache file). Never set this for a routine "
             "-instruct production run.",
    )
    args = ap.parse_args(argv)

    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[sft_formal] checkpoint={checkpoint} out={out_dir} seed={args.seed} "
          f"batch_size={args.batch_size} gradient_accumulation_steps={args.gradient_accumulation_steps} "
          f"dtype={args.dtype} lr={args.lr} save_steps={args.save_steps} max_steps={args.max_steps} "
          f"tokenizer_name={args.tokenizer_name} recompute_token_stats={args.recompute_token_stats}")
    sft_summary = run_formal_sft(
        checkpoint=checkpoint, out_dir=out_dir, seed=args.seed, batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, dtype=dtype,
        save_steps=args.save_steps, lr=args.lr, tokenizer_name=args.tokenizer_name,
        max_steps=args.max_steps, recompute_token_stats=args.recompute_token_stats,
    )
    print(f"[sft_formal] SFT done: {json.dumps(sft_summary, indent=2)}")

    # DDP: only the process that actually wrote the checkpoint runs eval /
    # instruction probes against it -- same rationale as src/posttrain/sft_probe.py's main().
    if not sft_summary["is_main_process"]:
        return

    print(f"[sft_formal] running lm-eval ({args.eval_task_set} task set)...")
    run_lm_eval(
        out_dir=out_dir, eval_python=args.eval_python,
        task_set=args.eval_task_set, eval_limit=args.eval_limit, tokenizer_name=args.tokenizer_name,
    )

    print("[sft_formal] running instruction-following probes (cold-loaded checkpoint)...")
    instruction_results = run_instruction_probes(out_dir=out_dir, tokenizer_name=args.tokenizer_name)
    (out_dir / INSTRUCTION_PROBE_RESULTS_NAME).write_text(json.dumps(instruction_results, indent=2))
    print(f"[sft_formal] instruction probes: {instruction_results['num_passed']}/{instruction_results['num_total']} passed")

    print(f"[sft_formal] done. Artifacts in {out_dir}: pytorch_model.bin, manifest.json, "
          f"eval_results.json, {INSTRUCTION_PROBE_RESULTS_NAME}")


if __name__ == "__main__":
    main()

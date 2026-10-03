"""Fine-tuning probe: `checkpoint -> quick SFT (SmolTalk smol-smoltalk) -> eval`.

Usage:

    python -m src.posttrain.sft_probe \\
        --checkpoint /path/to/trajectory/step_012345 \\
        --out /path/to/posttrain_probe/step_012345_sft \\
        --token-budget 20000000 \\
        --seed 0 \\
        --eval-python $CONDA_PREFIX/../looptrain-eval/bin/python \\
        --eval-task-set probe

The SFT recipe itself (learning rate, batch size, warmup, weight decay) is
NOT a CLI argument -- see `PROBE_RECIPE` below and "The probe recipe ... is
fixed, not a CLI knob".

## What this is (and is not)

A **probe**, not a real instruct-tuning run: given a mid-training checkpoint,
do a cheap SFT pass and a small eval subset, purely to answer "does this
architecture/checkpoint show any instruction-following life" (Snell et al.,
COLM 2024; Du, arXiv 2601.13244 -- the methodology's basis). It is not tuned
for output quality and its token budget is explicitly small (tens of millions
of tokens, not the billions a real SFT run would use).

## Tool choice: TRL `SFTTrainer`, not LLaMA-Factory, not a hand-rolled Trainer

Reasoning: `LoopMoEForCausalLM.forward()` already
folds the MoE aux losses (`lb_loss`, `z_loss`) into the `.loss` it returns when
given `labels` -- any trainer that just calls `model(**batch).loss` and
backprops gets correct gradients with zero framework-side MoE awareness. That
makes TRL's `SFTTrainer` (a thin wrapper around HF `Trainer`, handling chat
templates / prompt-completion masking) the right amount of machinery: enough to
not hand-roll tokenization/masking/the training loop, not so much that it drags
in a whole separate config/model-registry system (LLaMA-Factory) whose many
"known architecture" assumptions are a real compatibility risk for our
non-standard model (no `attention_mask` support, no `last_hidden_state` output,
no gradient checkpointing -- see below).

## Three non-default settings this script hard-codes, and why

Each was found by *running* TRL's `SFTTrainer` against a real (toy)
`LoopMoEForCausalLM` checkpoint, not by reading docs:

1. **`gradient_checkpointing=False`** -- `LoopMoEForCausalLM` does not implement
   `_set_gradient_checkpointing` (`supports_gradient_checkpointing=False`,
   inherited default from `transformers.PreTrainedModel`). `SFTConfig`'s own
   default is `gradient_checkpointing=True`; leaving that default raises
   `ValueError: LoopMoEForCausalLM does not support gradient checkpointing`
   before training starts. A probe's models are small and its budget is
   token-capped anyway, so the memory saving gradient checkpointing would have
   bought is not needed here.

2. **`loss_type="nll"`** -- THE ONE THAT MATTERS. TRL's newer default,
   `loss_type="chunked_nll"`, patches the model's `forward` to skip the
   `lm_head` matmul on masked-out tokens and read `outputs.last_hidden_state`
   to recompute cross-entropy itself, chunked, OUTSIDE the model. Our model's
   `forward()` never returns `last_hidden_state` (it computes CE + the
   `lb_loss_factor`/`lz_loss_factor`-weighted aux losses internally and returns
   only the combined `.loss`) -- so the chunked path raises
   `AttributeError: 'CausalLMOutputWithPast' object has no attribute
   'last_hidden_state'` on the first training step. `loss_type="nll"` reverts to
   the plain path (`trainer.compute_loss` reads `model(**inputs).loss`
   directly), which is the ONLY path that actually trains on the aux-loss-
   weighted objective the model was designed around. If a future TRL release
   changes the chunked path to fail more gracefully (e.g. silently falling back
   to a hidden-states-free CE that drops the aux loss instead of raising), that
   would be a *silent* failure instead of a loud error. Two properties guard
   against it: `nll`'s loss must match the model's own aux-loss-inclusive
   `.loss`, and `chunked_nll` must still raise `AttributeError` -- if TRL ever
   changes this contract, that check fails instead of the training objective
   silently degrading.

3. **`packing=False`** (never made a CLI flag -- always off). `LoopMoEForCausalLM.
   forward()` accepts an `attention_mask` argument but never uses it (verified
   by reading `src/model/modeling_loop_lm.py`: the mask never reaches
   `self.model(...)`). TRL's sequence packing (`packing=True`) concatenates
   multiple examples into one context and relies on `attention_mask`/position
   resets at the boundary to stop attention from crossing between them --
   without mask support, packed examples would genuinely attend into each
   other's tokens (real cross-contamination, not a theoretical risk). Right-
   padding single examples (this script's approach) is safe under a pure-causal
   model: earlier real tokens never see later padding, and padded positions'
   loss is masked via `labels=-100`.

## The probe recipe (lr / batch size / warmup / weight decay) is fixed, not a CLI knob

A probe is a **measurement instrument**,
not a tuning exercise -- its whole value is that the SAME recipe was applied to
every checkpoint it ever runs against, so results are comparable across
checkpoints and architectures. A `--lr` flag that's easy to nudge on the
command line defeats that: a probe run with a quietly different learning rate
produces a number that LOOKS comparable to every other probe run but isn't.
So `PROBE_RECIPE` below is a fixed, named, commented constant block, not
CLI-exposed arguments with defaults -- changing the recipe means editing this
file and going through code review, the same bar as any other semantic change,
not a flag typed once and forgotten. `--token-budget`/`--seed`/`--checkpoint`
stay CLI-required because they vary *by design* per invocation (a different
checkpoint, a different budget for a bigger model) -- that's a different kind
of parameter than the recipe itself.

## Checkpoint output format: `save_trajectory`, not `trainer.save_model()`

TRL/HF `Trainer.save_model()` writes `config.json` + weights only -- no
`manifest.json`, no bundled modeling source, no bf16 cast. `pretrain/eval/
run_eval.py` refuses to evaluate a directory without a valid manifest
(`src.pretrain.checkpointing.store.is_complete`), and the diagnostics pipeline more
broadly expects the trajectory-checkpoint shape everywhere else in this
project. So this script trains with `SFTTrainer`, then hands the trained
model's `state_dict()` to `src.pretrain.checkpointing.store.save_trajectory`
directly -- the exact function `src/pretrain/train_toy.py` and the production
training loop both use -- producing an output directory `run_eval.py` can load
unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from src.pretrain.checkpointing.store import is_complete, save_trajectory
from src.data.tokenizer import TOKENIZER_CLASSES, load_named_tokenizer
from src.metrics.provenance import code_revision
from src.posttrain.chat_template import INSTRUCTION_PROBES, MINIMAL_CHAT_TEMPLATE, run_instruction_probe
from src.posttrain.data import load_probe_dataset
from src.posttrain.eval_subset import resolve_eval_tasks

# Initial suggested token-budget values. NOT wired in as a CLI default --
# `--token-budget` is a required argument (semantic parameters have no
# default); this table exists only so a human invoking this script has a
# starting point, and is explicitly flagged as uncalibrated (to be
# calibrated once the first real probe run's actual
# wall-clock/tokens-per-second is measured).
SUGGESTED_TOKEN_BUDGETS_UNCALIBRATED = {
    "seedvar_d704": 20_000_000,
    "0p6b": 20_000_000,
    "1p3b": 40_000_000,
}

INSTRUCTION_PROBE_RESULTS_NAME = "instruction_probe_results.json"

# The fixed probe SFT recipe (see this module's "The probe recipe ... is
# fixed, not a CLI knob" section).
# Changing any of these is a code change + review, not a flag.
PROBE_RECIPE = {
    # Matches Ouro-1.4B-Thinking's SFT learning rate (ByteDance, arXiv
    # 2510.25741 -- the one competitor with a documented real SFT recipe) and
    # is also the common-sense
    # default across small-model instruction-SFT recipes generally (e.g.
    # SmolLM2/Llama-family quick-SFT configs).
    "lr": 2e-5,
    # Small-model SFT common default. The probe's token budget is the actual
    # cap on how much is trained on (src/posttrain/data.py) -- batch size here only
    # affects step granularity/wall-clock, not what gets trained on.
    "batch_size": 4,
    # 0: the probe is a single-epoch pass over a token-capped (tens-of-millions)
    # dataset -- warmup's benefit is negligible at this scale, and 0 avoids
    # tuning a schedule for a run this short. (Also the setting that surfaced
    # `transformers==5.15.0` dropped `TrainingArguments.warmup_ratio` in favor
    # of `warmup_steps`: passing `warmup_ratio` raises a TypeError.)
    "warmup_steps": 0,
    # No regularization for a short probe pass -- matches most small-model
    # quick-SFT recipes, and there's no overfitting concern worth guarding
    # against at this token budget.
    "weight_decay": 0.0,
}


def build_sft_config(*, out_dir: Path, seed: int, max_seq_len: int):
    """The three hard-coded, non-default settings this module's docstring
    explains (gradient_checkpointing / loss_type / packing) live here, in one
    place, so tests can import and exercise the exact config this script uses --
    not a re-typed copy that could drift.

    `lr`/`batch_size`/`warmup_steps`/`weight_decay` are read from
    `PROBE_RECIPE`, not accepted as parameters here -- the recipe is fixed
    (see this module's "The probe recipe ... is fixed, not a CLI knob"), so
    there is deliberately no call-site way to override it either.

    `warmup_steps`, not `warmup_ratio`: verified against this
    project's pinned `transformers==5.15.0` (`requirements-posttrain.txt`) --
    `TrainingArguments` (which `SFTConfig` subclasses) no longer has a
    `warmup_ratio` field in this version (`TypeError: SFTConfig.__init__() got
    an unexpected keyword argument 'warmup_ratio'`). `warmup_steps` is the
    field that still exists.

    `assistant_only_loss=True` (assistant-only masking policy, adopted
    before any of the three formal runs started): loss is computed only on assistant-turn tokens, user/system
    turns masked to -100. Requires `MINIMAL_CHAT_TEMPLATE` (`posttrain/
    chat_template.py`) to carry `{% generation %}` markers around each
    assistant turn -- without them TRL raises `ValueError` at dataset-
    preparation time (`SFTTrainer._prepare_dataset`, not a silent no-op), so
    a drift here fails loudly rather than quietly reverting to full-sequence
    loss. This is the ONE call site both the probe (`run_sft`) and the formal
    tier (`sft_formal.py::build_formal_sft_config`, which mirrors this same
    setting) key off -- earlier probe numbers produced BEFORE this switch
    (full-sequence loss) are not comparable to anything trained after it.
    """
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(out_dir),
        seed=seed,
        learning_rate=PROBE_RECIPE["lr"],
        per_device_train_batch_size=PROBE_RECIPE["batch_size"],
        num_train_epochs=1,
        max_length=max_seq_len,
        warmup_steps=PROBE_RECIPE["warmup_steps"],
        weight_decay=PROBE_RECIPE["weight_decay"],
        report_to=[],
        logging_steps=1,
        save_strategy="no",
        use_cpu=not torch.cuda.is_available(),
        # --- the three settings the module docstring explains at length ---
        gradient_checkpointing=False,
        loss_type="nll",
        packing=False,
        # --- assistant-only masking policy, see docstring above ---
        assistant_only_loss=True,
    )


def _is_main_process() -> bool:
    """True unless launched under `torchrun`/`accelerate` with a nonzero rank.

    `RANK` is the env var `torchrun` sets on every process (0-indexed); it is
    simply absent for a plain single-process invocation (the probe's current
    usage), which is why `os.environ.get("RANK", "0")` defaulting to `"0"`
    preserves that case exactly -- a single-process run is always "the main
    process". Used to gate the once-per-run side effects (writing the output
    checkpoint, shelling out to lm-eval, running the instruction probes) so a
    future multi-GPU (DDP) launch does not have every rank redundantly (and
    racily) repeat them -- see this module's "DDP" note in `run_sft`.
    """
    return os.environ.get("RANK", "0") == "0"


def _resolve_is_main_process(trainer) -> bool:
    """The single source of truth `run_sft` uses for "should THIS process do
    the once-per-run side effects" (write the checkpoint; later, in `main()`,
    run eval / the instruction probes).

    Cross-checks `_is_main_process()` (env-var reading) against
    `trainer.is_world_process_zero()` (the property HF Trainer itself uses
    for this same purpose internally, so it is the more authoritative of the
    two once a trainer exists) and returns the trainer's answer -- but only
    after confirming they agree. A plain `assert` was deliberately NOT used
    here: `python -O` strips `assert`
    statements, which would silently turn a real disagreement (e.g. a
    launcher that sets `RANK` differently than `torchrun` does) into "guess
    which process writes the checkpoint" instead of failing loudly. Raises
    `RuntimeError` (not an `AssertionError`, which is what -O strips) so this
    check survives regardless of how the interpreter is invoked.
    """
    is_main = trainer.is_world_process_zero()
    if _is_main_process() != is_main:
        raise RuntimeError(
            "RANK env var disagrees with trainer.is_world_process_zero() -- "
            "main-process detection is unreliable under this launcher, refusing "
            "to guess which process should write the checkpoint."
        )
    return is_main


def _refuse_silent_dataparallel() -> None:
    """Guard against a known failure: on a machine with more than one
    GPU visible, HF `Trainer` silently wraps the model in the legacy
    `nn.DataParallel` whenever it's launched as a single process (no
    `torchrun`/`accelerate launch`, i.e. no `RANK` env var) and sees
    `n_gpu > 1`. DP replicates the module across devices and scatters inputs,
    which crashes this project's hand-indexed embedding
    (`self.weight[token_ids]` in `src/model/modeling_loop_lm.py`) with
    a device-mismatch `RuntimeError` the moment a replica's `token_ids` lands
    on a different device than its copy of `weight`. A controlled
    comparison (same code, same env, only visible-GPU-count varied) confirmed
    this precisely: 8 GPUs visible -> 2 tests fail; `CUDA_VISIBLE_DEVICES=0`
    -> both pass.

    This script (`sft_probe.py`) is a single-process entry point by design
    (the probe's whole point is a cheap single-GPU measurement) -- it has no
    business ever touching more than one GPU, so rather than silently
    tolerating whatever DP does or doesn't break, it refuses outright the
    moment it detects the exact precondition for DP-wrapping (fail loudly
    rather than silently; the alternative of patching DP to tolerate the
    custom embedding was rejected because DP is a deprecated path not worth
    investing in). Genuine multi-GPU training goes through
    `src/posttrain/sft_formal.py`'s `torchrun`-launched, one-process-per-GPU path
    instead, which this check does not fire under (real multi-process
    launches set `RANK`).
    """
    if torch.cuda.device_count() > 1 and os.environ.get("RANK") is None:
        raise RuntimeError(
            f"{torch.cuda.device_count()} GPUs are visible to this single (non-torchrun) process -- "
            "refusing to proceed. HF Trainer would silently wrap the model in nn.DataParallel here, "
            "which crashes this project's hand-indexed embedding lookup with a device-mismatch "
            "error. This probe script is "
            "single-GPU by design -- restrict visibility before running, e.g. "
            "`CUDA_VISIBLE_DEVICES=0 python -m src.posttrain.sft_probe ...`. For genuine multi-GPU "
            "training, use src/posttrain/sft_formal.py's torchrun-launched path instead."
        )


def _shift_labels_for_next_token_loss(labels: torch.Tensor) -> torch.Tensor:
    """Fixes a label-convention mismatch that otherwise collapses the probe
    model into a copy-the-current-token degenerate solution.

    `LoopMoEForCausalLM.forward()` does NOT shift internally
    (`src/model/modeling_loop_lm.py:694` calls `F.cross_entropy` on
    `labels` exactly as given) -- by explicit, documented design: "the
    dataloader supplies pre-shifted labels during training, so the shift is
    explicit here" (`src/model/loop_trace.py:320`). The pretraining path
    honors this contract (`src/data/torchtitan_adapter.py`:
    `input_ids=tokens[:-1]`, `labels=tokens[1:]`, shifted BEFORE the model
    ever sees them) -- **that path is untouched by this fix** (only the
    posttrain call site is fixed, never the training path, so completed
    pretraining runs' loss numbers stay comparable).

    TRL's `SFTTrainer` has no way to know about this project's non-standard
    contract: it builds `labels` under the ordinary HuggingFace convention
    (`labels[t] == input_ids[t]`, pad positions set to -100), which is the
    right convention for a model that shifts internally -- exactly the
    opposite of what this project's model needs. Fed straight through,
    `src/posttrain/sft_probe.py`'s `loss_type="nll"` path (`model(**inputs).loss`)
    ends up optimizing `CE(logits[t], input_ids[t])`: a trivial identity-copy
    target the model can already see at position `t` under causal attention.
    The observed symptom is the smoking gun: step 1 loss
    of 11.43 (≈ln(vocab_size), the *pretrained* checkpoint correctly assigning
    near-zero probability to "copy yourself") collapsing within 15 steps to
    `mean_token_accuracy≈0` on the REAL next-token metric while `.loss`
    itself fell to ≈0 -- two curves that can only both be true if they are
    scoring two different targets, which is exactly this bug.

    Converts TRL's unshifted `labels` into the shifted convention this
    project's model actually expects: `shifted[t] = labels[t+1]` for all but
    the last position, and `shifted[..., -1] = -100` (there is no "next
    token" after the last position, so it must not contribute to the loss --
    mirrors `F.cross_entropy`'s own `ignore_index=-100` default, the same
    convention TRL already uses for padding).
    """
    shifted = labels.new_full(labels.shape, -100)
    shifted[..., :-1] = labels[..., 1:]
    return shifted


def build_shifted_labels_sft_trainer_class():
    """Returns `_ShiftedLabelsSFTTrainer`, a `trl.SFTTrainer` subclass with
    TWO changes from stock `SFTTrainer` -- both confirmed necessary by
    reading the pinned TRL 1.10.0 / transformers 5.15.0 source:

    1. `compute_loss` sets BOTH `inputs["labels"]` and `inputs["shift_labels"]`
       to the shifted tensor (`_shift_labels_for_next_token_loss`), not just
       `labels`. TRL's own metric computation (`sft_trainer.py:1711`,
       `:1768-1774`) branches on whether `"shift_labels"` is present: if it
       is, TRL trusts it as ALREADY shifted and compares it against the raw
       (unsliced) logits; if it is absent, TRL assumes `labels` is UNSHIFTED
       and shifts it itself before scoring. Setting only `labels` makes
       TRL's metric path re-shift an already-shifted tensor -- "off by two"
       -- which is why `mean_token_accuracy` read ~0.015 (near-random) even
       when the loss itself was correct (confirmed against TRL's own source,
       not a guess).
    2. `__init__` forces `self.model_accepts_loss_kwargs = False`. HF
       `Trainer.__init__` reflects on the model's `forward` signature to
       decide this flag; `LoopMoEForCausalLM.forward` has a trailing
       `**kwargs`, so `Trainer` sets it `True` and then (`trainer.py:1952`,
       `:2040-2046`) skips dividing by `gradient_accumulation_steps` AND
       multiplies the loss by `num_processes` under
       `average_tokens_across_devices` (default `True`) -- because THAT path
       assumes the model itself consumes `num_items_in_batch` to normalize
       its own loss (this model's `forward` never reads it, always plain
       `reduction="mean"`). Left unfixed, a multi-GPU + gradient-accumulation
       formal run's loss and backpropagated gradient would be inflated by
       roughly `num_processes × grad_accum_steps`, `max_grad_norm=1.0` would
       clip almost every step, and `training_loss` would stop being
       comparable to pretraining's loss scale. A single-process probe run
       is NOT affected (single process, accum=1, both factors are 1) -- this
       only bites the formal tier's real multi-GPU path (`sft_formal.py`).
       TRL's own DPO/Reward trainers force this same flag off for the same
       reason (confirmed by reading their source too, not by analogy alone).

    Also hard-`raise`s (not silently proceeds) on two configuration-drift
    conditions that would otherwise silently invalidate the shift logic
    above: `packing=True` (packed sequences need per-document boundary
    handling this shift was never designed or tested for) and `"shift_labels"`
    already present in `inputs` before this code runs (would mean some other
    code path already built a differently-conventioned tensor under the same
    key, and blindly overwriting it would hide that surprise rather than
    surface it).

    Deliberately a function (not a module-level class) because `SFTTrainer`
    itself is an optional/deferred import (`trl` is not installed in every
    environment that imports this module, same reason every other TRL import
    in this file is local to a function) -- subclassing it at module import
    time would force the import unconditionally.

    This is the ONE fix point both `src/posttrain/sft_probe.py` and
    `src/posttrain/sft_formal.py` share (both paths go through the same fix
    point; no second convention) -- `sft_formal.py` imports and uses this exact function/
    class rather than re-deriving its own shift logic. **Both `run_sft()` and
    `run_formal_sft()` MUST construct their trainer via this function's return
    value, never via a directly-imported `trl.SFTTrainer`** -- nothing
    previously enforced this: reverting either call site back to plain
    `SFTTrainer` would silently drop the label shift.

    Deliberately does NOT itself set `completion_only_loss`/`assistant_only_
    loss` -- that lives in `build_sft_config()`/`build_formal_sft_config()`
    (assistant-only masking policy), a separate policy decision from this shift fix.
    The two DO compose correctly: TRL builds `labels` (unshifted, HF
    convention) with -100 already on every non-assistant position BEFORE
    this trainer ever sees them, so `_shift_labels_for_next_token_loss`
    simply carries that masking forward by one position along with the
    correct next-token target (verified by checking the composed shifted +
    masked labels directly, rather than assuming the two features combine
    safely).
    """
    from trl import SFTTrainer

    class _ShiftedLabelsSFTTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # See this factory function's docstring, point 2. Must be set
            # AFTER super().__init__() -- that's what computes/sets it in the
            # first place via reflection on the model's forward signature.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, *args, **kwargs):
            if "labels" in inputs and inputs["labels"] is not None:
                if getattr(self.args, "packing", False):
                    raise RuntimeError(
                        "_ShiftedLabelsSFTTrainer does not support packing=True -- the shift "
                        "logic was never designed or tested for packed (multi-document-per-row) "
                        "sequences, where a naive shift would let position t's target leak across "
                        "a document boundary into the next document."
                    )
                if "shift_labels" in inputs:
                    raise RuntimeError(
                        "inputs already contains 'shift_labels' before this trainer set it -- "
                        "some other code path built a labels/shift_labels pair under an assumption "
                        "this trainer doesn't know about; refusing to silently overwrite it."
                    )
                shifted = _shift_labels_for_next_token_loss(inputs["labels"])
                # Both keys: see this factory function's docstring, point 1 --
                # `shift_labels` is what makes TRL's OWN metric computation
                # (mean_token_accuracy etc.) take the already-shifted branch
                # instead of shifting `labels` a second time.
                inputs = {**inputs, "labels": shifted, "shift_labels": shifted}
            return super().compute_loss(model, inputs, *args, **kwargs)

    return _ShiftedLabelsSFTTrainer


def run_sft(
    *, checkpoint: Path, out_dir: Path, token_budget: int, seed: int, tokenizer_name: str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """`tokenizer_name`: no default -- required, one of `src.data.tokenizer.TOKENIZER_CLASSES`
    (which tokenizer a run uses changes vocab_size/bos/eos and is a property of which checkpoint family is being
    fine-tuned, not something to silently infer). Resolved via `load_named_tokenizer`, which
    also hard-asserts the loaded tokenizer's vocab/bos/eos match that name's expected values
    before this function ever sees it.

    `dtype`: the model's own parameter dtype (pure storage precision, matching
    how this project's pretraining also stores parameters -- NOT
    `TrainingArguments.bf16`/`fp16` autocast-style mixed precision, which this
    function never sets). Defaults to `torch.float32`, the probe's existing,
    validated behavior -- unchanged for every current caller. The formal
    instruction-tuning tier is expected to pass `torch.bfloat16` (matching
    pretraining's storage precision); this module's three hardcoded
    `SFTConfig` settings (`build_sft_config`'s docstring) hold under bf16 as
    well as float32.

    DDP: this function has no `torch.distributed`-specific code (TRL's
    `SFTTrainer`/HF `Trainer` auto-detects and wraps in `DistributedDataParallel`
    when launched under `torchrun`, needing no code-level opt-in here). The one
    correctness requirement multi-process DOES add -- "only one process may
    write the output checkpoint" -- is handled below via `_is_main_process()`
    (checked against `trainer.is_world_process_zero()` for agreement, since
    that is the property HF Trainer itself uses and is the more authoritative
    of the two once the trainer exists). The DDP-wrapping/backward path has
    been checked structurally on CPU only; a real multi-process torchrun
    launch needs real GPUs and is a separate step (a small-scale multi-GPU +
    bf16 dry-run).
    """
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _refuse_silent_dataparallel()

    if not is_complete(checkpoint):
        raise ValueError(
            f"{checkpoint} has no valid manifest.json -- refusing to fine-tune an "
            "incomplete/unverified checkpoint (same discipline as "
            "src/eval/run_eval.py's `_read_manifest`)."
        )

    # Read BEFORE training, not after:
    # `code_revision()` raises if git is unusable, e.g. a transient lock from
    # another process's concurrent `git add`/`commit` in a shared working
    # tree. Calling it only after `trainer.train()` returns would mean that
    # failure -- entirely unrelated to whether training itself succeeded --
    # discards a completed training run's result instead of failing fast for
    # near-zero cost before any GPU time is spent.
    rev = code_revision()

    torch.manual_seed(seed)

    tokenizer_repo = load_named_tokenizer(tokenizer_name).repo_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
    tokenizer.chat_template = MINIMAL_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, dtype=dtype
    )
    # `context_length` is read off the checkpoint's own config rather than
    # taken as a CLI flag -- it is not an independent choice, it is a property
    # of the checkpoint being fine-tuned (a mismatched value would either
    # truncate silently short or overflow the model's RoPE table).
    max_seq_len = model.config.context_length

    examples, data_stats = load_probe_dataset(seed=seed, token_budget=token_budget, tokenizer=tokenizer)
    train_dataset = Dataset.from_list(examples)

    sft_config = build_sft_config(out_dir=out_dir / "_trainer_scratch", seed=seed, max_seq_len=max_seq_len)
    # ShiftedLabelsSFTTrainer, not plain SFTTrainer -- see
    # _shift_labels_for_next_token_loss's docstring.
    trainer_cls = build_shifted_labels_sft_trainer_class()
    trainer = trainer_cls(model=model, args=sft_config, train_dataset=train_dataset, processing_class=tokenizer)
    train_result = trainer.train()

    manifest_fields = dict(
        step=0,  # SFT probes are single-pass, not step-numbered like pretraining
        tokens_consumed=data_stats["num_tokens_selected"],
        git_commit=rev["commit"],
        code_dirty=rev["dirty"],
        code_diff_sha256=rev["diff_sha256"],
        tokenizer_name=tokenizer_name,
        compute_dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
        autocast={"enabled": False},
        torch_version=torch.__version__,
        transformers_version=__import__("transformers").__version__,
        model_config=model.config.to_dict(),
        # SFT has no shard/offset position (unlike pretraining's PackedSeqDataset) --
        # the semantically equivalent "where did the training data come from and
        # how much of it" is `data_stats` itself. `source_checkpoint` lives here
        # too (not as a top-level `write_manifest` kwarg -- its signature is a
        # fixed keyword-only list with no **kwargs, verified by reading
        # `src/pretrain/manifest.py` directly) so a probe checkpoint's
        # manifest always answers "which pretraining checkpoint produced this".
        data_position={"sft_probe_source": data_stats, "source_checkpoint": str(checkpoint)},
        perturbation_probe={"runs_offline": True},
        storage_dtype="bfloat16",
        rng_state_saved=False,
    )
    # DDP: only the main process writes the checkpoint -- every rank running
    # `save_trajectory` concurrently would race on the same output directory.
    # See `_resolve_is_main_process`'s docstring for why the cross-check
    # against `_is_main_process()` lives in a helper (testable in isolation,
    # and raises rather than asserts).
    is_main_process = _resolve_is_main_process(trainer)
    if is_main_process:
        save_trajectory(model, out_dir, manifest_fields=manifest_fields)

    # Release the GPU memory this process has
    # been holding BEFORE `main()` shells out to `run_lm_eval` in a separate
    # process on the same GPU -- see `_release_cuda_memory_before_eval`'s
    # docstring. Every rank releases its own memory, not just the main one
    # (cheap and correct regardless of which rank ends up running eval).
    del trainer, model
    _release_cuda_memory_before_eval()

    return {
        "train_loss": train_result.training_loss,
        "data_stats": data_stats,
        "max_seq_len": max_seq_len,
        "is_main_process": is_main_process,
    }


def _release_cuda_memory_before_eval() -> None:
    """PyTorch's CUDA caching allocator only GROWS across a process's lifetime
    -- it never returns freed blocks to the CUDA driver on its own. A short
    (few-step) run's cache stays small and the eval subprocess (shelled out
    to next, in a SEPARATE process, on the SAME GPU) has no trouble finding
    free VRAM. But `src/posttrain/sft_formal.py`'s real formal-tier runs train
    for 2 epochs over ~392M tokens -- by the time training ends, the
    training process's cache has grown to occupy nearly the entire GPU
    (measured: 139.03 GiB held after a 300-step run on a 139.80 GiB card).
    The eval subprocess then fails to allocate its own copy of the model:
    `CUDA out of memory ... 253.50 MiB is free`. The training itself
    completes successfully and the checkpoint is written correctly -- this
    failure mode is misleading: a run that LOOKS like a training failure
    (job exits non-zero) when training actually succeeded and only the
    unrelated eval step, afterward, ran out of room.

    ⚠️ Calling this function alone does NOTHING unless the CALLER has
    already dropped every Python reference to the model/trainer/optimizer
    it built (`del trainer, model` in the caller's own scope, BEFORE this
    call) -- PyTorch's allocator can only release a block once it can prove
    nothing still references the tensors backing it. This function does not
    (and cannot, from here) delete the CALLER's local variables; `del`
    inside this function would only remove ITS OWN local names, not the
    caller's, which is why the actual `del` statements live at each call
    site instead of being hidden inside this helper.

    Must be called AFTER every step that still needs the model in memory
    (e.g. `save_trajectory`) and BEFORE shelling out to `run_lm_eval`.
    Guards `torch.cuda.is_available()` before touching any CUDA API --
    on a machine with no functional CUDA (e.g. a CPU-only dev machine, or a
    driver too old), `torch.cuda.empty_cache()` would either no-op or raise depending
    on the CUDA build; the guard makes the CPU-only path never touch the
    CUDA API at all, matching this module's existing convention (`use_cpu=
    not torch.cuda.is_available()` etc.) of never assuming CUDA works just
    because `torch` was imported.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Env vars `torchrun`/`torch.distributed.elastic` injects into every process
# it launches -- meaningful to a process that IS a member of that launch, and
# actively harmful to one that is NOT (see `_sanitized_eval_env`'s docstring).
# Fixed list, not "everything torchrun happens to set today": a set this
# module owns and can grep for, not implicit knowledge of torchrun's internals.
_TORCHRUN_ENV_VARS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
    "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
    "MASTER_ADDR", "MASTER_PORT",
)


def _sanitized_eval_env() -> dict[str, str]:
    """A copy of `os.environ` with every `torchrun`/`torch.distributed.elastic` variable
    removed (the fixed list above, plus anything prefixed `TORCHELASTIC_`).

    The bug this fixes: `run_lm_eval` shells out to a SEPARATE interpreter
    (the `looptrain-eval` env, no `trl`/training deps) to run `lm_eval`'s own
    harness -- a genuinely single-process, non-distributed task. But
    `subprocess.run` with no `env=` argument makes the child inherit the
    parent's FULL environment. Under the formal tier's real launch (`torchrun
    --nproc_per_node=N`), the parent `sft_formal.py` process has `RANK`/
    `WORLD_SIZE`/`MASTER_ADDR`/etc set (that's how it knows its own rank) --
    the eval subprocess inherits all of it, and `lm_eval`'s own `accelerate`
    dependency (`accelerate/state.py`'s `PartialState.__init__`, called from
    `lm_eval/models/huggingface.py`'s `HFLM.__init__`) reads `WORLD_SIZE` at
    import time and calls `torch.distributed.init_process_group()`, which
    blocks in a collective rendezvous waiting for `WORLD_SIZE - 1` OTHER
    processes that were never launched -- permanently. Confirmed by `py-spy
    dump` on the actual hung subprocess: the stack bottoms out exactly in
    `init_process_group` (`torch/distributed/distributed_c10d.py`), called via
    `accelerate.state.PartialState.__init__` <- `Accelerator.__init__` <-
    `lm_eval.models.huggingface.HFLM.__init__`. `N=1` (single-process
    invocations, e.g. the probe tier) never hits this: `WORLD_SIZE`
    is unset/`1`, `accelerate` sees no distributed launch, and doesn't call
    `init_process_group` at all -- which is exactly why this bug was invisible
    until a genuinely multi-process (`torchrun --nproc_per_node>1`) formal-tier
    run reached the eval stage.

    A formal-tier run's training + checkpoint save complete normally BEFORE
    this hang (confirmed: `train_runtime` ~10s, model shards written)
    -- so the run doesn't fail loudly, it silently sits there until the job's
    walltime is exhausted, looking indistinguishable from "still training".
    """
    env = dict(os.environ)
    for key in _TORCHRUN_ENV_VARS:
        env.pop(key, None)
    for key in list(env):
        if key.startswith("TORCHELASTIC_"):
            env.pop(key, None)
    return env


def run_lm_eval(
    *, out_dir: Path, eval_python: str, task_set: str, eval_limit: float | None, tokenizer_name: str,
) -> None:
    """Shell out to `src/eval/run_eval.py` in the caller-supplied
    `looptrain-eval` interpreter -- reuse, not reimplementation, of
    the existing lm-eval plumbing. A separate interpreter/process by the same
    design rationale `run_eval.py`'s own docstring states for the training loop:
    eval dependencies (`lm_eval` and its transitive deps) do not belong in the
    SFT env (`trl` + friends), any more than they belong in the training env.

    `env=_sanitized_eval_env()`: see
    that function's docstring -- without it, this subprocess inherits
    `torchrun`'s distributed env vars under a real multi-process formal-tier
    launch and hangs forever in a phantom `init_process_group` rendezvous.

    `task_set`: no default -- passed straight to `src.posttrain.eval_subset.
    resolve_eval_tasks` (`"probe"` or `"full"`); see that module's docstring
    for why the previous boolean `include_optional_tasks` flag was retired
    (it never actually reached the full 11-task LT2 suite).

    `tokenizer_name`: no default -- forwarded verbatim to `src.eval.run_eval`'s
    required `--tokenizer-name`. Must be the SAME name `run_sft`/`run_formal_sft`
    used for this checkpoint; the caller (`main`) is responsible for that,
    this function does not read it back from the checkpoint itself.
    """
    tasks = resolve_eval_tasks(task_set)
    # `--max-length` is required in `src.eval.run_eval` (lm_eval's
    # HFLM does not read our config's `context_length` and would otherwise take
    # the tokenizer's 8192, beyond the 4096-position RoPE table). Same rule as
    # `run_sft`'s `max_seq_len`: it is a property of the checkpoint, read off its
    # own config.json, not an independent CLI choice. Missing key -> KeyError
    # on purpose (a guessed window is exactly what this flag exists to prevent).
    max_length = json.loads((out_dir / "config.json").read_text())["context_length"]
    cmd = [
        eval_python, "-m", "src.eval.run_eval",
        "--checkpoint-dir", str(out_dir),
        "--tasks", ",".join(tasks),
        "--out", str(out_dir / "eval_results.json"),
        "--tokenizer-name", tokenizer_name,
        "--max-length", str(max_length),
    ]
    if eval_limit is not None:
        cmd += ["--limit", str(eval_limit)]
    # `cwd` pinned to the repo root (two levels above this file's directory) rather than left
    # to whatever directory `sft_probe.py` itself was invoked from -- `-m
    # src.eval.run_eval` only resolves if the repo root is importable,
    # and a caller running this script from elsewhere would otherwise get an
    # opaque `ModuleNotFoundError` from a subprocess two layers removed from
    # the actual cwd mistake.
    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run(cmd, check=True, cwd=repo_root, env=_sanitized_eval_env())


def run_instruction_probes(*, out_dir: Path, tokenizer_name: str) -> dict[str, Any]:
    """Cold-load the SFT output (not the still-in-memory trainer model) and run
    the fixed instruction-following prompts. Cold-loading matters: it is an
    acceptance check (the fine-tuned weights must cold-load) and it is the
    only way to be sure `save_trajectory`'s output is
    actually loadable, not just that the in-memory trainer object still works.

    `tokenizer_name`: no default -- must be the SAME name `run_sft` was given
    for this checkpoint; the caller (`main`) is responsible for passing the same value to both,
    this function does not read it back from the checkpoint itself.

    Runs IN-PROCESS (no `subprocess`/shell-out), so the torchrun
    env-inheritance bug (`run_lm_eval`'s docstring) does not apply here
    today. If this is ever changed to shell out to a separate process, the
    same `env=_sanitized_eval_env()` rule must go with it -- a
    torchrun-launched multi-process formal run would otherwise hit the exact
    same phantom-`init_process_group` hang for the SAME reason.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_repo = load_named_tokenizer(tokenizer_name).repo_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
    tokenizer.chat_template = MINIMAL_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True, dtype=torch.bfloat16)
    model.eval()

    results = []
    for probe in INSTRUCTION_PROBES:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": probe["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            # use_cache=False is REQUIRED, not a tuning choice: transformers'
            # default cache setup path reads `config.num_hidden_layers`, which
            # `LoopMoEConfig` does not define (its layer-count fields are
            # `num_layers_in_stack`/`num_stacks`) -- `generate()` with the
            # library default (`use_cache=True`) raises
            # `AttributeError: 'LoopMoEConfig' object has no attribute
            # 'num_hidden_layers'` before producing a single token (verified
            # against the toy checkpoint). `use_cache=False` falls
            # back to recomputing the full forward pass each generated token --
            # slower, but the probe generates a handful of short responses to
            # three fixed prompts, so the cost is negligible.
            generated = model.generate(
                input_ids, max_new_tokens=64, do_sample=False, use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
        verdict = run_instruction_probe(response, probe["rule"])
        results.append({"id": probe["id"], "prompt": probe["prompt"], "response": response, **verdict})

    return {"probes": results, "num_passed": sum(1 for r in results if r["passed"]), "num_total": len(results)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Input pretraining trajectory checkpoint (HF format)")
    ap.add_argument("--out", required=True, help="Output directory for the SFT'd checkpoint + eval JSONs")
    ap.add_argument(
        "--token-budget", type=int, required=True,
        help="How many SmolTalk tokens to train on. No default -- see this module's "
             "SUGGESTED_TOKEN_BUDGETS_UNCALIBRATED table for suggested starting points "
             "(explicitly flagged uncalibrated, not a fallback).",
    )
    ap.add_argument("--seed", type=int, required=True, help="Data shuffle + torch seed (reproducibility)")
    ap.add_argument(
        "--tokenizer-name", dest="tokenizer_name", choices=sorted(TOKENIZER_CLASSES), required=True,
        help="Which tokenizer this checkpoint expects -- see src.data.tokenizer.TOKENIZER_CLASSES. "
             "No default: loading a fixed tokenizer regardless of which checkpoint was passed in "
             "would silently mismatch a checkpoint trained with a different tokenizer.",
    )
    ap.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32",
        help="Model parameter dtype. Default 'float32' keeps the probe's original behavior, so "
             "an invocation without this flag is unaffected; pass --dtype bfloat16 explicitly "
             "for a bf16 dry-run.",
    )
    ap.add_argument(
        "--eval-python", required=True,
        help="Path to the `looptrain-eval` env's python interpreter (lm_eval is not installed "
             "in this script's own env by design -- see run_lm_eval's docstring)",
    )
    # No --lr/--batch-size/--warmup-steps/--weight-decay flags -- the SFT
    # recipe is fixed (`PROBE_RECIPE`, module docstring's "The probe recipe
    # ... is fixed, not a CLI knob"), not a per-invocation choice.
    ap.add_argument(
        "--eval-task-set", dest="eval_task_set", choices=("probe", "full"), required=True,
        help="Which lm-eval task set to run post-SFT: 'probe' = fast 3-task subset "
             "(arc_easy/piqa/hellaswag), 'full' = complete 11-task LT2 suite. No default -- "
             "replaces the retired --eval-include-optional/--no-eval-include-optional flag "
             "(that boolean never actually reached the full 11-task suite despite looking like "
             "it should, see src/posttrain/eval_subset.py's docstring).",
    )
    ap.add_argument("--eval-limit", dest="eval_limit", type=float, default=None,
                     help="Per-task lm-eval example cap (unset = full eval for the chosen task set)")
    args = ap.parse_args(argv)

    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[sft_probe] checkpoint={checkpoint} out={out_dir} token_budget={args.token_budget} "
          f"seed={args.seed} dtype={args.dtype} tokenizer_name={args.tokenizer_name}")
    sft_summary = run_sft(
        checkpoint=checkpoint, out_dir=out_dir, token_budget=args.token_budget, seed=args.seed,
        tokenizer_name=args.tokenizer_name, dtype=dtype,
    )
    print(f"[sft_probe] SFT done: {json.dumps(sft_summary, indent=2)}")

    # DDP: under a multi-process (torchrun) launch, every rank returns from
    # `run_sft` (the trainer's `train()` call is a collective op all ranks
    # must join), but only the rank that actually wrote the checkpoint
    # (`run_sft`'s `is_main_process`) should shell out to eval / run the
    # instruction probes against it -- every other rank has nothing new to
    # read yet, and running these `len(nproc)` times would be both wasted
    # work and (for `run_lm_eval`'s subprocess) `nproc` redundant processes
    # competing for the same GPUs. See `run_sft`'s docstring for why this is
    # the only DDP-specific control-flow change this module needs.
    if not sft_summary["is_main_process"]:
        return

    print(f"[sft_probe] running lm-eval ({args.eval_task_set} task set)...")
    run_lm_eval(
        out_dir=out_dir, eval_python=args.eval_python,
        task_set=args.eval_task_set, eval_limit=args.eval_limit, tokenizer_name=args.tokenizer_name,
    )

    print("[sft_probe] running instruction-following probes (cold-loaded checkpoint)...")
    instruction_results = run_instruction_probes(out_dir=out_dir, tokenizer_name=args.tokenizer_name)
    (out_dir / INSTRUCTION_PROBE_RESULTS_NAME).write_text(json.dumps(instruction_results, indent=2))
    print(f"[sft_probe] instruction probes: {instruction_results['num_passed']}/{instruction_results['num_total']} passed")

    print(f"[sft_probe] done. Artifacts in {out_dir}: pytorch_model.bin, manifest.json, "
          f"eval_results.json, {INSTRUCTION_PROBE_RESULTS_NAME}")


if __name__ == "__main__":
    main()

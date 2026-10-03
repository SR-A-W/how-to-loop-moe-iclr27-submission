"""The LT2-aligned training recipe, as code.

Data / optimiser / training budget / eval cadence follow LT2 (arXiv
2605.20670); the MoE-side hyperparameters come from seedvar (2605.09165).
Two rules govern everything here:

  1. Only settings that mainstream practice would accept -- no clever choices.
  2. **Every model in the first round trains with identical hyperparameters**,
     so behavioural differences are attributable to architecture.

Rule 2 is why this module exists: the recipe is defined **once** and every run
derives from it. A per-run copy of the numbers is how "identical hyperparameters"
quietly stops being true.

Where LT2's paper and its config files disagree, the config wins -- that is why peak LR is 3e-4 and not the 1.5e-4 the paper's prose
mentions for the 0.6B tier.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from src.pretrain.checkpointing.schedule import trajectory_steps

__all__ = ["LT2_RECIPE", "ABLATION_RECIPE", "TrainingRecipe", "RunPlan", "plan_run", "RUN_LIST", "ABLATION_AC_MODE", "ac_mode_for_ablation",
           "tokens_for_flop_budget", "SEEDVAR_ANCHOR_FLOPS",
           "RESUME_CHECKPOINT_INTERVAL_STEPS", "KEEP_CHECKPOINT_STEPS", "ABLATION_RESUME_INTERVAL_STEPS", "keep_latest_k_for",
           "command_for", "plan_continuation", "CONTINUATION_STEPS_100B", "CONTINUATION_FROM_STEP",
           "CONTINUATION_TRAJECTORY_100B", "CONTINUATION_KEEP_LATEST_K",
           "CONTINUATION_100B_LAUNCHED", "CONTINUATION_100B_HELD"]


WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = "looped-pretrain"
"""The single W&B project page for these runs: training curves (via
torchtitan's native WandBLogger), the offline eval curves, and eventually the
orchestrator status table all land here, keyed by run name. The W&B entity
comes from the `WANDB_ENTITY` environment variable; when unset, `WANDB_TEAM`
is omitted and wandb falls back to the logged-in user's default entity."""

RESUME_CHECKPOINT_INTERVAL_STEPS = 2000
"""How often torchtitan writes a **resumable** (DCP) checkpoint, in steps.

This is `--checkpoint.interval`, and it governs *only* crash/walltime insurance.
It is deliberately decoupled from the trajectory-checkpoint schedule
(`RunPlan.trajectory_checkpoints`), which is triggered independently by
`TRAJECTORY_STATE.steps` inside `LoopMoETrainer` and is unaffected by this value.

Why a fixed 2000 rather than a fraction of the run: the interval used to be
`steps // 10`, which ties resume frequency to *run length* -- a quantity that has
nothing to do with the only constraint that matters here, the cluster's walltime
window. At ~25k steps that meant roughly one resumable checkpoint every 48 hours,
longer than any walltime window the cluster grants. The first cycle of the
production runs hit walltime with **zero** resumable checkpoints and all three
runs restarted from step 0 (~23.5 h x 3 nodes lost); the job wrapper was then
patched to override the generated value to 2000.

**This is a default suggestion, not the value a run used.** It is overridden
according to how hard the cluster is preempting -- it was later cut to 500
after preemption made 2000 too coarse. The constant is no longer "the same
literal value the wrapper writes", so the override is no longer a no-op:
**the value a run actually used comes from the job script and, authoritatively,
from that checkpoint's manifest (`resume_checkpoint_interval_steps`)** -- never
from this constant. Pass `resume_checkpoint_interval` to `torchrun_command` to
generate a command with a different value.

2000 steps is ~0.5-1.2 h per save at post-dataloader-fix speed -- comfortably
inside every walltime window, so a killed job loses at most ~1 hour. At 500 the
save is ~4x more frequent, which is the trade made against preemption.

Disk is unchanged: resume checkpoints roll (1-2 kept), so steady-state footprint
does not depend on how often they are written -- only trajectory checkpoints
accumulate. See `src/pretrain/checkpointing/schedule.py`.
"""


@dataclass(frozen=True)
class TrainingRecipe:
    """Hyperparameters shared by every run in the first round."""

    # --- optimiser (LT2 Appendix C; paper and config agree) ---
    optimizer: str = "AdamW"
    beta1: float = 0.9
    beta2: float = 0.95
    """LT2's value. seedvar uses 0.999; the conflict resolves to LT2."""
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # --- schedule ---
    peak_lr: float = 3e-4
    """Config-file value. LT2's prose says 1.5e-4 for the 0.6B tier; config wins."""

    # --- learning-rate schedule: stated, never defaulted ---
    #
    # These three have NO default and are keyword-only, so every recipe has to
    # say what schedule it trains under. They used to default to cosine, which
    # made "which schedule did this run use" a question you answered by knowing
    # which default was in force on the day -- and the answer changes the LR at
    # nearly every step. A schedule is part of a run's stated configuration, not
    # a knob inherited from whoever wrote the dataclass.
    lr_decay_style: str = field(kw_only=True)
    """`linear`, `sqrt` or `cosine` -- torchtitan's `lr_scheduler.decay_type`."""

    lr_min_ratio: float = field(kw_only=True)
    """Floor as a FRACTION OF PEAK, not an absolute learning rate.

    torchtitan scales its decay factor from 1 down to this value and multiplies
    `optimizer.lr` by it, so 0.05 means "end at 5% of peak" (v0.2.2,
    `components/lr_scheduler.py`). Reading it as an absolute LR is off by four
    orders of magnitude, which is why the unit is named here rather than left to
    the field name."""

    lr_decay_ratio: float | None = field(kw_only=True)
    """Fraction of the run spent decaying, for a warmup-stable-decay schedule.

    `None` means the flag is not emitted at all, which leaves torchtitan
    decaying from the end of warmup -- the cosine shape every finished run used,
    with byte-identical launch commands, so nothing already dispatched is
    perturbed.

    Set it and decay is compressed into the last fraction of the run, with a
    constant-LR stretch before it."""
    warmup_steps: int = 5000
    """LT2's value (its 0.6B config), which is ~2% of its 255k-step run.

    Used as-is for the 100B-token runs. See `warmup_for()` for why short runs
    cannot take this number literally."""

    warmup_fraction: float = 0.02
    """Warmup as a fraction of the run, for runs too short to take 5,000 steps.

    This exists because of a real trap: the seedvar-anchor runs are ~2,270 steps,
    so a literal 5,000-step warmup is **longer than the entire run** -- the LR
    would ramp to only ~45% of peak, never reach it, and never cosine-decay. The
    run would complete, produce checkpoints and look superficially fine while
    having trained under a schedule nobody chose.

    Holding the *absolute* warmup identical across runs of different length is
    not what "identical hyperparameters" means; holding the *schedule shape*
    identical is. 2% reproduces LT2's own ratio (5,000/255,000 = 1.96%)."""

    # --- data / batch ---
    seq_len: int = 4096
    global_batch_seqs: int = 96
    """~393k tokens/step, matching LT2's stated ~4e5 target."""

    # --- precision ---
    param_dtype: str = "bfloat16"
    """LT2 is pure bf16, not fp32-master mixed precision (source-verified)."""
    reduce_dtype: str = "float32"
    """Gradient reduction stays fp32: standard practice, and cheap insurance on a
    loss whose aux terms are small relative to CE."""

    # --- instrumentation ---
    trace_every: int = 100
    """Steps between router/residual traces.

    Matches the metrics layer's free-tier N. Under FSDP the online trace is
    currently inert (known gap; the metrics are recomputed offline from the
    trajectory checkpoints), but the flag is required by the launcher and the
    value must have a stated source rather than being whatever someone typed.

    # --- budget ---
    """

    total_tokens: int = 97_556_063_699
    """The corpus we actually have, not the nominal 100B.

    Summed from `num_tokens` over all 140 shards of the tokenization manifest
    (no shard truncated). This is the *entire* output of FineWeb-Edu `sample-100BT` under
    the Llama-3 vocabulary; the 2.4% shortfall against LT2's nominal 100B is an
    intrinsic property of that dataset x tokenizer pairing, **not a gap in our
    data pipeline** -- there is no missing data to go and fetch.

    Budgeting the nominal 100B instead costs two things, one loud and one silent:

      1. the run walks off the end of the corpus around step ~248k and is killed
         by `--on-boundary raise`, i.e. a multi-day run ends in a failure exit;
      2. worse, the cosine schedule is derived from the step count, so a budget
         that is 2.4% too long perturbs the LR at **every step from the first**,
         and the tail never decays to the floor. This one leaves no error behind.
    """

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch_seqs * self.seq_len

    def steps_for(self, total_tokens: int | None = None) -> int:
        """Floor, deliberately: a partial final step has no data to fill it.

        Not cosmetic rounding. At the production budget `round` gives 248,098
        steps = 97,556,103,168 tokens, which is 39,469 tokens *past* the end of
        the corpus -- the run would die at the boundary on its very last step,
        after multiple days of compute. `floor` gives 248,097 and leaves 353,747
        tokens of headroom.
        """
        return (total_tokens or self.total_tokens) // self.tokens_per_step

    def warmup_for(self, steps: int) -> int:
        """Warmup for a run of `steps`: LT2's 5,000, capped at `warmup_fraction`.

        Long runs get exactly LT2's number; short runs get the same *shape*.
        """
        return min(self.warmup_steps, max(1, round(self.warmup_fraction * steps)))


LT2_RECIPE = TrainingRecipe(
    # The schedule the (architecture, tier) runs were trained under. Stated
    # rather than defaulted, and unchanged: these runs are finished, and their
    # launch commands stay byte-identical.
    lr_decay_style="cosine",
    lr_min_ratio=1e-6,
    lr_decay_ratio=None,
)

ABLATION_RECIPE = replace(
    LT2_RECIPE,
    # Warmup-stable-decay for the eighteen ablations. Warmup is not repeated here: `warmup_for(50_000)` already
    # yields 1,000 steps from the existing 2% rule, and restating it would be a
    # second source for one number.
    #
    # `lr_decay_ratio=0.10` is read by torchtitan as a fraction of TOTAL steps,
    # not of the post-warmup remainder.
    lr_decay_style="sqrt",
    lr_min_ratio=0.05,
    lr_decay_ratio=0.10,
)
"""The eighteen ablations' schedule: warm up 2%, hold, then decay the last 10%.

Not applied to the (architecture, tier) family, whose runs are finished.
"""


@dataclass(frozen=True)
class RunPlan:
    """One concrete training run."""

    name: str
    arch: str
    tier: str
    seed: int
    total_tokens: int
    recipe: TrainingRecipe = field(default=LT2_RECIPE)
    on_boundary: str = "raise"
    """Production default: fail fast when training
    catches up to the tokenize pipeline rather than blocking a whole PBS job
    silently. Resume is bit-exact, so the cost of stopping is one restart --
    whereas `"wait"` gives up the reproducible data order."""

    ac_mode: str = "selective"
    ac_option: str = "2"
    """Activation checkpointing, stated per run rather than inherited.

    torchtitan's default is *not* `none` -- it is `selective` with
    `selective_ac_option=2`, so every run has silently had AC applied by a value
    nobody in this project chose. Emitting it explicitly makes the memory/compute
    tradeoff a stated parameter, per the project rule that semantic parameters
    carry no silent defaults.

    Why `full` for DVF-c specifically (an earlier run OOM'd at 120GiB/140GiB):
    `_apply_layer_sac` keeps a running layer counter and checkpoints a layer only
    when `count % freq == 0`. With `freq=2` that assumes a deep stack. DVF-c has
    **L=1**, so the counter reaches 1, `1 % 2 != 0`, and **no layer is wrapped at
    all** -- verified directly: L=1 -> 0/1 layers checkpointed, L=4 -> 2/4.

    The architecture that needs AC most gets none: DVF-c is the maximally
    flattened config (L=1, R=16), so its activation memory is the highest of the
    six *and* it is the exact shape that defeats the every-Nth-layer heuristic.
    `full` wraps the single layer, which is re-entered R=16 times, so each of the
    16 loop steps stores only its boundary and recomputes its interior.
    """

    lb_loss_factor: float | None = None
    """Load-balancing coefficient, when this run deliberately departs from the
    registry's shared 0.01.

    `None` means "use the registry value", which is what every run but
    `noloop_moe_1p3b_lb0p02` does -- so their launch commands stay byte-identical
    to the ones already dispatched. The single source of truth for the default
    stays `config_registry.SHARED`; this field only records a departure from it.

    It is a field rather than a bare edit to `SHARED` because `SHARED` is what
    makes the other runs comparable: changing it there would silently move every
    run at once. Stated here, the departure is attached to the one run that
    intends it, rides into `model_config` in that run's manifests, and is visible
    in the generated launch command.
    """

    is_ablation: bool = False
    """This plan names an ablation configuration, not an (architecture, tier) pair.

    The ablation family is built by `config_registry.build_ablation(name)`, which
    carries its own width tier internally -- so the launch command must NOT pass
    `--tier`, and `train_titan` refuses it. Stated as a field rather than inferred
    from the name's spelling: a name is a label, and deriving how a run is built
    from how its label looks is how a typo becomes a different experiment.
    """

    resume_interval: int | None = None
    """This run's own resumable-checkpoint interval, or `None` for the module
    constant.

    The ablation family states 500 here rather than leaving it to whoever
    generates the command. It is not a preference: `KEEP_CHECKPOINT_STEPS`
    requires an interval that divides 45,000, and the constant's 2,000 does not,
    so a command generated without it could not keep the checkpoint the analysis
    needs. A requirement that only holds when the caller remembers an argument
    is not a requirement.
    """

    keep_checkpoint_steps: tuple[int, ...] = ()
    """Resumable checkpoints this run must still have when it ends.

    Empty for the (architecture, tier) family: those runs are finished, and
    emitting a `keep_latest_k` they never had would change their commands. The
    ablation family sets it to `KEEP_CHECKPOINT_STEPS`.
    """

    continuation_from_step: int | None = None
    """This plan continues an existing run from that run's checkpoint at this step.

    `None` means "train from initialisation", which is every plan except the
    continuations. When set, the launch command gains torchtitan's
    `--checkpoint.initial_load_path` (the source checkpoint, passed in by the
    caller -- a cluster path is not something this file can know) together with
    `--checkpoint.no_initial_load_model_only`, which is what makes torchtitan
    restore the optimizer, the LR scheduler's `last_epoch`, the dataloader
    position and `train_state.step` rather than the weights alone.

    Recorded as a step number rather than a boolean because the number is the
    claim being made: "this run starts where that one reached 45,000". A reader
    of the plan, and any pre-launch check of the source checkpoint, both need it.
    """

    trajectory_override: tuple[int, ...] = ()
    """Explicit trajectory steps, replacing the computed schedule.

    Empty means "use `trajectory_checkpoints`'s log-then-uniform rule". The
    continuation plans state their points outright because the rule cannot
    produce them: it spaces points over a run that starts at zero, and these runs
    start at 45,000 with a schedule chosen against the decay boundary
    (228,882) rather than against the run length.
    """

    debug_seed_override: int | None = None
    """`--debug.seed` when it differs from `--seed`.

    `seed` normally feeds both: the data order (ours) and torchtitan's RNG,
    which decides initialisation. A seed-variant run needs them to part company
    -- a different initialisation over the SAME data order -- and that is the
    whole design of the comparison: if both moved, a difference in the result
    could be either cause and the run would answer nothing.

    `None` means "both come from `seed`", which is every plan except the seed
    variants, so their commands are unchanged.
    """

    seed_variant_of: str | None = None
    """This plan is another run repeated from a different initialisation.

    Recorded as the source's name rather than a boolean for the same reason
    `continuation_from_step` holds a number: the claim is "this is S1 again with
    a different init", and the consumers that must exclude it from
    configuration counts, and the test that compares the two commands word for
    word, both need to know which run it is a variant OF.
    """

    attention_variant_of: str | None = None
    """This plan is another ablation's shape again, with per-pass attention
    (U1-U4).

    Recorded as the source's name for the same reason `seed_variant_of` is: the
    claim is "this is S1's shape, with `per_pass_attention=True`", and the
    consumers that must exclude it from the 26-configuration count -- it is not
    a 27th/28th/... shape, `config_registry.ATTENTION_VARIANTS` resolves it to
    an existing one -- need to know which shape it varies.
    """

    warmup_steps_override: int | None = None
    """`--lr_scheduler.warmup_steps` stated outright, instead of derived.

    `warmup_for` scales warmup with the run's length, which is right for a run
    that starts at step 0 and wrong for a continuation: warmup finished 45,000
    steps ago in the source run, and re-deriving it from the longer horizon
    would silently change a number the continuation command is otherwise
    required to match. The value carried here is the SOURCE run's warmup, so the
    two commands differ only where they are meant to.
    """

    keep_latest_k_override: int | None = None
    """`--checkpoint.keep_latest_k` stated outright.

    The derived value answers "how many must be kept so the protected steps
    survive to the end", and for a continuation there is nothing to protect in
    *this* run -- the 45,000 it needs lives in the source run. Without an
    override the flag would be omitted entirely and torchtitan's default of 10
    would apply silently, which is a different number from the 11 these runs are
    specified with.
    """

    trajectory_uniform_every: int | None = None
    """Override for the trajectory schedule's uniform stride.

    `None` keeps the default (a tenth of the run). The ablation family states
    4,000 explicitly, because the schedule it was budgeted for -- and every
    "compare checkpoint X across the eighteen" analysis -- assumes exactly the 15
    points that stride produces. A tenth of 50,000 is 5,000, which yields 13
    perfectly reasonable points that are not the agreed ones.
    """

    def build_config(self):
        """The model configuration this run trains.

        One place that knows which of the two builders applies, so a caller cannot
        get it wrong: `build(arch, tier)` for the (architecture, tier) family,
        `build_ablation(name)` for the ablation family. Before this existed, every
        consumer called `build(run.arch, run.tier)` directly, which raises the
        moment an ablation run appears in `RUN_LIST`.
        """
        from src.model.config_registry import build, build_ablation

        if self.is_ablation:
            return build_ablation(self.arch)
        return build(self.arch, self.tier, lb_loss_factor=self.lb_loss_factor)

    @property
    def steps(self) -> int:
        return self.recipe.steps_for(self.total_tokens)

    @property
    def warmup_steps(self) -> int:
        """The warmup this run's command actually carries.

        One place, because there were two: the command emitted the override and
        the listing header re-derived it from the run length, so a continuation
        printed "warmup 5,000" above a command that said 1000. Anything that
        reports a parameter must read the same value the command does, or the
        listing quietly describes a run nobody is going to launch.
        """
        if self.warmup_steps_override is not None:
            return self.warmup_steps_override
        return self.recipe.warmup_for(self.steps)

    @property
    def trajectory_checkpoints(self) -> list[int]:
        """Log-dense early, then every 10%.

        The early density is the point: defect formation is expected early, and
        the OLMoE router-saturation analysis this project mirrors depended on
        exactly those early points. A uniform schedule spends its budget on the
        boring tail.
        """
        # `first_step` scales for short runs, exactly as warmup does. With a
        # fixed 1000 the anchor runs (2,267 steps) skip the log phase entirely --
        # 1000 already exceeds the 10% uniform stride -- degenerating to a purely
        # uniform schedule. Since defect formation is expected early, that would
        # spend the whole checkpoint budget on the boring tail of precisely the
        # runs meant to observe the interesting part.
        if self.trajectory_override:
            return list(self.trajectory_override)
        first = max(1, min(1000, round(0.02 * self.steps)))
        uniform_every = self.trajectory_uniform_every or max(1, self.steps // 10)
        return trajectory_steps(self.steps, first_step=first, uniform_every=uniform_every)

    def torchrun_command(
        self,
        *,
        data_dir: str,
        out_dir: str,
        nproc: int = 8,
        resume_checkpoint_interval: int | None = None,
        initial_load_path: str | None = None,
    ) -> str:
        """The launch command for this run.

        `data_dir` is a parameter, never baked in: the 100BT shards are still
        being produced, and the first runs start against whatever prefix exists.

        `resume_checkpoint_interval` overrides the resumable-checkpoint interval
        in steps. It exists because it has to be retuned against the cluster's
        preemption behaviour (e.g. 2000 -> 500), which used to be done by
        `sed`-ing the generated command. A `sed` on a generated command is
        invisible to everything downstream: the recorded plan and the job that
        ran disagree, and nothing notices. Passing it here keeps the generator
        the single place the value is decided.
        """
        r = self.recipe

        # A continuation cannot be launched without the checkpoint it continues
        # from, and a from-scratch run must not be handed one. Both directions
        # are refused rather than defaulted: a continuation that silently starts
        # at step 0 would burn the full run before anyone noticed it was not a
        # continuation at all, and that is the whole failure this plan exists to
        # avoid.
        if self.continuation_from_step is not None and not initial_load_path:
            raise ValueError(
                f"{self.name} continues from step {self.continuation_from_step}, so "
                "torchrun_command needs initial_load_path=<that run's checkpoint "
                f"directory for step {self.continuation_from_step}>. The path is a "
                "cluster location, so it is passed in rather than stored here; check "
                "that it holds a complete checkpoint of that step before launching."
            )
        if self.continuation_from_step is None and initial_load_path:
            raise ValueError(
                f"{self.name} trains from initialisation, but initial_load_path="
                f"{initial_load_path!r} was given. Loading weights into a plan that "
                "does not declare itself a continuation would make the command "
                "disagree with the plan every downstream consumer reads."
            )

        # Explicit argument wins (set from current preemption behaviour), then the run's own value, then the module constant.
        _resume_interval = (
            resume_checkpoint_interval
            if resume_checkpoint_interval is not None
            else self.resume_interval
            if self.resume_interval is not None
            else RESUME_CHECKPOINT_INTERVAL_STEPS
        )
        # Rejected rather than clamped: a non-positive interval is accepted by
        # torchtitan as "never write a resumable checkpoint", which is exactly
        # the state that cost cycle 1 three runs' worth of node-hours. A typo
        # must fail here, not surface as a silently unprotected multi-day job.
        _keep_latest_k = (
            self.keep_latest_k_override
            if self.keep_latest_k_override is not None
            else keep_latest_k_for(self.steps, _resume_interval, self.keep_checkpoint_steps)
            if self.keep_checkpoint_steps
            else None
        )

        if _resume_interval <= 0:
            raise ValueError(
                f"resume_checkpoint_interval must be a positive number of steps, "
                f"got {_resume_interval!r}. This is crash/walltime insurance; "
                "disabling it is never what someone means to configure."
            )
        # W&B live metrics. torchtitan v0.2.2's
        # WandBLogger (components/metrics.py:147-160) takes NOTHING from the
        # CLI except the --metrics.enable_wandb switch -- entity, project, run
        # name and run id all come from environment variables read at
        # wandb.init(). So they ride as an env prefix on the command line:
        #   WANDB_TEAM / WANDB_PROJECT  -> the one shared project page
        #   WANDB_RUN_NAME / WANDB_RUN_ID = run name -> merges with the eval
        #     curves already uploaded under the same run id
        #   WANDB_RESUME=allow (read by the wandb SDK itself, not titan) ->
        #     a restarted job continues the SAME curve instead of erroring on
        #     the existing run id. This cluster kill-cycles often; without it
        #     the first restart after enabling W&B would crash at init.
        # Logging-only: no effect on training math or bit-exactness. The API
        # key is NOT here -- the job environment injects WANDB_API_KEY separately
        # (never in git).
        wandb_env = (
            (f"WANDB_TEAM={WANDB_ENTITY} " if WANDB_ENTITY else "")
            + f"WANDB_PROJECT={WANDB_PROJECT} "
            f"WANDB_RUN_NAME={self.name} WANDB_RUN_ID={self.name} WANDB_RESUME=allow"
        )
        return " \\\n  ".join(
            [
                f"{wandb_env} \\\ntorchrun --nproc_per_node={nproc} --rdzv_backend c10d --rdzv_endpoint localhost:0",
                "-m src.pretrain.train_titan",
                # An ablation configuration carries its own width tier, and
                # `train_titan` REFUSES `--tier` for one -- passing it would make
                # two sources for a single number.
                (f"--arch {self.arch}" if self.is_ablation
                 else f"--arch {self.arch} --tier {self.tier}"),
                # The run's identity, recorded in every checkpoint manifest.
                # Two runs can share an architecture (the round-2 twins differ
                # only in lb_loss_factor), so the arch alone cannot say which
                # run a checkpoint came from once it leaves its directory.
                f"--run-name {self.name}",
                f"--data {data_dir}",
                f"--seed {self.seed} --verify-sha256 --on-boundary {self.on_boundary}",
                # Omitted entirely when the run uses the registry's coefficient,
                # so previously generated commands are unchanged.
                *(
                    [f"--lb-loss-factor {self.lb_loss_factor}"]
                    if self.lb_loss_factor is not None
                    else []
                ),
                f"--trace-every {r.trace_every}",
                f"--activation_checkpoint.mode {self.ac_mode}",
                f"--activation_checkpoint.selective_ac_option {self.ac_option}",
                f"--job.dump_folder {out_dir}",
                f"--training.steps {self.steps}",
                f"--training.local_batch_size {r.global_batch_seqs // nproc}",
                f"--training.global_batch_size {r.global_batch_seqs}",
                f"--training.mixed_precision_param {r.param_dtype}",
                f"--training.mixed_precision_reduce {r.reduce_dtype}",
                f"--optimizer.name {r.optimizer}",
                f"--optimizer.lr {r.peak_lr}",
                f"--optimizer.beta1 {r.beta1} --optimizer.beta2 {r.beta2}",
                f"--optimizer.weight_decay {r.weight_decay}",
                f"--training.max_norm {r.grad_clip}",
                f"--lr_scheduler.warmup_steps {self.warmup_steps}",
                f"--lr_scheduler.decay_type {r.lr_decay_style}",
                f"--lr_scheduler.min_lr_factor {r.lr_min_ratio}",
                # Omitted entirely when unset, so the cosine runs already
                # dispatched keep byte-identical commands.
                *(
                    [f"--lr_scheduler.decay_ratio {r.lr_decay_ratio}"]
                    if r.lr_decay_ratio is not None
                    else []
                ),
                # torchtitan's own RNG seed, which is NOT the `--seed` above.
                # `--seed` is ours and sets the data order; torchtitan reads
                # `debug.seed` for everything else, and with it unset
                # `set_determinism` either seeds nothing (single process) or
                # broadcasts rank 0's ambient RNG state. Measured on S1: two runs
                # one flag apart gave identical weights with it and different
                # weights without it. Every run dispatched before this flag was
                # added therefore has initial weights that cannot be reproduced
                # from its command -- for those, the weights on the Hub are the
                # record, not this command.
                f"--debug.seed {self.debug_seed_override if self.debug_seed_override is not None else self.seed}",
                f"--parallelism.data_parallel_shard_degree {nproc}",
                "--checkpoint.enable",
                # Resume insurance only -- sized against the cluster's walltime
                # window, NOT against the run length or the trajectory schedule.
                # `resume_checkpoint_interval` overrides the default; it is set
                # from current preemption behaviour. See
                # RESUME_CHECKPOINT_INTERVAL_STEPS on why the constant is a
                # suggestion and the manifest is the authority.
                f"--checkpoint.interval {_resume_interval}",
                # Sized so the protected steps survive to the end of the run;
                # see KEEP_CHECKPOINT_STEPS for why torchtitan's default loses
                # them. Omitted entirely for runs with nothing to protect, so the
                # finished family's commands stay byte-identical.
                *([f"--checkpoint.keep_latest_k {_keep_latest_k}"] if _keep_latest_k else []),
                # Explicit, never defaulted: torchtitan's default makes the FINAL
                # checkpoint weights-only and therefore not resumable, which is
                # a silent trap when restarting a multi-day run.
                "--checkpoint.no-last_save_model_only",
                # A continuation, and only a continuation, carries these three.
                # `initial_load_path` is read ONLY while this run's own directory
                # has no checkpoint of its own (torchtitan checkpoint.py:623), so
                # a preempted continuation restarts from its own latest step
                # rather than falling back to 45,000 -- the behaviour a restart
                # needs, and the reason this run writes to a NEW directory.
                # `no_initial_load_model_only` is what brings the optimizer, the
                # scheduler's last_epoch and the dataloader position with it;
                # without it torchtitan would load the weights alone and the run
                # would restart the data from the beginning at the peak LR,
                # producing a plausible-looking curve from the wrong state.
                *(
                    [
                        f"--checkpoint.initial_load_path {initial_load_path}",
                        "--checkpoint.no_initial_load_model_only",
                        # Stated even though it equals --training.steps: the LR
                        # schedule is what the continuation is FOR, and leaving
                        # it implicit means a future edit to one of the two
                        # numbers silently reshapes the schedule.
                        f"--lr_scheduler.total_steps {self.steps}",
                    ]
                    if self.continuation_from_step is not None
                    else []
                ),
                # See the wandb_env comment above; the switch is the only part
                # of the W&B wiring torchtitan reads from the CLI.
                "--metrics.enable_wandb",
            ]
        )


def tokens_from_manifest(path: str | Path, *, expected_shards: int | None = None) -> int:
    """Total tokens in a tokenisation manifest, or a refusal.

    `total_tokens` sets the step count, and the step count sets the LR schedule,
    so a total that is quietly a few percent wrong perturbs the learning rate at
    every step from the first and leaves no error behind. Every way this number
    can be wrong is therefore a refusal, not a smaller number:

    * a missing shard -- the sum is simply lower and looks entirely plausible;
    * a duplicated `shard_id` -- counted twice, and the run walks off the end of
      the corpus days later;
    * `truncated_for_dev` -- a development shard holds a fraction of its real
      content, and nothing downstream distinguishes it from a complete one;
    * a non-positive `num_tokens` -- "present but empty" passes a mere key check.

    `expected_shards` is how the caller states what a complete corpus is; without
    it this function can only verify internal consistency, which is exactly what
    a truncated tokenisation run also has.
    """
    import json

    data = json.loads(Path(path).read_text())
    shards = data.get("shards")
    if not shards:
        raise ValueError(f"{path}: no 'shards' -- this is not a tokenisation manifest")

    ids = [s["shard_id"] for s in shards]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicated shard ids {duplicates}; their tokens would be counted twice")
    if expected_shards is not None and len(shards) != expected_shards:
        raise ValueError(
            f"{path}: {len(shards)} shards, expected {expected_shards}. A short manifest "
            "produces a plausible, smaller total rather than an error."
        )
    truncated = sorted(s["shard_id"] for s in shards if s.get("truncated_for_dev"))
    if truncated:
        raise ValueError(f"{path}: shards {truncated} are marked truncated_for_dev")
    bad = sorted(s["shard_id"] for s in shards if int(s["num_tokens"]) <= 0)
    if bad:
        raise ValueError(f"{path}: shards {bad} report num_tokens <= 0")

    return sum(int(s["num_tokens"]) for s in shards)


def expected_initial_loss(vocab_size: int) -> float:
    """ln(V): the loss of a correctly initialised model, which predicts ~uniform.

    The ignition check for a new run. It is worth stating in one place because
    the number moves with the tokenizer -- ln(128256) = 11.761784 under Llama-3,
    ln(49152) = 10.802673 under SmolLM2 -- and a run compared against the wrong
    vocabulary's constant looks wrong by ~1.0 nats while being perfectly healthy,
    or (worse) looks healthy while being ~1.0 nats off.
    """
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    return math.log(vocab_size)


#: Below this block depth, torchtitan's every-Nth-layer selective AC wraps
#: nothing at all and the run gets no activation checkpointing whatsoever.
AC_FULL_MAX_DEPTH = 2


def ac_mode_for_depth(depth: int) -> str:
    """Choose the activation-checkpointing mode from D, the layers in one block.

    torchtitan's `_apply_layer_sac` keeps a running layer counter and wraps a
    layer only when `count % freq == 0`, with `freq = selective_ac_option = 2`.
    That assumes a deep stack. At D=1 the counter only ever reaches 1, `1 % 2 != 0`,
    and **no layer is wrapped at all**; at D=2 exactly one of the two is. So the
    configurations with the HIGHEST activation pressure -- the flattened ones,
    which loop the block many times -- are precisely the ones the heuristic
    leaves unprotected. `full` wraps the block itself, which is re-entered on
    every loop step, so each step stores only its boundary.

    This was found the expensive way: an earlier run (D=1, 16 loops) OOM'd at
    120GiB/140GiB while its config said `selective`, which everyone read as "AC
    is on". Stating the rule here means the next flattened architecture cannot
    inherit that silence by being added without someone remembering the episode.

    AC trades recompute for stored activations and does not touch the
    mathematics of the update, so varying it across runs does not breach the
    "identical hyperparameters" rule in this module's docstring.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    return "full" if depth <= AC_FULL_MAX_DEPTH else "selective"


ABLATION_RESUME_INTERVAL_STEPS = 500
"""The ablation family's resumable-checkpoint interval.

500 rather than the module constant's 2,000 for two reasons that happen to
agree: it is what observed preemption on the training cluster called for, and
it is the coarsest interval that still writes step 45,000.
"""

KEEP_CHECKPOINT_STEPS: tuple[int, ...] = (45_000,)
"""Resumable checkpoints that must still exist when the run ends.

Step 45,000 is where the pilot's analysis branches from, so losing it costs the
whole run, not a convenience.

torchtitan cannot be told to protect a step. `keep_latest_k` sorts every
`step-N` directory and deletes all but the last k (v0.2.2,
`components/checkpoint.py:855-867`), with no notion of an exempt step, and the
default k is 10. Two consequences, both of which cost us the checkpoint:

* At the default `--checkpoint.interval 2000`, **there is no step-45,000
  checkpoint at all** -- 45,000 is not a multiple of 2,000.
* At an interval of 500, it is written and then purged: the last ten at step 50,000 are
  45,500 ... 50,000. It misses surviving by exactly one.

So the interval must divide the protected step, and k must span from it to the
end. `torchrun_command` computes k and refuses a command that cannot keep it,
rather than emitting one that silently cannot.
"""

def keep_latest_k_for(total_steps: int, resume_interval: int,
                      protected: tuple[int, ...] = KEEP_CHECKPOINT_STEPS) -> int:
    """How many checkpoints torchtitan must keep so `protected` survive.

    Refuses rather than rounds when the interval cannot produce a protected step:
    a run launched with an interval that skips 45,000 would finish, look healthy,
    and simply not have the checkpoint the analysis needs. That is exactly the
    kind of failure this project pays for days later.
    """
    if resume_interval <= 0:
        raise ValueError(f"resume_interval must be positive, got {resume_interval}")
    keep = 10  # torchtitan's own default, the floor for crash insurance
    for step in protected:
        if step > total_steps:
            continue          # not reachable in this run; nothing to protect
        if step % resume_interval:
            raise ValueError(
                f"step {step} must be kept but --checkpoint.interval "
                f"{resume_interval} never writes it ({step} % {resume_interval} = "
                f"{step % resume_interval}). Choose an interval that divides it."
            )
        # +1 for the protected checkpoint itself.
        keep = max(keep, (total_steps - step) // resume_interval + 1)
    return keep


AC_MODES = ("none", "selective", "full")
"""What torchtitan accepts for `--activation_checkpoint.mode` (v0.2.2,
`config/job_config.py`; it also has `memory_budget`, which we do not use)."""

#: Activation-checkpointing mode for each ablation configuration, one entry per
#: run rather than one rule for all eighteen.
#:
#: Why a table and not `ac_mode_for_depth`: the depth rule answers "how deep is
#: one block", but the question that decides this is "does the run fit in
#: memory", and those stopped coinciding when the vocabulary shrank to 49,152.
#: All eighteen execute the same SIXTEEN loop layers -- they differ only in how
#: those sixteen are split into blocks -- so a per-run measurement is the only
#: thing that can say where recomputation is still needed.
#:
#: Initially filled with exactly what `ac_mode_for_depth` produces; entries move
#: away from it only when a measurement says so.
ABLATION_AC_MODE: dict[str, str] = {
    # Small tier: no recomputation, on measurement rather than on the depth rule.
    # S14 with `none` held a reserved peak of 82.93 GiB at 0.716 s/step, against
    # a 115 GiB bar.
    # S14 is the heaviest of the fifteen in parameters (sixteen distinct layers,
    # none shared) while every S configuration executes the same sixteen loop
    # layers, so it bounds the family rather than merely sampling it.
    #
    # S3 and S4 are INFERRED, not measured. They execute the same sixteen loop
    # layers as S14 and carry fewer parameters, so they should sit below it. The
    # inference is called out because S4 at `none` is the shape that OOM'd an earlier
    # run -- at D=1 the blanket `selective` wrapped no layer, so that run had
    # no recomputation while its config said otherwise. That OOM was under the
    # 128,256-entry vocabulary, whose float32 logits are 2.6x the current ones,
    # which is what changed rather than the architecture. Both runs are finished
    # and will not be resubmitted, so these values only bind a future rerun:
    # **if either is relaunched, run 20 steps and check the memory first.**
    "S1": "none", "S2": "none", "S3": "none", "S4": "none",
    "S5": "none", "S6": "none", "S7": "none", "S8": "none",
    "S9": "none", "S10": "none", "S11": "none", "S12": "none",
    "S13": "none", "S14": "none", "S15": "none",
    # Loop-count series: the mode follows how many times the stack
    # is CALLED, not the tier, because that is what stored activations scale
    # with. Estimated by the decomposition `fixed + float32 logits (9.0 GiB) +
    # per-call activation x stored calls`, where `selective` stores about half
    # the calls and `full` stores only head and tail. Two anchors, agreeing to
    # 3%, which is why the shape of this is worth anything:
    #
    #   S14, `none`, 16 calls, 82.93 GiB -> 3.93 GiB per stored call
    #        (a measurement file)
    #   S9, `selective`, 32 calls, 83.91 GiB -> 4.05 GiB per stored call
    #        (a `memory:` log line, not a measurement file -- weaker
    #        evidence, recorded as such)
    #
    # The numbers below use 3.93:
    #
    #     N1  32 calls  none 144 | selective  81 | full 18
    #     N2  32 calls  none 145 | selective  82 | full 19
    #     N3  24 calls  none 113 | selective  66 | full 19
    #     N4  64 calls  none 270 | selective 145 | full 19
    #
    # against a 115 GiB bar. N4 is the reason the series is not uniform:
    # `selective` alone would not fit it.
    #
    # These are ESTIMATES from a single measured point, not measurements.
    # **After launch, read `memory:` at steps 10-20; above 115 GiB, drop one
    # level (none -> selective -> full) and relaunch.**
    "N1": "selective", "N2": "selective", "N3": "selective", "N4": "full",
    # Large tier keeps recomputation: nothing has been measured at d_model 1792,
    # and the small tier's result does not carry -- activation memory scales with
    # width, and L1/L2/L3 already sat at 83-84 GiB WITH `selective`.
    "L1": "selective", "L2": "selective", "L3": "selective",
    # L4/L5 (registered alongside UL3/UL4): `full`, not `selective`,
    # by the depth rule itself (`ac_mode_for_depth`, D<=2 -> full) -- D=2 (L4)
    # and D=1 (L5) are exactly the shapes `selective` fails to protect (it wraps
    # at most one of two layers at D=2, and zero at D=1; this is the same D=1/D=2
    # combination that OOM'd an earlier run under `selective`, at a smaller width).
    # `full` is the only safe choice at D=1 in particular: `selective`'s every-Nth
    # counter never reaches 2, so it wraps literally nothing.
    #
    # AC mode is a memory/speed knob only -- it does not touch what the run
    # computes (see `ac_mode_for_depth`'s own docstring) -- so this is the only
    # place L4/L5's plans differ from what UL1/UL2's registration pattern would
    # otherwise suggest.
    #
    # Capacity bar: 115 GiB (the H200 has 141 GiB). INFERRED, not directly
    # measured at D<=2/d_model=1792 specifically, but there is now a real
    # empirical anchor: UL1/UL2 carry the IDENTICAL total
    # parameter count as UL3/UL4 (~1.40B, same +~180-193M attention weights and
    # their optimizer state -- see `config_registry.ATTENTION_VARIANTS`), and
    # measured 83.85 / 84.97 GiB peak under `selective` in their first
    # few dozen steps -- comfortably under the 115 GiB bar. UL3/UL4 differ from
    # that anchor only in activation footprint (8/16 loop calls vs UL1/UL2's
    # 2/4), and `full` stores strictly less per call than `selective` does, so
    # if anything UL3/UL4 under `full` should sit BELOW the UL1/UL2 anchor, not
    # above it. **After launch, read `memory:` at steps 10-20 anyway**; `full`
    # is already the most conservative mode, so there is no further level to
    # drop to if it still does not fit (the actual escalation available at that
    # point is reducing batch size, which would change the recipe).
    "L4": "full", "L5": "full",
    # Pilot tier, d_model 512. `none` is a decision, not an omission: at this
    # width the activations fit without recomputation, and recomputing would
    # spend time the pilot exists to save. It is also why the depth rule and this
    # table have parted company.
    "P6": "none", "P1": "none", "P2": "none", "P3": "none",
}


def ac_mode_for_ablation(name: str) -> str:
    """The configured mode for one ablation, refusing anything unregistered.

    No fallback to the depth rule on a miss: a nineteenth configuration added to
    the registry and not to this table would silently inherit a mode nobody
    chose for it, which is the failure this table exists to end.
    """
    try:
        mode = ABLATION_AC_MODE[name]
    except KeyError:
        raise ValueError(
            f"no activation-checkpointing mode registered for {name!r}. Add it to "
            "ABLATION_AC_MODE -- deliberately, with a memory measurement behind the "
            "choice, rather than letting it inherit a default."
        ) from None
    if mode not in AC_MODES:
        raise ValueError(f"{name}: {mode!r} is not one of {AC_MODES}")
    return mode


def plan_run(name: str, arch: str, tier: str, seed: int, total_tokens: int | None = None) -> RunPlan:
    from src.model.config_registry import build

    return RunPlan(
        name=name, arch=arch, tier=tier, seed=seed,
        total_tokens=total_tokens or LT2_RECIPE.total_tokens,
        # Derived, not typed per run: the mode follows from the architecture's
        # depth, and a hand-set exception is a thing someone has to remember for
        # every architecture added later.
        ac_mode=ac_mode_for_depth(build(arch, tier).num_layers_in_stack),
    )


#: The ablation family's budget: 50,000 steps at the
#: shared 393,216 tokens/step (~20B tokens, a single epoch over a fixed prefix).
#: Written as steps x tokens-per-step rather than as a token count, because the
#: step count is the number the recipe states and the number the LR schedule is
#: derived from -- a token count would have to be reverse-engineered back into it.
ABLATION_STEPS = 50_000

#: The trajectory stride the recipe's 15 checkpoints assume (1k, 2k, 4k, 8k, 12k,
#: 16k, then every 4k to 48k, plus 50k). See `RunPlan.trajectory_uniform_every`.
ABLATION_TRAJECTORY_UNIFORM_EVERY = 4_000


def plan_ablation(name: str, seed: int = 42) -> RunPlan:
    """One of the eighteen ablation runs, built from its registry entry.

    The launch command, the checkpoint schedule and the activation-checkpointing
    mode all follow from the registry, so adding a nineteenth configuration there
    is enough to make it launchable -- nothing here has to be edited in step.

    `ac_mode` comes from `ABLATION_AC_MODE`, one entry per configuration, rather
    than from the recipe table's blanket "selective (every 2 layers)". At D=1
    (S4) that setting wraps NO layer at all -- the counter never reaches 2 -- and
    at D=2 (S3) it wraps exactly one of the two, so the flattened configurations,
    which loop the block the most, are the least protected by it. That is the
    shape that OOM'd an earlier run.
    """
    from src.model.config_registry import ABLATION_TIERS, ABLATIONS, build_ablation

    if name not in ABLATIONS:
        raise ValueError(f"unknown ablation {name!r}; choose from {sorted(ABLATIONS)}")
    config = build_ablation(name)
    return RunPlan(
        name=name,
        arch=name,
        tier=ABLATION_TIERS[name],
        seed=seed,
        total_tokens=ABLATION_STEPS * LT2_RECIPE.tokens_per_step,
        recipe=ABLATION_RECIPE,
        ac_mode=ac_mode_for_ablation(name),
        keep_checkpoint_steps=KEEP_CHECKPOINT_STEPS,
        resume_interval=ABLATION_RESUME_INTERVAL_STEPS,
        is_ablation=True,
        trajectory_uniform_every=ABLATION_TRAJECTORY_UNIFORM_EVERY,
    )


#: The 100B-token continuation.
#:
#: 100e9 / 393,216 tokens-per-step = 254,313 steps. Not derived here from a token
#: count, because 100B divided by the batch is not an integer and the number the
#: cluster and this file must agree on is the STEP count.
CONTINUATION_STEPS_100B = 254_313

#: Where the continuation picks up: the step every ablation run protected.
CONTINUATION_FROM_STEP = 45_000

#: Trajectory checkpoints for the 100B continuations.
#:
#: Every 20,000 steps, plus two stated points: 228,882 is the last step BEFORE
#: the decay begins (254,313 x 0.9), so it is the last look at the model under
#: the constant LR, and 254,313 is the final model. The 45,000 starting point is
#: deliberately NOT re-saved -- it already exists in the source run, and writing
#: it again under a new run name would give the same weights two identities.
CONTINUATION_TRAJECTORY_100B: tuple[int, ...] = tuple(sorted(
    list(range(60_000, 240_001, 20_000)) + [228_882, CONTINUATION_STEPS_100B]
))

#: `--checkpoint.keep_latest_k` for the continuations: 11 at interval 500.
CONTINUATION_KEEP_LATEST_K = 11


def plan_continuation(
    source: str,
    *,
    suffix: str,
    steps: int,
    trajectory: tuple[int, ...],
    from_step: int = CONTINUATION_FROM_STEP,
    keep_latest_k: int = CONTINUATION_KEEP_LATEST_K,
) -> RunPlan:
    """A plan that continues `source` from its checkpoint at `from_step`.

    Built by copying the source plan and changing only what the continuation
    changes -- name, length, trajectory, and the continuation fields. Everything
    else (seed, recipe, batch, activation checkpointing, resume interval) is
    inherited rather than restated, because "identical to the original except
    for X" is the claim a reader will check, and restating twenty values
    is how that claim quietly stops being true.

    `arch` deliberately stays the source's: a continuation is the same
    architecture trained longer, and registering a new architecture name for it
    would create a configuration that exists in the registry, appears in every
    "how many configurations are there" count, and has no model behind it.
    """
    import dataclasses

    base = next((plan for plan in RUN_LIST if plan.name == source), None)
    if base is None:
        raise ValueError(
            f"no run named {source!r} in RUN_LIST; a continuation must name the run "
            "it continues, so that its command can be compared against that run's."
        )
    if from_step > base.steps:
        raise ValueError(
            f"{source} runs {base.steps} steps and never reaches step {from_step}."
        )
    if steps <= from_step:
        raise ValueError(
            f"continuation of {source} would end at step {steps}, at or before its "
            f"own starting point {from_step}."
        )
    if trajectory and max(trajectory) != steps:
        raise ValueError(
            f"the last trajectory point ({max(trajectory)}) is not the final step "
            f"({steps}); the final model would never be written."
        )
    return dataclasses.replace(
        base,
        name=f"{source}{suffix}",
        total_tokens=steps * base.recipe.tokens_per_step,
        continuation_from_step=from_step,
        trajectory_override=trajectory,
        keep_latest_k_override=keep_latest_k,
        warmup_steps_override=base.recipe.warmup_for(base.steps),
        keep_checkpoint_steps=(),
    )


SEEDVAR_ANCHOR_FLOPS = 1e18
"""seedvar's own compute budget for the checkpoints we anchor against.

Chosen over a default of 100B tokens: the anchor runs exist to be *terminal-state comparable* with the published checkpoints, and
training them 100x longer would destroy exactly that. Everything else about them
(optimiser, LR, schedule, batch) stays identical to the other runs.
"""


def tokens_for_flop_budget(config, flops: float) -> int:
    """Tokens that exhaust `flops` under seedvar's accounting, C = 6*N_active*D.

    N_active counts looped layers once per call and only the `k` activated
    experts. This is seedvar's convention, deliberately *not*
    our attention-inclusive `get_nparams_and_flops` -- matching their budget
    means matching how they counted it.
    """
    d, L, R = config.d_model, config.num_layers_in_stack, config.num_stacks
    E, k = config.num_experts, config.num_active
    per_layer = 4 * d * d + d * E + k * 3 * d * (config.d_ff // k)
    n_active = R * L * per_layer + d * config.vocab_size
    return round(flops / (6 * n_active))


def _anchor_tokens() -> int:
    from src.model.config_registry import build

    return tokens_for_flop_budget(build("baseline", "seedvar_d704"), SEEDVAR_ANCHOR_FLOPS)


#: The first-round training list. One 8xH200 node each.
#:
#: The three `seedvar_anchor_*` runs are **not** a 216M replication and are
#: deliberately not named as one: with the Llama-3 vocabulary this architecture
#: totals ~326M parameters (55% of them embeddings) against the published 216M
#: with a GPT-2 vocabulary. The architecture is reproduced; the parameter count
#: and the embedding/compute balance are not. Comparison with the public
#: checkpoints is directional only.
#:
#: ⚠️ Matching seedvar's *FLOP* budget does not match its *token* count: our
#: larger vocabulary raises N_active, so 1e18 FLOPs buys ~0.89B tokens where
#: seedvar's own model would have seen ~1.26B. Matching one means missing the
#: other; the FLOP budget is what was chosen, and is what its IsoFLOP study is
#: indexed by.
def _ablation_names() -> tuple[str, ...]:
    """The ablation names, in the registry's own order.

    Read from the registry rather than typed here: a list in two places is a list
    that will disagree, and the way it would show up is a configuration that
    exists, builds, and simply never gets a launch command.
    """
    from src.model.config_registry import ABLATIONS

    return tuple(ABLATIONS)


_ABLATION_NAMES = _ablation_names()

RUN_LIST: tuple[RunPlan, ...] = (
    plan_run("dvf_a_0p6b", "dvf_a", "0p6b", seed=42),
    plan_run("dvf_a_1p3b", "dvf_a", "1p3b", seed=42),
    plan_run("dvf_c_1p3b", "dvf_c", "1p3b", seed=42),  # D=1 -> full, by rule
    plan_run("seedvar_anchor_s42", "baseline", "seedvar_d704", seed=42, total_tokens=_anchor_tokens()),
    plan_run("seedvar_anchor_s43", "baseline", "seedvar_d704", seed=43, total_tokens=_anchor_tokens()),
    plan_run("seedvar_anchor_s44", "baseline", "seedvar_d704", seed=44, total_tokens=_anchor_tokens()),
    # Round 2: the conventional comparator for the expert-collapse comparison
    # (our own trained comparator).
    # Everything except the (L, R, E) triple is deliberately identical to
    # `dvf_a_1p3b` above -- same corpus, same seed, same recipe -- because that is
    # the whole basis for attributing a difference to expert placement.
    #
    # It is listed here, and not merely in `config_registry.ARCHITECTURES`, because
    # RUN_LIST is what `print_run_commands.py` iterates. An architecture present in
    # the registry but absent from RUN_LIST has a generated launch command of
    # "nothing".
    plan_run("baseline_1p3b", "baseline", "1p3b", seed=42),
    # The non-looped half of the "looped vs not looped" pair: same effective
    # depth (16 layer applications), same expert pool (E=8) and k, therefore the
    # same per-token active parameters and FLOPs as `baseline_1p3b` above -- only
    # the weight sharing differs. `selective` recompute is kept (not `full`):
    # both runs unroll to 16 applications and wrap half of them, so activation
    # memory is comparable, and keeping the setting equal keeps their wall-clock
    # comparable too. See `config_registry`'s `noloop_moe` comment for why this
    # architecture stores twice the expert parameters on purpose.
    plan_run("noloop_moe_1p3b", "noloop_moe", "1p3b", seed=42),
    # The balance-matched twin of the run above,
    # launched alongside it rather than replacing it.
    #
    # Why the coefficient is doubled for this architecture and only this one.
    # The load-balancing loss pushes each router towards spreading tokens over
    # its own pool, and the whole model carries L*E expert slots. The DVF family
    # and `baseline` all have L*E = 64; `noloop_moe` has 128, so at the shared
    # 0.01 each expert slot feels half the balancing pressure the other runs
    # apply. Doubling to 0.02 restores per-expert pressure, at the cost of
    # departing from "identical hyperparameters across runs" -- which is exactly
    # why it is a separate run rather than an edit to the existing one: with
    # both in flight, the coefficient is a variable we measured rather than a
    # choice we had to get right in advance.
    #
    # Everything else is byte-identical to `noloop_moe_1p3b`: same architecture,
    # tier, seed, token budget, recipe and recompute setting. The z-loss
    # coefficient is deliberately NOT touched -- it penalises router logit
    # magnitude, which has no L*E dependence to compensate for.
    RunPlan(
        name="noloop_moe_1p3b_lb0p02",
        arch="noloop_moe",
        tier="1p3b",
        seed=42,
        total_tokens=LT2_RECIPE.total_tokens,
        lb_loss_factor=0.02,
    ),
    # The eighteen ablation runs. Listed here for the same reason the comment above
    # gives: `config_registry.ABLATIONS` is where they are DEFINED, but RUN_LIST is
    # what `print_run_commands.py` iterates. Registered but unlisted means
    # `--config S1` produces no command at all.
    *(plan_ablation(_name) for _name in _ABLATION_NAMES),
)

#: Configurations continuing to 100B tokens.
#:
#: Fifteen are launched: seven S/L configurations; the per-pass-attention
#: variants `U1`-`U4` (AC stays the U plans' `none`); `UL1`-`UL2`, continued
#: straight after their 20B runs on the same node (AC stays the UL plans'
#: `selective`); and `UL3`-`UL4`, continued the same way -- AC stays the
#: UL3/UL4 plans' `full` (their base shapes L4/L5 are D<=2, where `selective`
#: wraps at most one layer; see `ABLATION_AC_MODE`). Two are planned but held
#: back -- `L3` and `S4`. They are listed here anyway so that "planned but not
#: launched" is a fact recorded in the plan -- and so the day one is launched,
#: its command comes from the same generator instead of being typed.
#:
#: `U1`-`U4` and `UL1`-`UL4` are sources here, so their plans must exist in
#: RUN_LIST before the continuations are built: the attention-variant block
#: below therefore precedes the continuation block.
CONTINUATION_100B_LAUNCHED: tuple[str, ...] = (
    "S1", "S2", "S3", "S6", "S14", "L1", "L2",
    "U1", "U2", "U3", "U4",
    "UL1", "UL2", "UL3", "UL4",
)
CONTINUATION_100B_HELD: tuple[str, ...] = ("L3", "S4")

def plan_seed_variant(source: str, *, debug_seed: int, ac_mode: str) -> RunPlan:
    """`source` repeated from a different initialisation.

    Everything is inherited from the source plan except the run name, the
    initialisation seed, and `ac_mode`. The data-order seed is NOT touched: the
    point of the run is that initialisation is the only difference.

    `ac_mode` has no default and must be stated by the caller, because the value
    wanted here is **what the source run actually launched with**, which is not
    necessarily what the table says today. `S1` ran under `selective`; the table
    moved it to `none` after the S14 memory measurement, which came after `S1`
    was dispatched. A seed comparison against a
    run that recomputed activations differently is not a seed comparison.
    """
    import dataclasses

    base = next((plan for plan in RUN_LIST if plan.name == source), None)
    if base is None:
        raise ValueError(f"no run named {source!r} in RUN_LIST to vary the seed of")
    if debug_seed == base.seed:
        raise ValueError(
            f"{source} already initialises from seed {base.seed}; a variant with the "
            "same seed would be the same run under a second name."
        )
    return dataclasses.replace(
        base,
        name=f"{source}_seed{debug_seed}",
        debug_seed_override=debug_seed,
        seed_variant_of=source,
        ac_mode=ac_mode,
    )


#: The seed-variant plans: one run for now.
#: `ablation_names.SEED_NAMES` is the torch-free copy of these names; the
#: assertion below keeps the two from drifting.
SEED_VARIANTS: tuple[tuple[str, int, str], ...] = (
    # `selective`, not the table's current `none`: this is what S1 actually ran
    # with. See plan_seed_variant's docstring.
    ("S1", 43, "selective"),
)


def plan_attention_variant(name: str) -> RunPlan:
    """U1-U4: the source ablation's plan (S1-S4), with per-pass attention.

    Every field but `name` and `arch` is copied byte-for-byte from the source
    plan -- same tier, AC mode, seed, token budget, trajectory, resume interval
    -- because the only thing under test is `per_pass_attention`, which
    `config_registry.build_ablation` applies from the new name via
    `ATTENTION_VARIANTS`. A plan that touched anything else would be answering
    a different question than "does per-pass attention change the outcome". `is_ablation` stays True (inherited), which
    is what makes `torchrun_command` omit `--tier` and `RunPlan.build_config`
    route through `build_ablation(self.arch)` -- unchanged from the source.
    """
    import dataclasses

    from src.model.config_registry import ATTENTION_VARIANTS

    if name not in ATTENTION_VARIANTS:
        raise ValueError(f"unknown attention variant {name!r}; choose from {sorted(ATTENTION_VARIANTS)}")
    source = ATTENTION_VARIANTS[name]
    base = next((plan for plan in RUN_LIST if plan.name == source), None)
    if base is None:
        raise ValueError(f"no run named {source!r} in RUN_LIST to build {name!r} from")
    return dataclasses.replace(base, name=name, arch=name, attention_variant_of=source)


#: `ablation_names.VARIANT_NAMES` is the torch-free copy of U1-U4/UL1-UL4; the
#: assertion below keeps the two from drifting. Built BEFORE the continuations
#: because U1-U4 AND UL1-UL4 are all continuation sources
#: (`CONTINUATION_100B_LAUNCHED`) -- their plans must exist in RUN_LIST first.
RUN_LIST = RUN_LIST + tuple(
    plan_attention_variant(_name)
    for _name in ("U1", "U2", "U3", "U4", "UL1", "UL2", "UL3", "UL4")
)

from src.pretrain.ablation_names import VARIANT_NAMES as _VARIANT_NAMES  # noqa: E402

# A continuation of U2 inherits `attention_variant_of="S2"` -- it IS still S2's
# shape with per-pass attention -- so the check is over from-scratch plans only.
assert {p.name for p in RUN_LIST if p.attention_variant_of is not None
        and p.continuation_from_step is None} == set(_VARIANT_NAMES), (
    "the attention-variant plans and ablation_names.VARIANT_NAMES have drifted apart: "
    f"only in RUN_LIST: {{p.name for p in RUN_LIST if p.attention_variant_of is not None and p.continuation_from_step is None}} - {set(_VARIANT_NAMES)}; "
    f"only in VARIANT_NAMES: {set(_VARIANT_NAMES)} - RUN_LIST"
)

RUN_LIST = RUN_LIST + tuple(
    plan_continuation(
        _source,
        suffix="_100B",
        steps=CONTINUATION_STEPS_100B,
        trajectory=CONTINUATION_TRAJECTORY_100B,
    )
    for _source in CONTINUATION_100B_LAUNCHED + CONTINUATION_100B_HELD
)

RUN_LIST = RUN_LIST + tuple(
    plan_seed_variant(_source, debug_seed=_seed, ac_mode=_ac)
    for _source, _seed, _ac in SEED_VARIANTS
)

# The torch-free name list and the real plans, pinned together at import time:
# a seed plan added in one place and not the other would be a run the validator
# rejects, or a name the validator accepts with no plan behind it.
from src.pretrain.ablation_names import SEED_NAMES as _SEED_NAMES  # noqa: E402

assert {p.name for p in RUN_LIST if p.seed_variant_of is not None} == set(_SEED_NAMES), (
    "SEED_VARIANTS and ablation_names.SEED_NAMES have drifted apart: "
    f"only in RUN_LIST: {{p.name for p in RUN_LIST if p.seed_variant_of is not None}} - {set(_SEED_NAMES)}; "
    f"only in SEED_NAMES: {set(_SEED_NAMES)} - RUN_LIST"
)


def command_for(
    plan: RunPlan,
    *,
    data_dir: str,
    out_dir: str,
    nproc: int = 8,
    continuation_source: str | None = None,
    **kwargs,
) -> str:
    """`plan.torchrun_command(...)` for any plan, continuation or not.

    Exists because a continuation REQUIRES a starting checkpoint and a
    from-scratch run REFUSES one, so every caller that sweeps `RUN_LIST` -- the
    command printer, the launch scripts -- otherwise has to
    branch on it, and the ones that forget crash on the eight continuation plans
    rather than skipping them.

    `continuation_source` is a *template* containing `{run}` and `{step}`, e.g.
    `"/runs/{run}/checkpoint/step-{step}"`, filled in from the plan. It has no
    default: where a cluster keeps its checkpoints is not something this file can
    guess, and guessing would produce a launchable command pointing at a path
    nobody chose.
    """
    if plan.continuation_from_step is None:
        return plan.torchrun_command(data_dir=data_dir, out_dir=out_dir, nproc=nproc, **kwargs)
    if continuation_source is None:
        raise ValueError(
            f"{plan.name} continues from step {plan.continuation_from_step}; pass "
            "continuation_source='<template with {run} and {step}>' so the command "
            "names a real checkpoint."
        )
    source_run = plan.name.rsplit("_", 1)[0]
    return plan.torchrun_command(
        data_dir=data_dir,
        out_dir=out_dir,
        nproc=nproc,
        initial_load_path=continuation_source.format(run=source_run, step=plan.continuation_from_step),
        **kwargs,
    )

"""torchtitan Trainer subclass that adds this project's instrumentation.

torchtitan's own checkpointing writes sharded DCP (`step-N/*.distcp`). That is the
right format to *resume* from and the wrong format for everything else we need:
it is not loadable by `from_pretrained`, carries no manifest, and is therefore
invisible both to the diagnostics pipeline and to the eval watcher.

Since the trajectory checkpoints are the project's primary scientific output —
and, being a time series, cannot be reconstructed after the fact — a production
run that emits only DCP is a run whose main product is missing. This subclass
adds, on top of everything torchtitan already does:

  1. **per-step loss components** to the metrics collector (task/lb/z + coefficients);
  2. **router/residual traces** every N steps;
  3. **HF-format trajectory checkpoints** on the log-dense-then-uniform schedule,
     identical in format to what `train_toy.py` produces — same function, so the
     two cannot drift.

Nothing here replaces torchtitan's DCP checkpointing; both run, for different
purposes (resume vs. science).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from torchtitan.tools.logging import logger
from torchtitan.train import Trainer

from src.pretrain.checkpointing.store import save_trajectory
from src.model.loop_trace import LossComponents, assert_trace_released
from src.metrics.provenance import resolve_code_revision

__all__ = ["LoopMoETrainer", "TRAJECTORY_STATE"]


class _TrajectoryState:
    """Run-scoped settings the launcher hands to the trainer.

    A module-level object rather than constructor arguments because torchtitan
    instantiates the Trainer itself (`main(trainer_class)`), so there is no call
    site of ours to pass anything through.
    """

    def __init__(self) -> None:
        self.steps: frozenset[int] = frozenset()
        self.out_dir: Path | None = None
        self.trace_every: int = 0
        self.collector: Any = None
        self.config: Any = None
        self.run_name: str = "run"
        self.training_steps: int | None = None
        """The run's total step count as launched.

        A checkpoint that has left its directory cannot otherwise tell a full run
        from a shortened one -- and a shortened run has a different LR schedule at
        every step, not merely a different ending."""

        self.steps_overridden: bool = False
        """True when `--training.steps` differed from the registered plan's."""

        self.arch: str | None = None
        """The registry entry this run was built from (`dvf_a`, `S1`, ...).

        Separate from `manifest_run_name`: two runs can share an architecture
        (the round-2 twins differ only in their balance coefficient), so neither
        name answers the other's question."""

        self.manifest_run_name: str | None = None
        """The run's name as it goes into checkpoint manifests. Distinct from
        `run_name` above, which falls back to the architecture for commands
        generated before `--run-name` existed: a manifest must not record an
        architecture name in a field that claims to name the run."""

    def configure(
        self, *, steps, out_dir, trace_every, collector, config, run_name,
        manifest_run_name=None, arch=None, training_steps=None, steps_overridden=False,
    ) -> None:
        self.steps = frozenset(steps)
        self.out_dir = Path(out_dir)
        self.trace_every = trace_every
        self.collector = collector
        self.config = config
        self.run_name = run_name
        self.manifest_run_name = manifest_run_name
        self.arch = arch
        self.training_steps = training_steps
        self.steps_overridden = steps_overridden


TRAJECTORY_STATE = _TrajectoryState()


class LoopMoETrainer(Trainer):
    """Trainer with trajectory checkpoints and metric collection wired in."""

    _warned_empty_trace = False
    _pending_grad_norm: float | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._install_grad_norm_capture()

    # -- grad_norm capture ---------------------------------------------------

    def _install_grad_norm_capture(self) -> None:
        """Tee the `grad_norm` float torchtitan already computes into
        `self._pending_grad_norm`, so `_emit_metrics` can log it.

        Why a wrapper and not a read of the value ourselves: torchtitan's
        `grad_norm` is a GPU tensor local to its `train_step`, never stored on
        `self`, and it crosses to the host exactly once -- as the 4th argument
        of `metrics_processor.log`, inside torchtitan's own
        `should_log(step)` branch. Reading that argument costs nothing and
        adds no GPU->CPU sync; computing it ourselves would add one per step.
        That is also why the field is sparse (log steps only) -- see
        `collector.loss_record`'s docstring.

        Purely additive: the wrapper forwards every argument untouched and
        returns whatever torchtitan's method returns. No training math, no
        logging of torchtitan's own, is altered.
        """
        log_fn = self.metrics_processor.log
        # Fail at construction, not at hour 60. We pin torchtitan to an exact
        # sha, so this signature is fixed -- but a pin bump that renamed or
        # moved this parameter would otherwise surface as a `grad_norm` field
        # that is silently absent from an entire multi-day run's metrics.
        params = list(inspect.signature(log_fn).parameters)
        if "grad_norm" not in params:
            raise RuntimeError(
                "torchtitan's MetricsProcessor.log has no `grad_norm` parameter "
                f"(found: {params}). The grad_norm capture in {__name__} was "
                "written against torchtitan v0.2.2 and must be updated before "
                "this run can log gradient norms."
            )
        index = params.index("grad_norm")

        def _capturing_log(*args, **kwargs):
            if "grad_norm" in kwargs:
                self._pending_grad_norm = kwargs["grad_norm"]
            elif len(args) > index:
                self._pending_grad_norm = args[index]
            return log_fn(*args, **kwargs)

        self.metrics_processor.log = _capturing_log

    def train_step(self, data_iterator):
        # Arm the trace before the forward: the model reads this attribute and
        # fills it in during its own forward pass. There is no way to pass a
        # kwarg down -- torchtitan calls `model(inputs, **extra)` itself.
        # torchtitan increments `self.step` *before* calling train_step, so
        # inside this method `self.step` is already the step being run. An
        # earlier `self.step + 1` traced the wrong steps.
        tracing = (
            TRAJECTORY_STATE.trace_every > 0
            and self.step % TRAJECTORY_STATE.trace_every == 0
        )
        from src.pretrain.train_spec import PENDING_TRACE_SINK

        from src.model.loop_trace import ACTIVE_TRACE_SINK

        # Two sinks, one of which is the one that actually comes back. The
        # keyword path (PENDING_TRACE_SINK) is kept because it is what the
        # single-process code has always used; the module-level one is what
        # survives FSDP2's per-block kwarg repacking. See `_ActiveTraceSink`.
        sink: dict | None = {} if tracing else None
        PENDING_TRACE_SINK.sink = sink
        if tracing:
            ACTIVE_TRACE_SINK.arm()
        try:
            super().train_step(data_iterator)
        finally:
            PENDING_TRACE_SINK.sink = None
            collected = ACTIVE_TRACE_SINK.disarm()
        if tracing and collected:
            sink = collected

        self._emit_metrics(sink, requested=tracing)
        if self.step in TRAJECTORY_STATE.steps:
            self._save_trajectory_point()

    # -- metrics ------------------------------------------------------------

    @staticmethod
    def _is_rank_zero() -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _current_lr(self) -> float | None:
        """The learning rate actually in force on this step.

        READ from the optimizer's parameter group -- the value the step just
        used -- rather than recomputed from the schedule. A recomputation would
        agree with reality right up until they diverged (a resume whose
        scheduler state came back differently is the obvious way), and the log
        would then describe a run that did not happen.

        Every parameter group carries the same rate here: this project uses one
        schedule for the whole model, and torchtitan's own container assumes
        every scheduler in it is identical. If that ever stops being true,
        returning the first group's rate would be a half-truth, so this raises
        instead.

        Returns `None` when the optimizer is not built yet, which is not an
        error: the field is optional and a row without it is honest about not
        knowing.
        """
        optimizers = getattr(self, "optimizers", None)
        if optimizers is None:
            return None
        rates = {group["lr"] for optimizer in optimizers for group in optimizer.param_groups}
        if len(rates) > 1:
            raise ValueError(
                f"parameter groups have different learning rates ({sorted(rates)}); "
                "this project trains with one schedule for the whole model, so a single "
                "`lr` field would misdescribe the run. Log per-group rates instead."
            )
        return float(next(iter(rates))) if rates else None

    def _emit_metrics(self, sink: dict | None, *, requested: bool = False) -> None:
        collector = TRAJECTORY_STATE.collector
        # Take-and-clear: torchtitan only supplies a grad_norm on its own log
        # steps, so a value left lying around would be re-logged on the
        # following steps as if it had been measured there. Cleared before the
        # rank gate below so non-zero ranks cannot accumulate a stale one.
        grad_norm = self._pending_grad_norm
        self._pending_grad_norm = None
        # Rank 0 only. Every rank runs this method, and with all of them writing
        # to one path the log got N rows per step (interleaved, from N processes
        # appending to the same file) -- observed as 40 rows for a 20-step
        # 2-rank run. The loss is already reduced across ranks, so rank 0's view
        # is the run's view.
        if collector is None or not self._is_rank_zero():
            return

        from src.pretrain.train_spec import LAST_LOSS_COMPONENTS

        parts = LAST_LOSS_COMPONENTS.take()
        if parts is not None:
            task, lb, lz, total = parts
            collector.on_metrics_step(
                self.step,
                LossComponents(
                    task_loss=task,
                    lb_loss=lb,
                    lb_loss_factor=TRAJECTORY_STATE.config.lb_loss_factor,
                    z_loss=lz,
                    z_loss_factor=TRAJECTORY_STATE.config.lz_loss_factor,
                    total_loss=total,
                ),
                grad_norm=grad_norm,
                lr=self._current_lr(),
            )

        if requested and not sink and not self._warned_empty_trace:
            # KNOWN GAP: under FSDP2 the model does not observe
            # `PENDING_TRACE_SINK` during its forward, so router/residual traces
            # are not collected on the multi-GPU path. Verified: the same
            # mechanism works single-process, and `model_parts[0]` is
            # `FSDPLoopMoEModel`; neither a module attribute nor a module-level
            # global reaches the forward under the wrapper.
            #
            # Warn once instead of raising: the per-step loss components and the
            # trajectory checkpoints -- the irreplaceable outputs -- are
            # unaffected, and killing a multi-day run over a secondary metric
            # would be the worse failure. Loud once, in the log, so it cannot be
            # mistaken for "there was nothing to report".
            self._warned_empty_trace = True
            logger.warning(
                "step %s: router/residual trace requested but not collected under "
                "FSDP (model_parts[0]=%s). Loss components and trajectory "
                "checkpoints are unaffected; router/residual metrics will be "
                "absent from this run. This is a known gap, not a data problem.",
                self.step,
                type(self.model_parts[0]).__name__,
            )

        if sink:
            collector.on_trace_step(self.step, sink, TRAJECTORY_STATE.config.trace_meta)
            # Enforce the tensor-lifetime contract on the real training path, not
            # only in unit tests: a collector that retained these would pin whole
            # activation graphs and surface as an OOM tens of thousands of steps in.
            assert_trace_released(sink)

    # -- trajectory checkpoints --------------------------------------------

    def _save_trajectory_point(self) -> None:
        """Gather a full state dict and write it in HF format from rank 0.

        Under FSDP2 the parameters are DTensors sharded across ranks;
        `model.state_dict()` on any single rank returns shards. `get_model_state_dict`
        with `full_state_dict=True` performs the all-gather and (with
        `cpu_offload=True`) materialises real tensors on rank 0. Writing without
        that step would produce a directory of plausible-looking shards that
        `from_pretrained` accepts and silently misinterprets.
        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        model = self.model_parts[0]
        state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0
        if rank == 0:
            # Keys are used AS-IS. `LoopMoEModel` and `LoopMoEForCausalLM` both
            # hold the transformer as `self.model`, so their state_dict keys are
            # already identical -- an earlier version stripped a "model." prefix
            # "to match", which made every key miss on load. HuggingFace does not
            # fail on unmatched keys: it randomly initialises them and prints a
            # notice, so the checkpoints loaded, ran, and were pure noise.
            # `test_saved_keys_match_the_hf_class` pins this.
            path = TRAJECTORY_STATE.out_dir / "trajectory" / f"step_{self.step:08d}"
            config = TRAJECTORY_STATE.config
            tokens = self.step * self._global_batch_size() * self.job_config.training.seq_len
            # Resolved together so the recorded source cannot disagree with
            # the recorded commit. Prefers the job-start value: this process
            # may run for days against a clone whose HEAD moves under it.
            rev, rev_source = resolve_code_revision()
            save_trajectory(
                _HFView(config),
                path,
                state_dict=state_dict,
                # The gathered dict has no module to introspect, so the
                # precision-critical buffer names are passed explicitly.
                preserve_dtype_keys={n for n, _ in model.named_buffers()},
                manifest_fields=dict(
                    step=self.step,
                    tokens_consumed=tokens,
                    git_commit=rev["commit"],
                    code_dirty=rev["dirty"],
                    code_diff_sha256=rev["diff_sha256"],
                    code_revision_source=rev_source,
                    compute_dtype=str(self.job_config.training.mixed_precision_param),
                    storage_dtype="bfloat16",
                    autocast={"enabled": False},
                    torch_version=torch.__version__,
                    transformers_version=__import__("transformers").__version__,
                    model_config=config.to_dict(),
                    run_name=TRAJECTORY_STATE.manifest_run_name,
                    arch=TRAJECTORY_STATE.arch,
                    training_steps=TRAJECTORY_STATE.training_steps,
                    steps_overridden=TRAJECTORY_STATE.steps_overridden,
                    # Initialization uses torchtitan's live debug.seed; the
                    # dataloader's independent seed is in data_position.
                    seed=self.job_config.debug.seed,
                    # The schedule that actually ran, read off the live config
                    # for the same reason as `resume_checkpoint_interval_steps`
                    # below: recording the recipe's values would describe the
                    # run someone intended, not the one that happened.
                    lr_schedule={
                        "warmup_steps": self.job_config.lr_scheduler.warmup_steps,
                        "decay_ratio": self.job_config.lr_scheduler.decay_ratio,
                        "decay_type": str(self.job_config.lr_scheduler.decay_type),
                        "min_lr_factor": self.job_config.lr_scheduler.min_lr_factor,
                        "peak_lr": self.job_config.optimizer.lr,
                    },
                    data_position=self._data_position(),
                    rng_state_saved=False,  # trajectory points carry weights only
                    perturbation_probe={"runs_offline": True},
                    # Read off the LIVE config, never from
                    # `recipe.RESUME_CHECKPOINT_INTERVAL_STEPS`: the constant is
                    # a default suggestion that is overridden per submission
                    # (e.g. 2000 -> 500). Recording the constant would
                    # produce a manifest that describes the recipe rather than
                    # the job, which is worse than recording nothing.
                    resume_checkpoint_interval_steps=getattr(
                        self.job_config.checkpoint, "interval", None
                    ),
                    # Same reasoning: the live config, not the recipe's rule.
                    activation_checkpoint_mode=getattr(
                        self.job_config.activation_checkpoint, "mode", None
                    ),
                    # Which tokenizer produced the tokens this checkpoint was
                    # trained on, read from the DATA manifest the loader actually
                    # opened -- not from a launch flag, which states an intention.
                    # A checkpoint whose vocabulary is unrecorded cannot be
                    # evaluated safely: the two vocabularies in this project are
                    # 49,152 and 128,256, and loading a checkpoint against the
                    # wrong one produces text, not an error.
                    tokenizer_name=self._tokenizer_name(),
                ),
            )

        if is_distributed:
            # Keep ranks in step: rank 0 does real I/O here and the others must
            # not race ahead into the next training step's collectives.
            torch.distributed.barrier()

    def _global_batch_size(self) -> int:
        """Sequences per optimizer step, resolved.

        `training.global_batch_size` defaults to **-1**, meaning "derive it from
        local_batch_size x data-parallel degree". Reading the raw field produced a
        *negative* `tokens_consumed` in the manifest -- a value that is non-empty,
        passes the manifest contract, and is nonsense.
        """
        configured = self.job_config.training.global_batch_size
        if configured and configured > 0:
            return configured
        world = (
            torch.distributed.get_world_size()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        )
        derived = self.job_config.training.local_batch_size * world
        if derived <= 0:
            raise ValueError(
                f"cannot determine global batch size (configured={configured!r}, "
                f"local={self.job_config.training.local_batch_size!r}, world={world}); "
                "refusing to write a manifest with a bogus token count."
            )
        return derived

    def _data_position(self) -> dict[str, Any]:
        # `position()`, not `state_dict()`: the latter is DCP-shaped (rank-keyed,
        # pickled) and is not JSON-serialisable.
        try:
            reader = getattr(self.dataloader, "position", None)
            if callable(reader):
                return dict(reader())
            return dict(self.dataloader.state_dict())
        except Exception:  # pragma: no cover - never fail a run over provenance
            return {"unavailable": "dataloader position unavailable"}

    def _tokenizer_name(self) -> str | None:
        """The tokenizer named by the data manifest the loader is reading.

        Derived from the data, not from a launch flag: the flag says what someone
        meant to use, the manifest says what the bytes on disk were produced with.

        Matched against `src.data.tokenizer.TOKENIZER_CLASSES` so the recorded
        value is one of the names the rest of the project can resolve; an
        unrecognised repository records None rather than a guessed name, because
        a plausible-but-wrong vocabulary name is worse here than an absent one.
        """
        from src.data.tokenizer import TOKENIZER_CLASSES

        # The chain is loader -> dataset -> PackedSeqDataset.manifest. Walked
        # defensively rather than assumed: the first version of this reached for
        # `dataloader._ds` (the attribute on the *dataset*, not the loader), which
        # passed its unit tests against a stub and recorded None in every real
        # manifest -- the exact defect it was written to fix.
        try:
            node: Any = self.dataloader
            for _ in range(4):
                manifest = getattr(node, "manifest", None)
                if isinstance(manifest, dict):
                    repo = str(manifest.get("tokenizer_repo", ""))
                    break
                node = getattr(node, "_dataset", None) or getattr(node, "_ds", None)
                if node is None:
                    return None
            else:
                return None
        except Exception:  # pragma: no cover - never fail a run over provenance
            return None
        matches = [name for name in TOKENIZER_CLASSES if repo.rstrip("/").endswith(name)]
        return matches[0] if len(matches) == 1 else None


class _HFView:
    """Minimal stand-in exposing the `.config` that `save_trajectory` needs.

    The real module is sharded and already gathered into a plain dict, so only
    the config is required to write `config.json`.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def state_dict(self) -> dict[str, Any]:  # pragma: no cover - never reached
        raise RuntimeError("state_dict must be supplied explicitly for sharded models")

"""torchtitan integration for the looped-MoE model.

This is the *only* module in `src/pretrain/` that imports torchtitan.
`modeling_loop_lm.py` stays a plain HF-style `nn.Module`, so porting to another
trainer (e.g. Megatron-Core) means rewriting this file and
nothing else.

The model is registered from outside the torchtitan package via
`register_train_spec`, which torchtitan explicitly supports ("user can define a
TrainSpec from outside of torchtitan" -- its own source comment). There is no
fork of torchtitan anywhere in this project.

Pinned against torchtitan v0.2.2 (sha 73a0e69). Note that current torchtitan main
has replaced `TrainSpec` with a different `ModelSpec` API; if the pin is ever
moved forward, this file is what has to change.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.config import JobConfig
from torchtitan.protocols.model import BaseModelArgs, ModelProtocol
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

from src.model.loop_trace import LossComponents, TraceSink
from src.model.modeling_loop_lm import LoopedMoETransformer, LoopMoEConfig

__all__ = [
    "LoopMoEModelArgs",
    "LoopMoEModel",
    "LoopMoEDataLoader",
    "AuxLossState",
    "LAST_LOSS_COMPONENTS",
    "PENDING_TRACE_SINK",
    "build_loopmoe_loss_fn",
    "parallelize_loopmoe",
    "register_loopmoe",
]


# ---------------------------------------------------------------------------
# Aux-loss plumbing
# ---------------------------------------------------------------------------
#
# torchtitan's loss hook is `loss_fn(pred, labels)`: it gets the model's output
# and the labels, and no reference to the model. Our loss is not a pure function
# of (pred, labels) -- it also needs the router's load-balancing and z-losses,
# which are produced deep inside the forward.
#
# torchtitan's own MoE cannot be copied here: it uses aux-loss-*free* balancing
# (a bias term updated in an optimizer pre-hook), so it never needs to move an
# aux loss into the total loss. seedvar does use aux losses, so
# we have to carry them ourselves.
#
# The carrier is this small handoff object. `consume()` raises if nothing was
# published, so a wiring mistake fails loudly instead of silently training with
# no load-balancing pressure -- which would look like a perfectly healthy run
# right up until the expert-utilisation metrics came out wrong.


class AuxLossState:
    """Single-slot handoff for aux losses, from model forward to loss fn."""

    def __init__(self) -> None:
        self._pending: tuple[Tensor, Tensor] | None = None

    def publish(self, lb: Tensor, lz: Tensor) -> None:
        self._pending = (lb, lz)

    def consume(self) -> tuple[Tensor, Tensor]:
        if self._pending is None:
            raise RuntimeError(
                "aux losses were consumed twice, or the loss function ran without a "
                "forward pass having published them. Each forward must be followed by "
                "exactly one loss call."
            )
        pending, self._pending = self._pending, None
        return pending


AUX_LOSS_STATE = AuxLossState()
"""Process-wide handoff. One model per process, which is torchtitan's assumption."""


class _LastLossComponents:
    """Carries the step's loss breakdown from the loss fn to the trainer.

    Same reason as `AuxLossState`: torchtitan's loss hook returns a single scalar
    and the trainer never sees the parts. `take()` clears, so a step whose loss
    fn did not run cannot silently re-log the previous step's numbers.
    """

    def __init__(self) -> None:
        self._value: tuple[float, float, float, float] | None = None

    def put(self, task: float, lb: float, lz: float, total: float) -> None:
        self._value = (task, lb, lz, total)

    def take(self) -> tuple[float, float, float, float] | None:
        value, self._value = self._value, None
        return value


LAST_LOSS_COMPONENTS = _LastLossComponents()


class _PendingTraceSink:
    """Where the trainer parks the sink for the next forward.

    Module-level rather than an attribute on the model: FSDP2's `fully_shard`
    swaps the module's class (`LoopMoEModel` -> `FSDPLoopMoEModel`), and an
    attribute armed on the pre-wrap object did not reach the forward. Setting a
    module global sidesteps `nn.Module`'s attribute machinery and the wrapper
    classes entirely, so it cannot be defeated by whatever the trainer wraps the
    model in next.
    """

    def __init__(self) -> None:
        self.sink: dict | None = None


PENDING_TRACE_SINK = _PendingTraceSink()


# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------


@dataclass
class LoopMoEModelArgs(BaseModelArgs):
    """Thin carrier for `LoopMoEConfig`.

    Deliberately *not* a mirror of the 13 shape fields: duplicating them would
    create two sources of truth that could drift, and the drift would show up as
    a model whose config.json disagrees with its own weights. `LoopMoEConfig`
    remains the single definition.

    `BaseModelArgs` declares a field with a default, so dataclass rules force a
    default here too. `None` plus a raising `__post_init__` keeps the project's
    "no silent defaults" rule intact.
    """

    config: LoopMoEConfig | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            raise ValueError(
                "LoopMoEModelArgs requires an explicit LoopMoEConfig; there is no "
                "default architecture."
            )

    def update_from_config(self, job_config: JobConfig, **kwargs: Any) -> None:
        """Reconcile the architecture with the job's sequence length.

        Raises rather than silently truncating: a seq_len longer than the RoPE
        table would index out of bounds much later, and a shorter one would
        quietly train a model at a different context length than its config
        claims.
        """
        seq_len = job_config.training.seq_len
        if seq_len != self.config.context_length:
            raise ValueError(
                f"training.seq_len ({seq_len}) != model context_length "
                f"({self.config.context_length}). Set them equal explicitly; this "
                "project does not silently reconcile shape parameters."
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        """Return (unique parameter count, FLOPs per token).

        Two traps live here, both specific to looped MoE, and both of the
        "plausible number, wrong answer" kind that only surface as a bogus MFU:

        1. **Parameters are shared across loop steps, compute is not.** The stack
           is stored once but executed `num_stacks` times, so the parameter count
           uses the physical depth L while the FLOP count uses the unrolled depth
           R*L. (FLOPs are accounted on N_active.)
        2. **Only `num_active` of `num_experts` experts run per token.** Counting
           all E would overstate compute by E/k -- 4x at the DVF-c setting.
        """
        config = self.config
        d = config.d_model
        L, R = config.num_layers_in_stack, config.num_stacks
        E, k = config.num_experts, config.num_active
        d_ff_expert = config.d_ff // k

        # Per physical layer, parameters actually touched by one token.
        attn = 4 * d * d  # q, k, v, o
        router = d * E
        experts_active = k * 3 * d * d_ff_expert  # w1, w2, w3 for k experts
        active_per_layer = attn + router + experts_active

        # Compute sees the unrolled depth; storage does not.
        active_params = R * L * active_per_layer + d * config.vocab_size

        nparams_unique = sum(p.numel() for p in model.parameters())

        # 6*N for fwd+bwd through the weights, plus the attention score/context
        # matmuls, which scale with sequence length and are not weight FLOPs.
        head_dim = d // config.num_heads
        attn_flops = 12 * (R * L) * config.num_heads * head_dim * seq_len
        num_flops_per_token = 6 * active_params + attn_flops

        return nparams_unique, num_flops_per_token


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LoopMoEModel(ModelProtocol):
    """torchtitan-facing wrapper around `LoopedMoETransformer`.

    Owns no architecture of its own -- it adapts calling conventions only:
    torchtitan wants `forward(inputs) -> pred` and a separate `loss_fn`, while
    our model produces three tensors.
    """

    def __init__(self, model_args: LoopMoEModelArgs) -> None:
        super().__init__(model_args)
        self.model_args = model_args
        config = model_args.config
        self.config = config
        # One construction entry point (`from_config`): this site and the HF wrapper each
        # used to carry their own keyword list, and this one fell behind when head/tail
        # layers were added -- building a model that silently disagreed with its config.
        self.model = LoopedMoETransformer.from_config(config)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialise every parameter and buffer. **Not optional.**

        torchtitan builds the model under `torch.device("meta")`, then calls
        `to_empty(device=...)` and `init_weights()`. On meta the constructor's
        `trunc_normal_` calls do nothing, and `to_empty` hands back uninitialised
        storage -- so a model that relies on constructor-time init alone starts
        from garbage.

        The observed symptom is worth recording, because it does not look like a
        crash: training runs, checkpoints save, and the loss sits at exactly
        ln(vocab_size) forever with `grad_norm: 0.0000`. Weights near zero
        produce uniform logits, uniform logits produce no gradient signal, and
        the run looks healthy while learning nothing.

        Both parameters *and* buffers must be redone: `to_empty()` replaces
        buffer storage too, which would leave the RoPE rotation table as noise.
        """
        for module in self.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset) and module is not self:
                reset()

        if buffer_device is not None:
            for module in self.modules():
                rotations = getattr(module, "rotations", None)
                if isinstance(rotations, Tensor):
                    module.rotations = rotations.to(buffer_device)

    def forward(
        self,
        input_ids: Tensor,
        *,
        trace_sink: Optional[TraceSink] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Return logits; hand the aux losses to the loss function separately."""
        # torchtitan calls `model(inputs, **extra)` itself, so a trace cannot be
        # requested through the call. The trainer arms this attribute instead.
        if trace_sink is None:
            trace_sink = PENDING_TRACE_SINK.sink
        logits, lb, lz = self.model(input_ids, trace_sink=trace_sink)
        AUX_LOSS_STATE.publish(lb, lz)
        return logits


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def loopmoe_loss(
    pred: Tensor,
    labels: Tensor,
    *,
    lb_loss_factor: float,
    lz_loss_factor: float,
) -> Tensor:
    """Cross-entropy (summed) plus weighted aux losses.

    ``reduction="sum"`` is not a style choice -- torchtitan divides this by the
    global valid-token count to normalise across microbatches
    (``loss = loss_sum / global_valid_tokens``).

    That normalisation is the trap the aux terms have to survive. `lb` and `lz`
    are already per-token *means*; adding them directly to a summed CE would get
    them divided by the global token count too, shrinking their effective weight
    by ~10^5 and quietly training a model with essentially no load-balancing
    pressure. Scaling each by the local token count makes the microbatch
    contributions sum to exactly the intended mean-scale coefficient.
    """
    ce_sum = torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum"
    )
    lb, lz = AUX_LOSS_STATE.consume()
    n_tokens = labels.numel()
    aux_sum = (lb_loss_factor * lb + lz_loss_factor * lz) * n_tokens
    total = ce_sum + aux_sum

    # Hand the breakdown to the trainer. Per-token means, so the logged numbers
    # are comparable across batch sizes and to the toy harness.
    LAST_LOSS_COMPONENTS.put(
        task=(ce_sum / n_tokens).item(),
        lb=lb.item(),
        lz=lz.item(),
        total=(total / n_tokens).item(),
    )
    return total


def build_loopmoe_loss_fn(job_config: JobConfig, **kwargs: Any):
    """Build the loss closure. Coefficients come from the model config."""
    config = _CONFIG_REGISTRY_ACTIVE.get("config")
    if config is None:
        raise RuntimeError(
            "no active LoopMoEConfig; register_loopmoe() must run before the "
            "trainer builds the loss function."
        )
    lb_loss_factor = config.lb_loss_factor
    lz_loss_factor = config.lz_loss_factor

    def loss_fn(pred: Tensor, labels: Tensor) -> Tensor:
        return loopmoe_loss(
            pred, labels,
            lb_loss_factor=lb_loss_factor,
            lz_loss_factor=lz_loss_factor,
        )

    return loss_fn


def loss_components(
    total_loss: Tensor,
    task_loss: Tensor,
    lb: Tensor,
    lz: Tensor,
    config: LoopMoEConfig,
) -> LossComponents:
    """Package the breakdown for the metrics collector (logged every step)."""
    return LossComponents(
        task_loss=float(task_loss),
        lb_loss=float(lb),
        lb_loss_factor=config.lb_loss_factor,
        z_loss=float(lz),
        z_loss_factor=config.lz_loss_factor,
        total_loss=float(total_loss),
    )


_CONFIG_REGISTRY_ACTIVE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Parallelisation
# ---------------------------------------------------------------------------


def parallelize_loopmoe(
    model: nn.Module,
    parallel_dims: Any,
    job_config: JobConfig,
) -> nn.Module:
    """Apply activation checkpointing and FSDP2.

    Only data parallelism is wired. TP/PP/EP are deliberately absent: the target
    models are 0.6B-1.3B on a single 8-GPU node, and the config
    fields for the other axes exist but stay at 1.

    Activation checkpointing matters more here than in a normal model. LT2 ran
    T=4 with AC off, but DVF-c runs R=16: activation memory scales with the loop
    count, so it is ~4x the pressure LT2 measured. The switch is exercised in
    small tests rather than discovered at scale.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torchtitan.config import TORCH_DTYPE_MAP
    from torchtitan.distributed.activation_checkpoint import apply_ac

    if getattr(job_config.activation_checkpoint, "mode", "none") != "none":
        # AC is applied per stack layer: the stack is one module reused R times,
        # so checkpointing it saves memory on every loop step at once.
        #
        # Head and tail layers are NOT wrapped. They run once per forward, so
        # recomputing them buys one layer's worth of activations while costing a
        # second forward pass through it -- the trade that makes AC worth it for
        # the stack (R recomputations saved per wrap) does not hold here.
        apply_ac(model.model.stack, job_config.activation_checkpoint)

    if parallel_dims.fsdp_enabled:
        # v0.2.2 addresses meshes through `get_mesh(names)`, not by indexing
        # `world_mesh["dp_shard"]` -- that is the *current main* API, and using it
        # here raises `Invalid mesh_dim_names ('dp_shard',)` because the built
        # mesh exposes 'fsdp'/'batch'/'loss' instead.
        names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        mesh = parallel_dims.get_mesh(names)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )
        fsdp_config: dict[str, Any] = {"mesh": mesh, "mp_policy": mp_policy}

        # llama3's `apply_fsdp` cannot be reused: it reaches for `tok_embeddings`,
        # `layers`, `norm` and `output` by name, which are llama3's attribute
        # names, not ours. Sharding our own module tree is a dozen lines and
        # avoids contorting the model to match another architecture's layout.
        inner = model.model
        fully_shard(inner.token_embeddings, **fsdp_config)

        # Head layers run ONCE, before the loop, so each gets its own shard
        # group like any ordinary layer. They are deliberately not folded in
        # with the embedding or the stack: FSDP all-gathers a shard group as a
        # unit, and grouping a once-per-forward layer with a module called R
        # times would drag it through every loop step.
        for layer in inner.head_layers:
            fully_shard(layer, **fsdp_config)

        # Shard each physical block. There are L of them; each is *called R
        # times* per forward, so FSDP will all-gather and reshard the same
        # parameters once per loop step. That is correct but not free -- it is
        # the looped architecture's distinctive cost, and the reason
        # `reshard_after_forward` is worth revisiting at scale.
        for block in inner.stack.layers:
            fully_shard(block, **fsdp_config)

        # Tail layers: once each, same reasoning as the head layers above.
        for layer in inner.tail_layers:
            fully_shard(layer, **fsdp_config)

        fully_shard([inner.ln_final, inner.lm_head], **fsdp_config)
        fully_shard(model, **fsdp_config)
    return model


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_loopmoe(
    name: str,
    config: LoopMoEConfig,
    *,
    tokenized_dir: str,
    seed: int,
    verify_sha256: bool,
    on_boundary: str,
) -> None:
    """Register this architecture with torchtitan under `name`.

    Called from `config_registry.py` for each of baseline / DVF-a / DVF-b / DVF-c.
    The data parameters have no defaults: which shards a run consumed, and under
    which seed, is part of what makes the run reconstructible.
    """
    from torchtitan.components.lr_scheduler import build_lr_schedulers
    from torchtitan.components.optimizer import build_optimizers

    _CONFIG_REGISTRY_ACTIVE["config"] = config
    _CONFIG_REGISTRY_ACTIVE["data"] = {
        "tokenized_dir": tokenized_dir,
        "seed": seed,
        "verify_sha256": verify_sha256,
        "on_boundary": on_boundary,
    }

    register_train_spec(
        name,
        TrainSpec(
            model_cls=LoopMoEModel,
            model_args={"default": LoopMoEModelArgs(config=config)},
            parallelize_fn=parallelize_loopmoe,
            pipelining_fn=None,
            build_optimizers_fn=build_optimizers,
            build_lr_schedulers_fn=build_lr_schedulers,
            build_dataloader_fn=_build_dataloader,
            build_tokenizer_fn=None,
            build_loss_fn=build_loopmoe_loss_fn,
        ),
    )


class LoopMoEDataLoader(BaseDataLoader):
    """Adapts the bin/idx dataset to torchtitan's dataloader contract.

    Thin on purpose. DP sharding lives inside `PackedSeqIterableDataset` (each
    rank consumes `global_pos % dp_world_size == dp_rank`), so this must **not**
    re-shard; doing so would silently give every rank a subset of a subset.

    `state_dict`/`load_state_dict` delegate straight through, because the dataset
    is the only thing that knows the shard, byte offset and sample index a
    restart has to land on.
    """

    def __init__(self, dataset: Any, batch_size: int, dp_rank: int = 0) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        # State is PER-RANK, so it must be stored under a per-rank key.
        # DCP merges same-named non-tensor entries across ranks and keeps one
        # arbitrary winner: an unkeyed state_dict meant every rank resumed from
        # whichever rank happened to write last. Observed in a 2-rank run --
        # the checkpoint held `dp_rank: 1, global_sample_idx: 41`, so on resume
        # rank 0 adopted rank 1's position, both ranks then walked the same
        # interleaved stream, and half the data was never seen. The run
        # continued happily with a different data order, which is exactly the
        # resume/continuous loss divergence that was measured on the cluster.
        #
        # Mirrors torchtitan's own `ParallelAwareDataloader._rank_id`.
        self._rank_id = f"dp_rank_{dp_rank}"

    def __iter__(self):
        inputs: list[Tensor] = []
        labels: list[Tensor] = []
        for input_dict, label in iter(self._dataset):
            inputs.append(input_dict["input"])
            labels.append(label)
            if len(inputs) == self._batch_size:
                # torchtitan expects ({"input": ...}, labels) -- a 2-tuple, with
                # labels outside the dict. Anything else in the dict is forwarded
                # to the model as a kwarg and would be silently swallowed.
                yield {"input": torch.stack(inputs)}, torch.stack(labels)
                inputs, labels = [], []

    def position(self) -> dict[str, Any]:
        """Human-readable data position, for the checkpoint manifest.

        Deliberately separate from `state_dict()`: that one is shaped for DCP
        (rank-keyed and pickled to bytes), and feeding it to the manifest writer
        raises `Object of type bytes is not JSON serializable`. Provenance wants
        readable fields; DCP wants a restorable blob. Two consumers, two shapes.
        """
        return dict(self._dataset.state_dict())

    def state_dict(self) -> dict[str, Any]:
        return {self._rank_id: pickle.dumps(self._dataset.state_dict())}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self._rank_id not in state_dict:
            raise KeyError(
                f"checkpoint has no dataloader state for {self._rank_id} "
                f"(present: {sorted(state_dict)}). Resuming from another rank's "
                "position would silently change the data order; refusing."
            )
        self._dataset.load_state_dict(pickle.loads(state_dict[self._rank_id]))


def _build_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Any,
    job_config: JobConfig,
) -> BaseDataLoader:
    """Build the bin/idx loader for this rank.

    `tokenizer` is unused: the shards are pre-tokenised, and the tokenizer
    identity is pinned by sha256 in the dataset manifest rather than by whatever
    the trainer happens to hold.
    """
    from src.data.torchtitan_adapter import PackedSeqIterableDataset

    data_config = _CONFIG_REGISTRY_ACTIVE.get("data")
    if data_config is None:
        raise RuntimeError(
            "no active data config; register_loopmoe() must be called with the "
            "tokenized data directory before the trainer builds the dataloader."
        )

    dataset = PackedSeqIterableDataset(
        tokenized_dir=data_config["tokenized_dir"],
        seq_len=job_config.training.seq_len,
        seed=data_config["seed"],
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        verify_sha256=data_config["verify_sha256"],
        # What to do when training catches up to the tokenize pipeline.
        # No default: "wait" silently trades away reproducible data order, and
        # "raise" kills the job -- both are consequential enough to state.
        on_boundary=data_config["on_boundary"],
    )
    return LoopMoEDataLoader(dataset, job_config.training.local_batch_size, dp_rank=dp_rank)

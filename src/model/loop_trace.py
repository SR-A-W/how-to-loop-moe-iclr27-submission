"""Per-loop-step trace contract between the model and the metrics layer.

This module is the *interface* between `src.model` (which produces raw
tensors) and `src.metrics` (which turns them into numbers). It deliberately
imports **nothing** from torchtitan or from any trainer, so that:

  * the metrics layer can be developed and tested without a trainer, and
  * swapping the training backend touches nothing here.

Division of labour:

    the model / trainer  ->  *when* to collect and *how* to persist
    the metrics layer    ->  *what* to collect and *how* to compute it

Hence the model hands over **raw tensors, never reduced scalars**. A norm
computed in two places is two definitions of that norm; the project has already
paid for that mistake once (same-name metrics with several non-equivalent
definitions). The single definition lives in the metrics layer.

Tensor lifetime contract (IMPORTANT)
------------------------------------
`TracePoint` tensors are ``detach()``-ed **views of live activations**, not
copies. Passing them costs ~0 extra memory, which is why the model can afford to
hand over all R x L residual tensors instead of subsampling layers. The price is
a rule the consumer MUST obey:

  1. The collector callback is **synchronous**. When it returns, the tensors are
     considered dead.
  2. The collector MUST NOT store a TracePoint (or any tensor inside one) in any
     container that outlives the call -- no ``self.foo = point``, no appending to
     a list, no closure capture.
  3. Anything needed across steps must first be reduced to a Python scalar.

Violating this pins whole activation graphs in memory and turns a constant-memory
probe into a leak that grows with training length. `assert_trace_released` below
turns that failure into a loud test failure instead of a slow OOM at step 40k.
"""

from __future__ import annotations

import random
import weakref
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "ACTIVE_TRACE_SINK",
    "ROUTER_LOGITS_ARE_PRESOFTMAX",
    "LOOP_AXIS_PROVENANCE",
    "TracePoint",
    "LossComponents",
    "TraceMeta",
    "TraceSink",
    "LoopTraceCollector",
    "NullCollector",
    "run_probe_forward",
    "assert_trace_released",
]


# --- semantic markers -------------------------------------------------------
#
# The project has been burned by mislabelling sigmoid/logits,
# which produced results that "looked entirely reasonable but were all wrong".
# The collector schema has a `router_logits_are_presoftmax` field that pairs with
# this constant; it is a constant rather than a runtime flag because the model
# has exactly one behaviour and a flag that can only take one value is a lie
# waiting to happen.

ROUTER_LOGITS_ARE_PRESOFTMAX: bool = True
"""`TracePoint.router_logits` is the raw router linear output (pre-softmax)."""

LOOP_AXIS_PROVENANCE: str = "hook_call_index"
"""How `TracePoint.loop_step` was determined.

The value means the loop index was threaded down from the Python `for` loop that
drives the stack -- a per-module call counter. It was **not** recovered by
reshaping a flattened layer axis, which silently yields transposed semantics.

**The exact string matters.** It is not a description; it is a value the
diagnostics pipeline validates. The trace schema accepts only::

    allowed = ("hook_call_index", "hook_call_index+external")

and raises otherwise (schema.py:398-401), with `probe_*.py` scripts asserting the
same. An earlier draft of this constant read ``"explicit_call_counter"`` -- the
same claim in different words, and it would have made every diagnostics-pipeline
ingestion of our traces raise. Do not "improve" the wording.
"""


@dataclass(frozen=True)
class TracePoint:
    """Raw per-(loop_step, layer) tensors. See the lifetime contract above.

    All tensors are detached views; none require grad. Shapes use
    B=batch, S=sequence, E=num_experts, k=num_active, d=d_model.
    """

    block: str
    """Which part of the network produced this row: "head", "loop" or "tail".

    Head and tail are single MoE layers run once, outside the loop. `loop_step` and
    `layer_idx` describe a position inside the
    shared stack and are meaningless for them; they are recorded as 0/0 and must not be
    read as coordinates unless `block == "loop"`.
    """

    unrolled_pos: int
    """Position in the unrolled network, 0-based: THE depth coordinate.

    head = 0, the loop's layers run 1 .. num_stacks*num_layers_in_stack in execution
    order, tail = 1 + num_stacks*num_layers_in_stack (and correspondingly higher when a
    configuration has more than one head/tail layer). This is the sink's key, so it is
    also the only ordering that is guaranteed unique: sorting by (loop_step, layer_idx)
    puts head, tail and the loop's first layer on top of each other.
    """

    loop_step: int
    """Which pass through the shared stack, 0-based. Range: [0, num_stacks).

    Only meaningful when `block == "loop"`; 0 for head and tail.
    """

    layer_idx: int
    """Which physical layer inside the stack, 0-based. Range: [0, num_layers_in_stack).

    Only meaningful when `block == "loop"`; 0 for head and tail.
    """

    num_experts: int
    """E for THIS layer, taken from the router tensor's own shape.

    Per row, not per model, because head/tail layers keep 8 experts whatever the loop
    block uses (ablation recipe 2.1) -- S4 gives the loop block 64. A consumer that reads
    the model-level `TraceMeta.num_experts` and applies it to a head row computes, for
    example, L2 >= 1 - 8/64 = 0.875 and reads a perfectly healthy head layer as heavily
    collapsed. Nothing raises: the number is in range and looks plausible, and head/tail
    behaviour is exactly what the ablation is there to measure.
    """

    top_k: int
    """k for THIS layer, from the selection tensor's own shape. Same reasoning as above."""

    router_logits: Tensor
    """[B, S, E] raw router output, pre-softmax. See ROUTER_LOGITS_ARE_PRESOFTMAX."""

    router_probs: Tensor
    """[B, S, E] softmax over **all** E experts (not renormalised over top-k)."""

    topk_idx: Tensor
    """[B, S, k] indices of the selected experts."""

    topk_weights: Tensor
    """[B, S, k] combine weights: top-k of `router_probs`, renormalised to sum to 1.

    This is the quantity the MoE combine step already computes; exposing it adds
    no arithmetic.
    """

    residual: Tensor
    """[B, S, d] the block's output tensor, with **no reduction applied**.

    Semantically this is the diagnostics repo's KIND_RESIDUAL / CAPTURE_BLOCK_OUTPUT:
    the return value of the decoder block's forward. The metrics layer derives
    both the within-step and the step-boundary norms from this one tensor, so
    that both come from a single definition.
    """


@dataclass(frozen=True)
class LossComponents:
    """Per-step loss breakdown. Cheap enough to emit on *every* step.

    The aux losses are the run's health signal and are wanted every step. The
    *factors* are included alongside the values so that the logs alone show
    which coefficient was in force at any step.
    """

    task_loss: float
    """Cross-entropy on next-token prediction, before any aux term."""

    lb_loss: float
    """Switch-style load-balancing loss, E * sum_i f_i * p_i, averaged over all
    unrolled layers (num_stacks * num_layers_in_stack)."""

    lb_loss_factor: float
    """Coefficient multiplying `lb_loss` in `total_loss`."""

    z_loss: float
    """Router z-loss, mean((logsumexp logits)^2), averaged the same way."""

    z_loss_factor: float
    """Coefficient multiplying `z_loss` in `total_loss`."""

    total_loss: float
    """task_loss + lb_loss_factor * lb_loss + z_loss_factor * z_loss."""


@dataclass(frozen=True)
class TraceMeta:
    """Shape/provenance context so a collector never has to infer structure."""

    num_stacks: int
    """R: how many times the shared stack is called (loop count)."""

    num_layers_in_stack: int
    """L: physical layers per stack."""

    num_experts: int
    """E: experts per MoE layer."""

    num_active: int
    """k: experts activated per token."""

    router_logits_are_presoftmax: bool = ROUTER_LOGITS_ARE_PRESOFTMAX
    loop_axis_provenance: str = LOOP_AXIS_PROVENANCE


# A sink is just the dict the model fills in during a traced forward. The
# trainer creates one, passes it to the model, hands it to the collector, and
# drops it -- so no hook registration/deregistration dance, and no state living
# on the modules between steps.
TraceSink = dict[int, TracePoint]
"""Keyed by ``unrolled_pos`` -- explicit, never shape-derived.

It was keyed by ``(loop_step, layer_idx)`` until the head/tail layers arrived. Those
run outside the loop, so they have no meaningful value for
either coordinate; recording them as 0/0 -- which is what they are -- would have put
head, the loop's very first layer, and tail on the same key, and a dict assignment does
not complain. Two of the three rows would have vanished with nothing in the output
saying so. `unrolled_pos` is the depth coordinate used throughout, and keying on it
makes the key unique by construction rather than by a sentinel convention that a later
reader has to know about.
"""


class _ActiveTraceSink:
    """The sink the blocks write into, reached as a module-level object.

    Why this exists rather than passing the dict down as a keyword argument:
    `fully_shard` is applied per block (`src/pretrain/train_spec.py`), and FSDP2
    repacks kwargs across that boundary -- a dict passed by keyword arrives as a
    fresh copy at every sharded block. The blocks were writing faithfully into
    copies that were then discarded, so multi-GPU runs produced no router rows
    at all while every intermediate layer looked correct. Diagnosed
    by printing `id()` on both sides: identical single-process, different at
    every block under two ranks.

    A module-level object does not cross that boundary, which is exactly why
    `AUX_LOSS_STATE` never had the problem -- the aux losses travel up through
    return values and are published to a global. This is the same pattern.

    Process-local by construction: each rank has its own interpreter and so its
    own instance, which is what makes the collected values rank-local. That is a
    property to label in the output, not to hide -- see `aggregation_scope` in
    the router rows.
    """

    def __init__(self) -> None:
        self.sink: TraceSink | None = None

    def arm(self) -> TraceSink:
        """Start collecting; returns the dict that will be filled."""
        self.sink = {}
        return self.sink

    def disarm(self) -> TraceSink | None:
        """Stop collecting and hand back whatever was gathered."""
        sink, self.sink = self.sink, None
        return sink


ACTIVE_TRACE_SINK = _ActiveTraceSink()


@runtime_checkable
class LoopTraceCollector(Protocol):
    """What the metrics layer implements; what the training loop calls."""

    def on_metrics_step(
        self, step: int, loss: LossComponents, grad_norm: float | None = None
    ) -> dict[str, float]:
        """Called on **every** optimizer step. Returns flat scalars to log.

        `grad_norm` is passed only on the steps where the training loop
        already has it as a host float; on all other steps it is `None` and
        the field is omitted from the row rather than guessed at.
        """
        ...

    def on_trace_step(
        self, step: int, sink: TraceSink, meta: TraceMeta
    ) -> dict[str, float]:
        """Called every N steps with the raw tensors.

        MUST be synchronous and MUST NOT retain any tensor from `sink`
        (see the lifetime contract at the top of this module).
        """
        ...

    def on_probe_step(
        self, step: int, sink: TraceSink, meta: TraceMeta, per_token_loss: Tensor
    ) -> dict[str, float]:
        """Called every M steps with a forward over the fixed probe corpus.

        Deliberately a separate method rather than `on_trace_step` with a
        `source="probe"` flag: probe data answers different questions
        (cross-loop-step overlap, repetition-stratified loss) and lands in a
        different file, and a flag is something a downstream analysis can forget
        to filter on.

        `sink` is structurally identical to `on_trace_step`'s. `per_token_loss`
        is [B, S] and obeys the same no-retention contract.
        """
        ...


class NullCollector:
    """No-op collector, so training runs with metrics switched off."""

    def on_metrics_step(
        self, step: int, loss: LossComponents, grad_norm: float | None = None
    ) -> dict[str, float]:
        return {}

    def on_trace_step(
        self, step: int, sink: TraceSink, meta: TraceMeta
    ) -> dict[str, float]:
        return {}

    def on_probe_step(
        self, step: int, sink: TraceSink, meta: TraceMeta, per_token_loss: Tensor
    ) -> dict[str, float]:
        return {}


def _snapshot_rng() -> dict[str, Any]:
    """Capture every RNG stream a probe forward could disturb."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_probe_forward(
    model: Any,
    input_ids: Tensor,
    *,
    collector: LoopTraceCollector,
    step: int,
) -> dict[str, float]:
    """Run the fixed probe corpus through the model and hand it to the collector.

    Three isolation properties, each of which exists because violating it would
    corrupt something silently:

    1. **No gradients, eval mode.** LT2 (arXiv 2605.20670) disabled probing outright because
       activation memory multiplies by the loop count; at R=16
       this is 4x the pressure they measured, so a probe that built a graph would
       OOM at exactly the configurations we most want to measure.
    2. **RNG is restored on exit** (python / numpy / torch / cuda). Without this,
       whether a probe ran would change the training trajectory, and a resumed
       run would silently diverge from an uninterrupted one -- breaking the
       requirement that the two be statistically indistinguishable.
    3. **The training dataloader is never touched.** The probe corpus is a fixed
       tensor held separately, so probing does not advance the data position that
       checkpoints record.

    The model's training/eval mode is restored even if the collector raises.

    Returns whatever scalars the collector produced.
    """
    was_training = model.training
    rng_state = _snapshot_rng()
    sink: TraceSink = {}
    try:
        model.eval()
        with torch.no_grad():
            out = model(input_ids, trace_sink=sink)
            logits = out.logits if hasattr(out, "logits") else out

            # Next-token targets. The model does not shift internally (the
            # dataloader supplies pre-shifted labels during training), so the
            # shift is explicit here. The final position has no next token and is
            # masked out; cross_entropy returns 0.0 there.
            labels = input_ids.new_full(input_ids.shape, -100)
            labels[:, :-1] = input_ids[:, 1:]
            per_token_loss = F.cross_entropy(
                logits.transpose(1, 2).float(), labels,
                reduction="none", ignore_index=-100,
            )  # [B, S]

            meta = model.config.trace_meta
            return collector.on_probe_step(step, sink, meta, per_token_loss)
    finally:
        sink.clear()
        _restore_rng(rng_state)
        if was_training:
            model.train()


def assert_trace_released(sink: TraceSink) -> None:
    """Raise if a collector retained tensors from `sink` past its callback.

    Call this immediately after the collector returns and after clearing local
    references. It weak-references every tensor, drops the sink, and checks that
    nothing kept them alive.

    This is the enforcement half of the lifetime contract. It is meant to be
    wired into the toy/CI run rather than the production hot path.
    """
    refs: list[weakref.ref] = []
    for point in sink.values():
        for tensor in (
            point.router_logits,
            point.router_probs,
            point.topk_idx,
            point.topk_weights,
            point.residual,
        ):
            try:
                refs.append(weakref.ref(tensor))
            except TypeError:  # pragma: no cover - torch tensors are weakref-able
                continue
    # After a `for` loop the loop variables stay bound in this frame, so `point`
    # and `tensor` would still reference the *last* TracePoint when we collect --
    # and this function would report itself as the leaker. Drop them explicitly.
    point = tensor = None  # noqa: F841 - rebinding to release references
    sink.clear()

    import gc

    gc.collect()
    leaked = sum(1 for ref in refs if ref() is not None)
    if leaked:
        raise RuntimeError(
            f"{leaked}/{len(refs)} trace tensors are still alive after the collector "
            "returned. A collector must not store TracePoint tensors beyond the "
            "callback; reduce to scalars first. See the lifetime contract in "
            "src/model/loop_trace.py."
        )

def unrolled_position(loop_step: int, layer_idx: int, num_layers_in_stack: int) -> int:
    """Depth coordinate of a loop-block layer: `loop_step * n_layers + layer_idx`.

    THE definition of that arithmetic. It is needed in two places -- the legacy
    checkpoint adapter in the cross-architecture probe, and the `loop_only`
    branch of `cross_arch_v2_offline` that reads pre-head/tail artifacts -- and
    the two must agree, because one writes the coordinate and the other reads it
    back. Two copies of a formula that indexes into a dump do not fail loudly
    when they drift; they silently address different cells.

    Only meaningful for `block == "loop"`. Head and tail run once and are their
    own physical layers, so their position comes from the network's order, not
    from this arithmetic.
    """
    if num_layers_in_stack <= 0:
        raise ValueError(
            f"num_layers_in_stack must be positive, got {num_layers_in_stack}. "
            "A zero or negative stack size collapses every pass onto the same "
            "position, which reads as a valid coordinate."
        )
    return loop_step * num_layers_in_stack + layer_idx


def legacy_trace_point_factory(num_layers_in_stack: int):
    """A `TracePoint` stand-in accepting the PRE-head/tail keyword set.

    Checkpoints trained before the head/tail change bundle their own
    `modeling_loop_lm.py`, which constructs `TracePoint(loop_step=..., layer_idx=...,
    <tensors>)`. Those four fields are now required, so such a checkpoint cannot
    be loaded at all -- the failure is a `TypeError` inside the bundled code,
    raised before any forward pass completes.

    The four fields are filled rather than defaulted, and deliberately NOT given
    defaults on `TracePoint` itself: `num_experts` being required per cell is
    what stops a head layer's balanced 8 experts being divided by the loop
    block's 64, which would put a floor of 0.875 under `L2` and read as severe
    collapse in every configuration. That protection is worth keeping for new
    checkpoints even though old ones need this adapter.

    A pre-head/tail model is all loop block by construction, so `block` is
    `"loop"` for every row and the depth coordinate is the unrolled arithmetic.
    """

    # Bound HERE, not looked up inside `make`. The adapter is installed by
    # REPLACING the module-level `TracePoint` name, so a lookup at call time
    # would find this factory instead of the class -- the adapter would call
    # itself. Capturing the class before installation is what makes the
    # substitution safe.
    cls = TracePoint

    def make(**kwargs: Any) -> "TracePoint":
        logits = kwargs.get("router_logits")
        topk_idx = kwargs.get("topk_idx")
        if logits is None or topk_idx is None:
            raise ValueError(
                "legacy TracePoint adapter needs router_logits and topk_idx to "
                f"infer num_experts and top_k; got keys {sorted(kwargs)}"
            )
        return cls(
            block="loop",
            unrolled_pos=unrolled_position(
                int(kwargs["loop_step"]), int(kwargs["layer_idx"]), num_layers_in_stack
            ),
            # Read off the tensors, which are the only source that exists here.
            num_experts=int(logits.shape[-1]),
            top_k=int(topk_idx.shape[-1]),
            **kwargs,
        )

    return make

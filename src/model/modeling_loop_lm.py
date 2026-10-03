"""Looped-MoE modeling code (port of the seedvar `loop-lm` architecture).

Provenance
----------
Ported from ``modeling_loop_lm.py`` as published with
``ml-ryanlee/seedvar-looped-moe-1e18-d704-seed42..47`` (arXiv 2605.09165,
*Sparse Layers are Critical to Scaling Looped Language Models*). That file is the
architecture specification: the published checkpoints are its training product,
and the diagnostics pipeline is already validated against it.

This is a **port, not a copy**. The numerics are kept faithful (see "Faithful to
the original" below) because the baseline's whole job is to reproduce the
published architecture; the deviations are all in service of three project
requirements that the original file does not meet:

  1. **Semantic parameters have no defaults.** The original ``LoopLMConfig``
     defaults ``d_model=1024``, ``num_experts=8`` and so on. Defaults are how a
     silently-wrong run happens: a typo'd key name falls back to a plausible
     number and the run looks fine. Every shape/semantic field here is required
     and a missing one raises.
  2. **Every loop step is hookable, by explicit index.** The loop counter is
     threaded down to each block, so a consumer never recovers the loop axis by
     reshaping a flattened layer axis (which silently yields transposed
     semantics).
  3. **Router logits are exposed with their semantics labelled.** Both the
     pre-softmax logits and the post-softmax probabilities are handed out, tagged
     via ``src.model.loop_trace.ROUTER_LOGITS_ARE_PRESOFTMAX``.

Scope
-----
Only the ``looped-moe`` variant is implemented. The original file carries four
variants (base / looped / moe / looped-moe); all four architectures this project
trains -- baseline and dvf_a/b/c -- are looped-moe, differing only in the (L, R, E)
triple. Porting the unused three would be dead code (project rule: no entities
beyond necessity). They remain available in the original file if ever needed.

Shape parameters, and what the experiment varies
------------------------------------------------
====================  ======  =========================================
config field          symbol  meaning
====================  ======  =========================================
num_layers_in_stack   L       physical layers in the shared stack
num_stacks            R       times the stack is called (loop count)
num_experts           E       experts per MoE layer
num_active            k       experts activated per token
====================  ======  =========================================

The fixed-budget series (config names dvf_a/b/c) holds L*R = 16 and E*L = 64 fixed
and only moves capacity around: baseline (8,2,8), dvf_a (4,4,16), dvf_b (2,8,32),
dvf_c (1,16,64).

Faithful to the original (do not "fix" these -- they are muP, not bugs)
----------------------------------------------------------------------
* ``RMSNorm`` has **no** gain parameter.
* Attention is scaled by ``1/d_k``, not ``1/sqrt(d_k)``.
* Softmax upcasts to float32.
* Expert FFN width is ``d_ff // num_active`` -- divided by k, *not* by E. This is
  what makes E*L=64 hold total expert parameters constant across the fixed-budget series.
* Initialisation is muP: ``std = std_base / sqrt(width_ratio)`` with
  ``std_base = sqrt(2/(fan_in_base + fan_out_base))`` against a d_base=128 proxy.

Deliberate deviation
--------------------
``RotaryPositionalEmbedding`` builds its rotation table with vectorised torch ops
instead of the original's ``max_seq_len * d_k/2`` nested Python loop, which costs
minutes at seq_len 4096. The vectorised table was checked against a literal
transcription of the original loop.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, reduce, repeat
from torch import Tensor
from torch.nn.functional import grouped_mm, silu
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

# This file is loaded in three different ways, and the trace types have to
# resolve in all of them:
#   1. as part of this repo             -> `src.model.loop_trace`
#   2. via HuggingFace trust_remote_code -> copied into a generated package under
#      `transformers_modules/<ckpt>/`, where the sibling is a *relative* import
#   3. as a loose script with the checkpoint directory on sys.path
# Case 2 is the one that matters most: the diagnostics pipeline is a
# separate repository that has never heard of `pretrain`, and `save_trajectory`
# bundles `loop_trace.py` next to this file so the checkpoint stands alone.
try:
    from src.model.loop_trace import ACTIVE_TRACE_SINK, TraceMeta, TracePoint, TraceSink
except ImportError:  # pragma: no cover - covered by the cold-load test
    try:
        from .loop_trace import ACTIVE_TRACE_SINK, TraceMeta, TracePoint, TraceSink  # type: ignore[no-redef]
    except ImportError:
        from loop_trace import ACTIVE_TRACE_SINK, TraceMeta, TracePoint, TraceSink  # type: ignore[no-redef]

__all__ = ["LoopMoEConfig", "LoopMoEForCausalLM", "LoopedMoETransformer"]

# muP proxy-model widths. The initialisation std of every weight is derived from
# a d_base=128 model and rescaled by width_ratio = d_model / 128.
HEAD_TAIL_NUM_EXPERTS = 8
"""Experts in a head/tail layer, fixed across every ablation configuration.

Head and tail are identical in all configurations (1 layer, 8 experts, top-2).
It is deliberately independent of the loop block's E --
S4 gives the loop 64 experts per layer and its head still has 8 -- so the loop block's
count must not be reused here.
"""

BASE_D_MODEL = 128
BASE_D_FF = 384


def softmax(logits: Tensor, dim: int) -> Tensor:
    """Max-shifted softmax in float32 (verbatim semantics from the original)."""
    logits = logits.float()
    max_values = torch.max(logits, dim=dim, keepdim=True).values
    shifted = logits - max_values
    shifted_exps = torch.exp(shifted)
    shifted_exp_sums = torch.sum(shifted_exps, dim=dim, keepdim=True)
    return shifted_exps / shifted_exp_sums


class Linear(nn.Module):
    """Bias-free linear layer with muP initialisation."""

    def __init__(self, in_features, out_features, width_ratio, std_base, device=None, dtype=None):
        super().__init__()
        # Registered before init so the shape exists under HF meta-device loading.
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        # Kept so the init can be replayed: torchtitan builds the model on the
        # meta device and then calls `init_weights()` on materialised (but
        # uninitialised) storage, so constructor-time init alone leaves the
        # model full of garbage.
        self._init_std = std_base / math.sqrt(width_ratio)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self._init_std
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMS norm **without** a gain parameter (muP convention)."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_squared_sum = (1 / self.d_model) * einsum(x, x, "... seq d, ... seq d -> ... seq")
        rms = torch.sqrt(mean_squared_sum + self.eps)
        rms_norm = einsum(x, 1 / rms, "... seq d, ... seq -> ... seq d")
        return rms_norm.to(in_dtype)


class PositionwiseFeedforward(nn.Module):
    """SwiGLU: W2(SiLU(W1 x) * W3 x)."""

    def __init__(self, d_model: int, d_ff: int, width_ratio: float, device=None, dtype=None):
        super().__init__()
        w_std_base = math.sqrt(2 / (BASE_D_MODEL + BASE_D_FF))
        self.w1 = Linear(d_model, d_ff, width_ratio, w_std_base, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, width_ratio, w_std_base, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, width_ratio, w_std_base, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE with a precomputed [seq, d_k/2, 2, 2] rotation table.

    Vectorised rebuild of the original's nested Python loop; see the module
    docstring.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        # Retained so the table can be rebuilt: `to_empty()` replaces buffer
        # storage with uninitialised memory just as it does for parameters, so a
        # meta-device build leaves the rotation table as garbage unless
        # `reset_parameters()` regenerates it.
        self._rope_theta, self._rope_d_k = theta, d_k
        self._rope_max_seq_len, self._rope_dtype = max_seq_len, dtype
        rotations = self._build_table(theta, d_k, max_seq_len, device, dtype)
        self.register_buffer("rotations", rotations, persistent=True)

    @staticmethod
    def _build_table(theta: float, d_k: int, max_seq_len: int, device, dtype) -> Tensor:
        """[seq, d_k/2, 2, 2] rotation table.

        Angles are built in float64 and only then cast down. At seq_len 4096 the
        largest angle is ~4096 rad, where float32 spacing is ~2.4e-4; computing
        cos/sin at float32 there loses ~4 decimal digits. The original does this
        implicitly (Python floats are float64), so float64 here is both more
        accurate and what keeps the table equal to the reference.
        """
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float64)
        pair_idx = torch.arange(d_k // 2, device=device, dtype=torch.float64)
        inv_freq = theta ** (2 * pair_idx / d_k)
        angles = positions[:, None] / inv_freq[None, :]
        cos, sin = torch.cos(angles), torch.sin(angles)
        # rows of the 2x2 rotation: [[cos, -sin], [sin, cos]]
        table = torch.stack(
            [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2
        )
        return table.to(dtype if dtype is not None else torch.float32)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Regenerate the rotation table in place (buffers survive nothing)."""
        self.rotations.copy_(
            self._build_table(
                self._rope_theta, self._rope_d_k, self._rope_max_seq_len,
                self.rotations.device, self.rotations.dtype,
            )
        )

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        rot = self.rotations[token_positions].to(dtype=x.dtype)
        x_pairs = rearrange(x, "... seq_dim (feature_dim i) -> ... seq_dim feature_dim i", i=2)
        y_pairs = einsum(
            rot,
            x_pairs,
            "... seq_dim feature_dim i j, ... seq_dim feature_dim j -> ... seq_dim feature_dim i",
        )
        return rearrange(y_pairs, "... seq_dim feature_dim i -> ... seq_dim (feature_dim i)")


class MultiheadSelfAttention(nn.Module):
    """Causal MHSA with RoPE. muP: attention logits scaled by 1/d_k."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float,
                 width_ratio: float, device=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads

        attn_std_base = math.sqrt(2 / (BASE_D_MODEL + BASE_D_MODEL))
        self.q_proj = Linear(d_model, d_model, width_ratio, attn_std_base, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, width_ratio, attn_std_base, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, width_ratio, attn_std_base, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, width_ratio, attn_std_base, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device, dtype)

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        d_k = self.d_model // self.num_heads
        q_heads = rearrange(self.q_proj(x), "... seq (heads d_k) -> ... heads seq d_k", d_k=d_k)
        k_heads = rearrange(self.k_proj(x), "... seq (heads d_k) -> ... heads seq d_k", d_k=d_k)
        v_heads = rearrange(self.v_proj(x), "... seq (heads d_v) -> ... heads seq d_v", d_v=d_k)

        if token_positions is None:
            token_positions = rearrange(torch.arange(x.shape[-2], device=x.device), "seq -> 1 seq")
        q_heads = self.rope(q_heads, token_positions)
        k_heads = self.rope(k_heads, token_positions)

        mha_heads = F.scaled_dot_product_attention(
            q_heads, k_heads, v_heads, is_causal=True, scale=1.0 / d_k
        )
        return self.output_proj(rearrange(mha_heads, "... heads seq d_v -> ... seq (heads d_v)"))


class Router(nn.Module):
    """Top-k softmax router. Returns pre-softmax logits *and* probabilities.

    The two are returned side by side, and the caller labels which is which via
    ``ROUTER_LOGITS_ARE_PRESOFTMAX``. There is no jitter noise and no temperature;
    routing is deterministic given the input (matching the original).
    """

    def __init__(self, d_model: int, num_experts: int, num_active: int, width_ratio: float,
                 device=None, dtype=None):
        super().__init__()
        std_base = math.sqrt(2 / (BASE_D_MODEL + num_experts))
        self.gate = Linear(d_model, num_experts, width_ratio, std_base, device=device, dtype=dtype)
        self.num_active = num_active

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits = self.gate(x)  # [B, S, E] -- pre-softmax
        probs = softmax(logits, dim=-1)  # [B, S, E] -- over all E experts
        top_scores, top_experts = torch.topk(probs, k=self.num_active, dim=-1)
        # Renormalise within the selected set so the combine weights sum to 1.
        top_scores = top_scores / torch.sum(top_scores, dim=-1, keepdim=True)
        return logits, probs, top_scores, top_experts


class GroupedMoEPrenormBlock(nn.Module):
    """Pre-norm block whose FFN is a grouped top-k MoE.

    Layout: x -> +attn(ln1(x)) -> +moe(ln2(.)). Aux losses are returned rather
    than stashed on the module, so nothing has to be reset between loop steps.
    """

    @staticmethod
    def _init_expert_weights(num_experts, in_features, out_features, width_ratio, std_base,
                             device, dtype) -> nn.Parameter:
        w = torch.empty(num_experts, in_features, out_features, device=device, dtype=dtype)
        std_scaled = std_base / math.sqrt(width_ratio)
        nn.init.trunc_normal_(w, mean=0.0, std=std_scaled, a=-3 * std_scaled, b=3 * std_scaled)
        return nn.Parameter(w)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Re-init the grouped expert weights (see Linear.reset_parameters)."""
        std = self._expert_init_std
        for w in (self.experts_w1, self.experts_w2, self.experts_w3):
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_experts: int, num_active: int,
                 max_seq_len: int, theta: float, width_ratio: float, device=None, dtype=None,
                 num_attention_sets: int = 1):
        """`num_attention_sets` = how many independent attention parameter sets
        this block holds, one per pass over the looped stack.

        1 is the shared-attention architecture every non-per-pass run uses.
        It keeps `self.attn` a single module, so the state dict key stays
        `attn.q_proj.weight` and existing checkpoints load unchanged -- a
        ModuleList of one would rename every key to `attn.0.*` and silently
        invalidate every archive we hold.

        >1 makes `self.attn` a ModuleList of that many attention modules, each
        initialised independently under the same muP standard deviation, chosen
        at forward time by `loop_step`. Everything else in the block -- norms,
        router, experts -- remains shared across passes, which is the point of
        the comparison.
        """
        super().__init__()
        if num_attention_sets < 1:
            raise ValueError(f"num_attention_sets must be >= 1, got {num_attention_sets}")
        self.num_attention_sets = num_attention_sets
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        _make_attn = lambda: MultiheadSelfAttention(  # noqa: E731 -- one expression, used twice
            d_model, num_heads, max_seq_len, theta, width_ratio, device, dtype
        )
        self.attn = (
            _make_attn() if num_attention_sets == 1
            else nn.ModuleList([_make_attn() for _ in range(num_attention_sets)])
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.router = Router(d_model, num_experts, num_active, width_ratio, device=device, dtype=dtype)

        self.num_experts = num_experts
        self.num_active = num_active

        # NOTE: divided by num_active (k), not by num_experts (E). This is what
        # keeps total expert parameters constant across the fixed-budget series.
        d_ff_expert = d_ff // num_active
        w_std_base = math.sqrt(2 / (BASE_D_MODEL + BASE_D_FF))
        self._expert_init_std = w_std_base / math.sqrt(width_ratio)
        self.experts_w1 = self._init_expert_weights(num_experts, d_model, d_ff_expert, width_ratio, w_std_base, device, dtype)
        self.experts_w2 = self._init_expert_weights(num_experts, d_ff_expert, d_model, width_ratio, w_std_base, device, dtype)
        self.experts_w3 = self._init_expert_weights(num_experts, d_model, d_ff_expert, width_ratio, w_std_base, device, dtype)

    def attention_for(self, loop_step: Optional[int]):
        """The attention module this pass uses.

        `loop_step` is the explicit loop variable of
        `LoopedMoETransformer.forward` -- the index of the current pass over the
        shared stack -- threaded down as an argument. This block keeps **no
        cross-call state**: nothing here remembers which pass ran last, and two
        calls with the same `loop_step` select the same module. That is why the
        looped model can be traced, resumed and re-entered in any order.

        There is deliberately **no loop-step embedding**: the passes differ only
        by their attention parameters, which
        is what keeps U1-U4 a controlled comparison against S1-S4.

        Refuses rather than defaults when `loop_step` is missing on a per-pass
        block: falling back to set 0 would run every pass through the first
        pass's attention, which trains, converges and reports nothing unusual
        while being a different architecture from the one on the config.
        """
        if self.num_attention_sets == 1:
            return self.attn
        assert self.num_attention_sets == len(self.attn), (
            f"{self.num_attention_sets} sets declared but {len(self.attn)} modules held; "
            "the count and the container have drifted apart"
        )
        if loop_step is None:
            raise ValueError(
                "this block has per-pass attention "
                f"({self.num_attention_sets} sets) and needs loop_step to choose one; "
                "got None. Every caller in this file passes it -- a new caller must too."
            )
        if not 0 <= loop_step < len(self.attn):
            raise IndexError(
                f"loop_step {loop_step} is outside the {self.num_attention_sets} attention "
                "sets this block holds. The block is built with one set per pass, so a "
                "loop_step beyond that means the model and the config disagree about R."
            )
        return self.attn[loop_step]

    def forward(
        self,
        x: Tensor,
        token_positions: Optional[Tensor] = None,
        *,
        loop_step: Optional[int] = None,
        layer_idx: Optional[int] = None,
        block: Optional[str] = None,
        unrolled_pos: Optional[int] = None,
        trace_sink: Optional[TraceSink] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq, dim = x.shape
        total_tokens = batch * seq

        norm1_out = self.ln1(x)
        attn_out = self.attention_for(loop_step)(norm1_out, token_positions)
        assert x.shape == attn_out.shape
        resid1_out = attn_out + x

        norm2_out = self.ln2(resid1_out)
        logits, probs, top_scores, top_experts = self.router(norm2_out)

        # `softmax` computes in float32 and does not cast back, so `top_scores`
        # is float32 regardless of the activation dtype. The combine weights get
        # multiplied into bf16 expert outputs below, so they must match, or
        # einsum raises "expected m1 and m2 to have the same dtype".
        #
        # This is invisible in the original: seedvar's published checkpoints are
        # float32, where the cast is a no-op. Under the bf16 training this
        # project uses (LT2, arXiv 2605.20670, runs pure bf16) it is a hard failure
        # on the first forward.
        #
        # Only the combine weights are cast. `probs` and `logits` stay float32
        # for the aux-loss and z-loss reductions, which is where the extra
        # precision is worth having.
        top_scores = top_scores.to(x.dtype)

        # Flatten and sort by expert so grouped_mm can run one matmul per expert.
        x_flat = rearrange(norm2_out, "b s d -> (b s) d")
        flat_expert_ids = rearrange(top_experts, "b s k -> (b s k)")
        flat_scores = rearrange(top_scores, "b s k -> (b s k)")
        flat_positions = torch.arange(total_tokens, device=x.device)
        flat_token_ids = repeat(flat_positions, "n -> (n k)", k=self.num_active)

        sort_indices = flat_expert_ids.argsort(stable=True)
        sorted_expert_ids = flat_expert_ids[sort_indices]
        sorted_token_ids = flat_token_ids[sort_indices]
        sorted_scores = flat_scores[sort_indices]
        sorted_x = x_flat[sorted_token_ids]

        counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        offs = counts.cumsum(0).to(torch.int32)

        h1 = grouped_mm(sorted_x, self.experts_w1, offs=offs)
        h3 = grouped_mm(sorted_x, self.experts_w3, offs=offs)
        gated = silu(h1) * h3
        expert_out = grouped_mm(gated, self.experts_w2, offs=offs)

        expert_out = einsum(expert_out, sorted_scores, "n d, n -> n d")
        output_flat = torch.zeros(total_tokens, dim, device=x.device, dtype=expert_out.dtype)
        output_flat.index_add_(0, sorted_token_ids, expert_out)
        experts_out = rearrange(output_flat, "(b s) d -> b s d", b=batch, s=seq)

        # Aux losses:
        #   L_LB = E * sum_i f_i * p_i   (switch-style load balancing)
        #   L_RZ = mean( (logsumexp logits)^2 )   (router z-loss)
        # Both are computed per layer per loop step; the caller averages over the
        # unrolled depth (num_stacks * num_layers_in_stack).
        fi = counts.float() / (total_tokens * self.num_active)
        pi = reduce(probs, "b s e -> e", "mean")
        lb = self.num_experts * einsum(fi, pi, "e, e ->")

        logsumexp = torch.logsumexp(logits.float(), dim=-1)
        lz = reduce(logsumexp**2, "... -> ", "mean")

        assert experts_out.shape == resid1_out.shape
        final_out = resid1_out + experts_out

        # Write into every sink that is armed. Under FSDP2 the `trace_sink`
        # keyword arrives as a per-block COPY (see `_ActiveTraceSink`), so the
        # module-level one is the only sink the trainer can actually read back;
        # the keyword remains for callers that pass their own dict directly
        # (the toy launcher, the offline probes), where it is the same object.
        # Writing to both is harmless: each is first-write-wins.
        sinks = [s for s in (ACTIVE_TRACE_SINK.sink, trace_sink) if s is not None]
        if sinks:
            if loop_step is None or layer_idx is None or block is None or unrolled_pos is None:
                raise ValueError(
                    "trace_sink was provided but loop_step/layer_idx/block/unrolled_pos "
                    "were not. Both axes must be explicit counters, never inferred."
                )
            if block not in ("head", "loop", "tail"):
                raise ValueError(f"block must be head/loop/tail, got {block!r}")
            # Keyed by the depth coordinate: head, the loop's first layer and tail all
            # carry loop_step=layer_idx=0, so the old pair-key silently collapsed them.
            key = unrolled_pos
            # First write wins. Under selective activation checkpointing the
            # block's forward runs a second time during backward to recompute
            # activations, so every key is legitimately visited twice -- that is
            # how AC works, not a bug. The recomputed values are identical by
            # construction, so keeping the first and ignoring the rest is both
            # correct and cheap.
            #
            # An earlier version raised on the second visit. That guard was aimed
            # at double-*counting*, which first-write-wins prevents directly; as
            # written it instead killed every AC-enabled run at the first traced
            # step. The property that actually matters: same keys, same values,
            # with AC on.
            # Detached views, not copies -- see the lifetime contract in
            # src/model/loop_trace.py.
            point = TracePoint(
                block=block,
                unrolled_pos=unrolled_pos,
                # From the tensors themselves, never from the model config: this layer's
                # E and k are what produced these numbers, and head/tail differ from the
                # loop block.
                num_experts=int(logits.shape[-1]),
                top_k=int(top_experts.shape[-1]),
                loop_step=loop_step,
                layer_idx=layer_idx,
                router_logits=logits.detach(),
                router_probs=probs.detach(),
                topk_idx=top_experts.detach(),
                topk_weights=top_scores.detach(),
                residual=final_out.detach(),
            )
            for sink in sinks:
                sink.setdefault(key, point)

        return final_out, lb, lz


class LoopedStack(nn.Module):
    """The stack of L MoE blocks that gets called R times."""

    def __init__(self, context_length: int, d_model: int, num_layers_in_stack: int, num_heads: int,
                 d_ff: int, rope_theta: float, width_ratio: float, num_experts: int,
                 num_active: int, device=None, dtype=None, num_attention_sets: int = 1):
        super().__init__()
        #: One attention set per pass when per-pass attention is on, 1 otherwise.
        #: Only the LOOPED layers get this: head and tail run once, so "per pass"
        #: has no meaning for them and they keep a single set in every variant.
        self.num_attention_sets = num_attention_sets
        self.layers = nn.ModuleList(
            [
                GroupedMoEPrenormBlock(
                    d_model, num_heads, d_ff, num_experts, num_active,
                    context_length, rope_theta, width_ratio, device, dtype,
                    num_attention_sets=num_attention_sets,
                )
                for _ in range(num_layers_in_stack)
            ]
        )

    def forward(
        self,
        x: Tensor,
        *,
        loop_step: int,
        unrolled_pos_start: int,
        trace_sink: Optional[TraceSink] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """`unrolled_pos_start` is the depth coordinate this call's first layer occupies.

        Passed in rather than recomputed from `loop_step`, so the caller owns the depth
        axis in one place: the stack does not need to know how many layers ran before it.
        """
        lb_total = x.new_zeros(())
        lz_total = x.new_zeros(())
        for layer_idx, layer in enumerate(self.layers):
            x, lb, lz = layer(
                x, loop_step=loop_step, layer_idx=layer_idx, block="loop",
                unrolled_pos=unrolled_pos_start + layer_idx, trace_sink=trace_sink,
            )
            lb_total = lb_total + lb
            lz_total = lz_total + lz
        return x, lb_total, lz_total


class LoopedMoETransformer(nn.Module):
    """Looped MoE transformer: one shared stack applied ``num_stacks`` times.

    The loop is an explicit Python ``for``; ``loop_step`` is the loop variable and
    is threaded all the way down to each block. Nothing downstream ever has to
    recover it from tensor shapes.
    """

    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers_in_stack: int,
                 num_stacks: int, num_heads: int, d_ff: int, num_experts: int, num_active: int,
                 rope_theta: float, width_ratio: float, num_head_layers: int = 0,
                 num_tail_layers: int = 0, head_tail_num_experts: int = HEAD_TAIL_NUM_EXPERTS,
                 device=None, dtype=None, per_pass_attention: bool = False):
        super().__init__()
        self.num_stacks = num_stacks
        self.num_layers_in_stack = num_layers_in_stack
        self.total_layers = num_stacks * num_layers_in_stack
        self.num_head_layers = num_head_layers
        self.num_tail_layers = num_tail_layers
        # The unrolled depth, and the denominator the aux losses are averaged over.
        # Written as head + loop + tail rather than the recipe's shorthand "2 + D*L":
        # S15 has two head and two tail layers, so a literal 2 would divide by the wrong
        # number there -- and it would not fail, it would just make the auxiliary losses
        # quietly larger than intended.
        self.unrolled_depth = num_head_layers + self.total_layers + num_tail_layers

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # Head and tail are ordinary MoE layers that run once. They keep E=8/top-2
        # regardless of the loop block's expert count (recipe section 2.1: "identical in
        # every configuration"), so the loop block's E is deliberately not passed here.
        make_outer = lambda: GroupedMoEPrenormBlock(
            d_model, num_heads, d_ff, head_tail_num_experts, num_active,
            context_length, rope_theta, width_ratio, device, dtype,
        )
        self.head_layers = nn.ModuleList([make_outer() for _ in range(num_head_layers)])
        self.tail_layers = nn.ModuleList([make_outer() for _ in range(num_tail_layers)])
        self.per_pass_attention = per_pass_attention
        self.stack = LoopedStack(
            context_length, d_model, num_layers_in_stack, num_heads, d_ff, rope_theta,
            width_ratio, num_experts, num_active, device=device, dtype=dtype,
            # R sets when on: the stack is entered `num_stacks` times, and each
            # entry is what "a pass" means here.
            num_attention_sets=num_stacks if per_pass_attention else 1,
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        std_base_lm_head = math.sqrt(2 / (BASE_D_MODEL + vocab_size))
        self.lm_head = Linear(d_model, vocab_size, width_ratio, std_base_lm_head, device=device, dtype=dtype)

    @classmethod
    def from_config(cls, config: "LoopMoEConfig", *, device=None, dtype=None) -> "LoopedMoETransformer":
        """THE way to build this model from a config. Both call sites use it.

        There were two: the HF wrapper and `pretrain/train_spec.py`, each with its own
        hand-written keyword list. When head/tail layers were added, the training path's
        list was not updated, so it silently built a model with no head or tail while its
        config said otherwise -- it trained, the loss looked plausible, and every artifact
        recorded the config's depth rather than the depth that ran. Nothing could raise,
        because a shorter model is a perfectly valid model.

        A single entry point makes that class of drift impossible rather than merely
        tested-for: a field added to the config is read here once, and both paths get it.
        """
        return cls(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers_in_stack=config.num_layers_in_stack,
            num_stacks=config.num_stacks,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_experts=config.num_experts,
            num_active=config.num_active,
            rope_theta=config.rope_theta,
            width_ratio=config.width_ratio,
            num_head_layers=config.num_head_layers,
            num_tail_layers=config.num_tail_layers,
            # `getattr` with the config's own default, not a bare False: a config
            # object built by older code has no such attribute, and this is the
            # one entry point, so a silent False here would be the only place the
            # variant could be lost.
            per_pass_attention=getattr(config, "per_pass_attention", False),
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        *,
        trace_sink: Optional[TraceSink] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        lb_total = None
        lz_total = None

        x = self.token_embeddings(x)

        def run_outer(layers, block: str, pos: int, x, lb_total, lz_total):
            for i, layer in enumerate(layers):
                x, lb, lz = layer(
                    x, loop_step=0, layer_idx=0, block=block, unrolled_pos=pos + i,
                    trace_sink=trace_sink,
                )
                lb_total = lb if lb_total is None else lb_total + lb
                lz_total = lz if lz_total is None else lz_total + lz
            return x, lb_total, lz_total

        # head -> loop x num_stacks -> tail, with one running depth coordinate. head and
        # tail record loop_step=layer_idx=0 because neither coordinate means anything
        # outside the loop; `block` and `unrolled_pos` are what identifies them.
        x, lb_total, lz_total = run_outer(self.head_layers, "head", 0, x, lb_total, lz_total)
        pos = self.num_head_layers
        for loop_step in range(self.num_stacks):
            x, lb, lz = self.stack(
                x, loop_step=loop_step, unrolled_pos_start=pos, trace_sink=trace_sink,
            )
            pos += self.num_layers_in_stack
            lb_total = lb if lb_total is None else lb_total + lb
            lz_total = lz if lz_total is None else lz_total + lz
        x, lb_total, lz_total = run_outer(self.tail_layers, "tail", pos, x, lb_total, lz_total)

        x = self.lm_head(self.ln_final(x))

        # Averaged over the *unrolled* depth: every physical layer contributes once per
        # loop step, and head/tail contribute once each (recipe section 2.1). Equals
        # `total_layers` exactly when there are no head/tail layers, which is every
        # pre-ablation configuration -- so their published losses are unchanged.
        return x, lb_total / self.unrolled_depth, lz_total / self.unrolled_depth


def _require(kwargs: dict[str, Any], name: str) -> Any:
    """Fetch a required config field or raise.

    Project rule: semantic parameters get no defaults.
    A default is a silent-wrong-answer generator -- a mistyped or dropped key
    becomes a plausible number instead of an error.
    """
    if name not in kwargs or kwargs[name] is None:
        raise ValueError(
            f"LoopMoEConfig: required field {name!r} is missing. Semantic "
            "parameters have no defaults in this project; state it explicitly."
        )
    return kwargs.pop(name)


class LoopMoEConfig(PretrainedConfig):
    """Config for the looped-MoE architecture. **Every field is required.**

    Compatible with ``save_pretrained``/``from_pretrained``: a config.json written
    by this class round-trips, and one is rejected loudly if a field is absent.
    """

    model_type = "loop-moe"

    # Tells transformers not to introspect defaults by constructing `cls()` with
    # no arguments -- which this class deliberately rejects. Without it,
    # `save_pretrained` fails inside `_get_generation_parameters`. This is the
    # supported escape hatch for configs whose fields are all required.
    has_no_defaults_at_init = True

    def __init__(self, **kwargs: Any):
        # `from_pretrained` on a *torch-saved* config, and some HF-internal paths,
        # construct with no arguments at all; only a fully-specified call is valid.
        self.vocab_size = _require(kwargs, "vocab_size")
        self.context_length = _require(kwargs, "context_length")
        self.d_model = _require(kwargs, "d_model")
        self.num_heads = _require(kwargs, "num_heads")
        self.d_ff = _require(kwargs, "d_ff")
        self.rope_theta = _require(kwargs, "rope_theta")
        self.width_ratio = _require(kwargs, "width_ratio")
        self.num_layers_in_stack = _require(kwargs, "num_layers_in_stack")  # L
        self.num_stacks = _require(kwargs, "num_stacks")  # R
        self.num_experts = _require(kwargs, "num_experts")  # E
        self.num_active = _require(kwargs, "num_active")  # k
        self.lb_loss_factor = _require(kwargs, "lb_loss_factor")
        # Head/tail layers: MoE layers run once, outside the loop. 0 means the
        # architecture has none, which is not a guess -- it is what every configuration
        # built before the ablation family actually is, so their parameter counts are
        # unchanged. Real ablation configs never rely on the fallback:
        # `config_registry.build_ablation` states both counts for every entry.
        self.num_head_layers = int(kwargs.pop("num_head_layers", 0))
        self.num_tail_layers = int(kwargs.pop("num_tail_layers", 0))

        # Per-pass attention: each of the R passes over the shared stack gets its
        # OWN attention parameters, while the MoE, the norms and the router stay
        # shared. Default False, and defaulted rather than required precisely
        # because every config written before per-pass attention existed lacks the key: those
        # files must keep loading, and they must keep meaning what they meant.
        # False reproduces the previous model bit for bit, including the state
        # dict's key names -- see GroupedMoEPrenormBlock.
        self.per_pass_attention = bool(kwargs.pop("per_pass_attention", False))
        self.lz_loss_factor = _require(kwargs, "lz_loss_factor")

        # Stated, not inherited. `PretrainedConfig` defaults this to True, and the
        # only reason the embedding and the LM head are not already sharing storage
        # is that this model never implemented `get_output_embeddings()`. The day
        # someone adds it for tool compatibility, every configuration would start
        # tying weights -- a different model, trained to a different loss, with
        # nothing in any artifact saying so. The architecture uses untied weights
        # (the parameter counts in the recipe assume it), so the config says so.
        kwargs.pop("tie_word_embeddings", None)
        self.tie_word_embeddings = False

        self._validate()

        # Derived, for readers; never an input. The unrolled depth now includes the
        # layers that run once outside the loop.
        self.num_layers = self.num_stacks * self.num_layers_in_stack
        self.unrolled_depth = self.num_head_layers + self.num_layers + self.num_tail_layers

        # The original config mirrored `context_length` into `max_length` for
        # lm-evaluation-harness. transformers >=5 classifies `max_length` as a
        # generation parameter and refuses to serialise a config carrying one
        # (the check is `hasattr`, so even a property trips it). `context_length`
        # is therefore the single source of truth for sequence length; pass
        # `max_length` to the harness explicitly at eval time instead.
        # Popped so that loading a seedvar-era config.json cannot reintroduce it.
        kwargs.pop("max_length", None)

        super().__init__(**kwargs)

    def _validate(self) -> None:
        """Reject out-of-domain values loudly rather than failing deep in a kernel."""
        positive = (
            "vocab_size", "context_length", "d_model", "num_heads", "d_ff",
            "num_layers_in_stack", "num_stacks", "num_experts", "num_active",
        )
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"LoopMoEConfig.{name} must be a positive int, got {value!r}")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.num_active > self.num_experts:
            raise ValueError(
                f"num_active ({self.num_active}) cannot exceed num_experts ({self.num_experts})"
            )
        if self.d_ff % self.num_active != 0:
            raise ValueError(
                f"d_ff ({self.d_ff}) must be divisible by num_active ({self.num_active}); "
                "expert width is d_ff // num_active and truncation would silently "
                "change the parameter count."
            )
        for name in ("num_head_layers", "num_tail_layers"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"LoopMoEConfig.{name} must be a non-negative int, got {value!r}")
        if (self.d_model // self.num_heads) % 2 != 0:
            raise ValueError(
                f"head dim ({self.d_model // self.num_heads}) must be even for RoPE"
            )

    @property
    def num_hidden_layers(self) -> int:
        """Unrolled attention depth, for transformers' generation machinery.

        `generate()` in transformers 5.15 builds a `DynamicCache(config)`
        unconditionally, and that reads `config.num_hidden_layers` to decide how
        many per-layer cache slots to make. This model has no such field: its
        depth is `head + L*R + tail`, which is exactly `unrolled_depth`.

        A **property, not a stored field**, on purpose: `to_dict` serialises
        `self.__dict__`, so a stored attribute would start appearing in every
        `config.json` this class writes while the checkpoints already on the Hub
        would not have it -- two spellings of the same model. Derived here, the
        number is always consistent with the fields it comes from, and no
        artifact changes.

        Read-only for the same reason. If some future transformers path assigns
        to it, the AttributeError is the right outcome: silently accepting a
        depth that contradicts `num_stacks`/`num_layers_in_stack` would build a
        cache of the wrong shape and mis-attribute every generated token.
        """
        return self.unrolled_depth

    @property
    def trace_meta(self) -> TraceMeta:
        """Shape/provenance block handed to the metrics collector."""
        return TraceMeta(
            num_stacks=self.num_stacks,
            num_layers_in_stack=self.num_layers_in_stack,
            num_experts=self.num_experts,
            num_active=self.num_active,
        )


class LoopMoEForCausalLM(PreTrainedModel, GenerationMixin):
    """HF-compatible causal LM wrapper.

    Kept HF-shaped on purpose: "the diagnostics pipeline ingests our checkpoints
    unchanged" is an acceptance criterion, and that pipeline loads models through
    ``from_pretrained``.
    """

    config_class = LoopMoEConfig

    def __init__(self, config: LoopMoEConfig):
        super().__init__(config)
        self.model = LoopedMoETransformer.from_config(config)
        self.post_init()
        # Generation defaults to cache-free for this architecture; see
        # `_supports_cache_class`. Set on the instance's own generation config so
        # a caller that never touches generation settings still gets the correct
        # behaviour, while an explicit `model.generate(..., use_cache=True)`
        # still fails loudly instead of silently producing wrong text.
        if getattr(self, "generation_config", None) is not None:
            self.generation_config.use_cache = False

    def get_input_embeddings(self):
        return self.model.token_embeddings

    def set_input_embeddings(self, value):
        self.model.token_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[Tensor] = None,  # unused: the mask is built in
        labels: Optional[torch.LongTensor] = None,
        trace_sink: Optional[TraceSink] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Forward pass.

        Returns a ``CausalLMOutputWithPast`` whose ``loss`` is the *total* loss
        (CE + weighted aux). The unweighted components are attached as
        ``task_loss`` / ``lb_loss`` / ``z_loss`` so the training loop can log the
        breakdown without recomputing anything.

        **Label contract**: ``labels`` must already be
        next-token-shifted by the caller -- ``labels[..., t] == input_ids[..., t+1]``,
        with the last position set to ``-100`` (no target exists after it). This
        method does **not** shift internally; it passes ``labels`` to
        ``F.cross_entropy`` exactly as given. Previously the only written
        record of this contract was a comment in
        ``src/model/loop_trace.py`` ("the dataloader supplies pre-shifted
        labels during training, so the shift is explicit here") -- not here, at
        the definition itself. That gap let four separate call sites
        (the pretraining dataloader path aside, which was correct) independently
        get this wrong the same way, rather than it being four unrelated
        mistakes. Any caller not shifting first -- e.g. ``model(input_ids=ids,
        labels=ids)`` -- silently trains/evaluates on the trivial
        copy-the-current-token target instead of next-token prediction.
        """
        logits, lb, lz = self.model(input_ids, trace_sink=trace_sink)

        loss = task_loss = None
        if labels is not None:
            task_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = (
                task_loss
                + self.config.lb_loss_factor * lb
                + self.config.lz_loss_factor * lz
            )

        out = CausalLMOutputWithPast(loss=loss, logits=logits)
        # Unweighted components; the trainer pairs them with the factors from
        # config to build LossComponents.
        out.task_loss = task_loss
        out.lb_loss = lb
        out.z_loss = lz
        return out

    #: This model recomputes the whole prefix on every generation step: the
    #: looped stack reuses one set of attention weights `R` times, so a
    #: key/value cache would have to be keyed by (layer, pass) rather than by
    #: layer, which is not the shape `DynamicCache` provides. Generation is
    #: therefore cache-free -- correct but O(n^2) in the generated length, which
    #: is acceptable for the short-generation evaluation tasks this exists for.
    #: Stated as a class attribute so `generate()` does not build a cache at all
    #: rather than building one this model then ignores.
    _supports_cache_class = False
    _supports_static_cache = False

    def can_generate(self) -> bool:  # pragma: no cover - trivial
        return True

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Only the token ids. `past_key_values` is dropped rather than passed on.

        The forward pass takes the full prefix every time (see
        `_supports_cache_class`), so handing it a cache object would either be
        ignored silently or, worse, make a caller believe incremental decoding
        happened. `use_cache` is likewise not forwarded.
        """
        return {"input_ids": input_ids}

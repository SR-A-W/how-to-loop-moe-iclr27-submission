"""OLMoE (`allenai/OLMoE-1B-7B-0924`) router adapter.

Verified against the installed transformers 5.15.0 source
(`models/olmoe/modeling_olmoe.py::OlmoeTopKRouter.forward`, read line by
line, and a real-weight forward):

    router_logits = F.linear(hidden_states, self.weight)            # no bias
    router_probs  = softmax(router_logits, dtype=float32, dim=-1)   # over ALL E
    top_value, top_idx = topk(router_probs, top_k)
    if norm_topk_prob: top_value /= top_value.sum(-1)               # False for this model
    return router_logits, top_value, top_idx

So: scoring order is softmax-over-all-then-top-k; the combine weights the
model multiplies by are the UN-normalized top-k softmax probabilities
(they sum to ~0.25-0.45, not 1); no expert-bias mechanism; no shared
experts; balancing is a switch-style aux loss plus a router z-loss.

Coefficients: the paper (arXiv 2409.02060) states aux 0.01 and z-loss
0.001. The HF `config.json` on the `main` branch says 0.01 but on the
`fp32` and `step*` branches says 0.001 -- one training run cannot have had
two values, so the branch configs are unreliable and the PAPER values are
recorded, with this note.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.eval.probes.adapters.base import HFRouterAdapter, RoutingSpec

PAPER_AUX_LOSS_COEF = 0.01
PAPER_Z_LOSS_COEF = 0.001


class OlmoeRouterAdapter(HFRouterAdapter):
    router_class_name = "OlmoeTopKRouter"
    router_path_filter = None

    def unpack(self, output: tuple) -> tuple[Tensor, Tensor, Tensor]:
        # (router_logits, router_scores, router_indices) -- logits FIRST.
        # Granite/JetMoE put indices first; feeding that order here must raise.
        logits, weights, idx = output
        return logits, idx, weights

    def recompute(self, logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
        probs = torch.softmax(logits.float(), dim=-1)
        vals, idx = probs.topk(k, dim=-1)
        return idx, vals  # NOT renormalized: norm_topk_prob=False

    def top_k(self, config: Any) -> int:
        return int(config.num_experts_per_tok)

    def model_spec(self, model, *, release_dtype: str, transformers_version: str) -> tuple[dict, dict]:
        """Model-side RoutingSpec fields and, for each, its actual source
        Run-side fields (tokenizer, token counts, module paths,
        identity) are filled by the driver."""
        cfg = model.config
        if bool(cfg.norm_topk_prob):
            raise ValueError(
                "config.norm_topk_prob=True: this adapter's recompute rule assumes the "
                "un-normalized OLMoE-1B-7B-0924 convention; update `recompute` before use."
            )
        # Opens with the authority tag the RoutingSpec guard requires: these two coefficients
        # are stated numbers, so the source has to name where the numbers came from.
        paper = "paper: OLMoE arXiv:2409.02060 (training objective section: load-balance loss weight 0.01, router z-loss weight 0.001)"
        fields = dict(
            forward_dtype=str(next(model.parameters()).dtype), release_dtype=release_dtype,
            n_experts=int(cfg.num_experts), top_k=int(cfg.num_experts_per_tok),
            num_moe_layers=int(cfg.num_hidden_layers), num_stacks=1,
            scoring_order="softmax_all_then_topk", scores_are_probs=False,
            gate_bias=False, expert_bias_balancing=False, load_balancing_mechanism="auxiliary_loss",
            shared_experts=0, combine_weights_normalized_by_model=False,
            aux_loss_coef=PAPER_AUX_LOSS_COEF, z_loss_coef=PAPER_Z_LOSS_COEF, dropless=True,
            router_module_class=self.router_class_name, router_module_kind="feedforward_experts",
            transformers_version=transformers_version, variant=self.variant,
        )
        sources = dict(
            forward_dtype="dtype of the loaded parameters (driver --dtype); not inferred from code casts",
            release_dtype="safetensors header of the first weight shard",
            n_experts="config.json:num_experts", top_k="config.json:num_experts_per_tok",
            num_moe_layers="config.json:num_hidden_layers (every layer is MoE in OlmoeDecoderLayer)",
            num_stacks="non-looped architecture (transformers OlmoeModel: one pass over layers)",
            scoring_order="transformers 5.15.0 modeling_olmoe.py::OlmoeTopKRouter.forward (softmax over all experts, then topk on probs)",
            scores_are_probs="OlmoeTopKRouter.forward returns router_logits before softmax",
            gate_bias="OlmoeTopKRouter: weight-only nn.Parameter, no bias",
            expert_bias_balancing="modeling_olmoe.py: no expert-bias term in routing", load_balancing_mechanism=paper,
            shared_experts="modeling_olmoe.py / paper: no shared expert",
            combine_weights_normalized_by_model="config.json:norm_topk_prob=false",
            aux_loss_coef=paper + "; branch config.json values contradict each other (main 0.01 vs fp32/step 0.001) and are not used",
            z_loss_coef=paper, dropless="paper: dropless token-choice routing",
            router_module_class="class name of the hooked module", router_module_kind="OlmoeSparseMoeBlock is the FFN of each decoder layer",
            transformers_version="transformers.__version__", variant="no expert-bias mechanism -> not_applicable",
        )
        return fields, sources

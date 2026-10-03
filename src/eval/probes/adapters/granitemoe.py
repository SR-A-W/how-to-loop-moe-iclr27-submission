"""Granite 3.0 MoE (`ibm-granite/granite-3.0-1b-a400m-base`) router adapter.

Verified against transformers 5.15.0 `models/granitemoe/modeling_granitemoe.py::
GraniteMoeTopKRouter.forward` and a real-weight forward:

    router_logits = F.linear(hidden_states, self.weight).float()   # no bias
    top_k_logits, top_k_index = router_logits.topk(top_k, dim=-1)
    top_k_weights = softmax(top_k_logits, dim=-1)                  # SELECTED k only
    return top_k_index, top_k_weights, router_logits

No full-E softmax probability exists in this forward; combine weights are
normalized by construction. Balancing (Granite 3.0 technical report §2):
Fedus-style aux loss + router z-loss, dropless (ScatterMoE); the report
states no coefficients, and `config.router_aux_loss_coef=0.001` is an
inference-config value -> both recorded as "unknown". `model.forward` does
not support `output_router_logits` (returns None) -- hooks are the only way.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.eval.probes.adapters.base import HFRouterAdapter, RoutingSpec


class GraniteMoeRouterAdapter(HFRouterAdapter):
    router_class_name = "GraniteMoeTopKRouter"
    router_path_filter = None

    def unpack(self, output: tuple) -> tuple[Tensor, Tensor, Tensor]:
        idx, weights, logits = output  # indices FIRST (opposite of OLMoE)
        return logits, idx, weights

    def recompute(self, logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
        vals, idx = logits.float().topk(k, dim=-1)
        return idx, torch.softmax(vals, dim=-1)

    def top_k(self, config: Any) -> int:
        return int(config.num_experts_per_tok)

    def model_spec(self, model, *, release_dtype: str, transformers_version: str) -> tuple[dict, dict]:
        cfg = model.config
        report = "Granite 3.0 Language Models technical report §2 (eqs. 1-2 gating, eq. 4 load-balancing loss, eq. 5 router z-loss); no coefficients stated"
        fields = dict(
            forward_dtype=str(next(model.parameters()).dtype), release_dtype=release_dtype,
            n_experts=int(cfg.num_local_experts), top_k=int(cfg.num_experts_per_tok),
            num_moe_layers=int(cfg.num_hidden_layers), num_stacks=1,
            scoring_order="topk_then_softmax_selected", scores_are_probs=False,
            gate_bias=False, expert_bias_balancing=False, load_balancing_mechanism="auxiliary_loss",
            shared_experts=0, combine_weights_normalized_by_model=True,
            aux_loss_coef="unknown", z_loss_coef="unknown", dropless=True,
            router_module_class=self.router_class_name, router_module_kind="feedforward_experts",
            transformers_version=transformers_version, variant=self.variant,
        )
        sources = dict(
            forward_dtype="dtype of the loaded parameters (driver --dtype)", release_dtype="safetensors header of the first weight shard",
            n_experts="config.json:num_local_experts", top_k="config.json:num_experts_per_tok",
            num_moe_layers="config.json:num_hidden_layers (every layer has a GraniteMoeMoE FFN)", num_stacks="non-looped architecture",
            scoring_order="transformers 5.15.0 modeling_granitemoe.py::GraniteMoeTopKRouter.forward", scores_are_probs="router returns raw logits (3rd tuple element)",
            gate_bias="GraniteMoeTopKRouter: weight-only nn.Parameter", expert_bias_balancing="modeling_granitemoe.py: no expert-bias term",
            load_balancing_mechanism=report, shared_experts="report + modeling_granitemoe.py: no shared expert",
            combine_weights_normalized_by_model="softmax over the selected k sums to 1 by construction",
            aux_loss_coef="unknown: " + report + "; config.json:router_aux_loss_coef=0.001 is an inference-config value and is not used",
            z_loss_coef="unknown: " + report, dropless=report + " (ScatterMoE, dropless)",
            router_module_class="class name of the hooked module", router_module_kind="GraniteMoeMoE is the FFN of each decoder layer",
            transformers_version="transformers.__version__", variant="no expert-bias mechanism -> not_applicable",
        )
        return fields, sources

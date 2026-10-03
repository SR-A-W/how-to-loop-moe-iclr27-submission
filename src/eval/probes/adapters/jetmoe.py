"""JetMoE (`jetmoe/jetmoe-8b`) MLP-router adapter.

Verified against transformers 5.15.0 `models/jetmoe/modeling_jetmoe.py::
JetMoeTopKGating.forward` and a real-weight forward:

    logits = self.layer(hidden_states).float()          # nn.Linear(bias=False)
    top_k_logits, top_k_indices = logits.topk(top_k, dim=1)
    top_k_gates = softmax(top_k_logits, dim=1)          # over the SELECTED k only
    ... return index_sorted_experts, batch_index, batch_gates, expert_size, logits

Two things specific to this family:

1. **The same gating class routes attention experts too** (`JetMoeMoA`,
   `model.layers.{i}.self_attention.experts.router`), and
   `output_router_logits=True` returns all of them mixed (72 tensors for 24
   layers). Only MLP routers are wanted, so `router_path_filter=".mlp."`.
2. The hook output carries the decision in **expert-grouped order**
   (`index_sorted_experts` maps flat `(token*k + slot)` positions to sorted
   positions; `batch_gates` is in sorted order). `unpack` inverts that
   permutation back to per-token `(n_tokens, k)` tensors; the self-check
   then validates the inversion against the recomputed rule.

Balancing: aux loss (`aux_loss_coef`), coefficient NOT stated in the repo's
config.json (the installed `JetMoeConfig` default 0.01 is a library
default, not a training fact) -> recorded as "unknown". The repo config
uses keys `moe_num_experts` / `moe_top_k` that the installed config class
does not read; its defaults (8 / 2) happen to equal them, and the adapter
asserts that equality rather than trusting it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.eval.probes.adapters.base import HFRouterAdapter, RoutingSpec


class JetMoeRouterAdapter(HFRouterAdapter):
    router_class_name = "JetMoeTopKGating"
    router_path_filter = ".mlp."

    def unpack(self, output: tuple) -> tuple[Tensor, Tensor, Tensor]:
        index_sorted, batch_index, batch_gates, expert_size, logits = output
        n_tok = logits.shape[0]
        k = index_sorted.numel() // n_tok
        # sorted position j holds flat position index_sorted[j]; its expert is
        # the j-th entry of the experts sorted ascending (that is how the
        # model built `index_sorted_experts`: sort of topk_indices.flatten()).
        experts_sorted = logits.topk(k, dim=1).indices.flatten()[index_sorted]
        idx = torch.empty(n_tok * k, dtype=torch.long, device=logits.device)
        w = torch.empty(n_tok * k, dtype=torch.float32, device=logits.device)
        idx[index_sorted] = experts_sorted
        w[index_sorted] = batch_gates.float()
        return logits, idx.view(n_tok, k), w.view(n_tok, k)

    def recompute(self, logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
        vals, idx = logits.float().topk(k, dim=-1)
        return idx, torch.softmax(vals, dim=-1)

    def router_weight(self, module: nn.Module) -> Tensor:
        return module.layer.weight

    def top_k(self, config: Any) -> int:
        return int(config.num_experts_per_tok)

    def model_spec(self, model, *, release_dtype: str, transformers_version: str) -> tuple[dict, dict]:
        cfg = model.config
        repo_E = getattr(cfg, "moe_num_experts", None)
        repo_k = getattr(cfg, "moe_top_k", None)
        if repo_E is not None and int(repo_E) != int(cfg.num_local_experts):
            raise ValueError(f"repo config moe_num_experts={repo_E} != class num_local_experts={cfg.num_local_experts}")
        if repo_k is not None and int(repo_k) != int(cfg.num_experts_per_tok):
            raise ValueError(f"repo config moe_top_k={repo_k} != class num_experts_per_tok={cfg.num_experts_per_tok}")
        fields = dict(
            forward_dtype=str(next(model.parameters()).dtype), release_dtype=release_dtype,
            n_experts=int(cfg.num_local_experts), top_k=int(cfg.num_experts_per_tok),
            num_moe_layers=int(cfg.num_hidden_layers), num_stacks=1,
            scoring_order="topk_then_softmax_selected", scores_are_probs=False,
            gate_bias=False, expert_bias_balancing=False, load_balancing_mechanism="auxiliary_loss",
            shared_experts=0, combine_weights_normalized_by_model=True,
            aux_loss_coef="unknown", z_loss_coef="unknown", dropless="unknown",
            router_module_class=self.router_class_name, router_module_kind="feedforward_experts",
            transformers_version=transformers_version, variant=self.variant,
        )
        sources = dict(
            forward_dtype="dtype of the loaded parameters (driver --dtype)", release_dtype="safetensors header of the first weight shard",
            n_experts="config.json:moe_num_experts (asserted equal to JetMoeConfig.num_local_experts, whose library default 8 happens to match)",
            top_k="config.json:moe_top_k (asserted equal to JetMoeConfig.num_experts_per_tok, library default 2)",
            num_moe_layers="config.json:num_hidden_layers (every layer has a JetMoeMoE FFN)",
            num_stacks="non-looped architecture",
            scoring_order="transformers 5.15.0 modeling_jetmoe.py::JetMoeTopKGating.forward (topk on logits, softmax over the selected k)",
            scores_are_probs="JetMoeTopKGating.forward returns raw logits (5th tuple element)",
            gate_bias="JetMoeTopKGating.layer = nn.Linear(bias=False)", expert_bias_balancing="modeling_jetmoe.py: no expert-bias term",
            load_balancing_mechanism="JetMoeForCausalLM uses load_balancing_loss_func (aux loss); coefficient not in repo config",
            shared_experts="modeling_jetmoe.py: no shared expert", combine_weights_normalized_by_model="softmax over the selected k sums to 1 by construction",
            aux_loss_coef="unknown: repo config.json has no aux_loss_coef; library default 0.01 is NOT a training fact; paper not consulted for a number",
            z_loss_coef="unknown: not stated in repo config or model card", dropless="unknown: not stated",
            router_module_class="class name of the hooked module",
            router_module_kind="only routers under model.layers.{i}.mlp (JetMoeMoE); attention-expert routers (JetMoeMoA) deliberately not hooked",
            transformers_version="transformers.__version__", variant="no expert-bias mechanism -> not_applicable",
        )
        return fields, sources

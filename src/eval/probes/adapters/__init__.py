"""Adapters that attach this project's router metrics to open-source HF MoE
models. See `base.py` for the contract; one module per model family."""

from src.eval.probes.adapters.granitemoe import GraniteMoeRouterAdapter
from src.eval.probes.adapters.jetmoe import JetMoeRouterAdapter
from src.eval.probes.adapters.olmoe import OlmoeRouterAdapter

ADAPTERS = {
    "olmoe": OlmoeRouterAdapter,
    "jetmoe": JetMoeRouterAdapter,
    "granitemoe": GraniteMoeRouterAdapter,
}

__all__ = ["ADAPTERS", "OlmoeRouterAdapter", "JetMoeRouterAdapter", "GraniteMoeRouterAdapter"]

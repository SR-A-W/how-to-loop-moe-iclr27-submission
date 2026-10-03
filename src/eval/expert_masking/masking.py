"""Apply a `(layer, expert)` mask to a looped-MoE model's routing, for one forward.

Masking sets the routing PROBABILITY of the masked experts to zero before the
top-k, so the token picks its k from the survivors and the combine weights are
renormalised over that pair (procedure section 3). The equivalent spelling --
setting the raw score to -inf -- produces the identical selection and identical
renormalised weights, but leaves -inf in `router_logits`, and the collapse
metric estimates `sigma_hat` from those, turning it and everything derived from
it into NaN. Zeroing the probabilities keeps the logits finite and untouched.

Only the loop block is masked. Head and tail layers run once, keep their own
expert count, and are out of scope by section 1 -- and an expert id chosen from
a 64-expert loop block is not even a valid index into an 8-expert head.
"""
from __future__ import annotations

import contextlib
from typing import Iterable

import torch

Expert = tuple[int, int]


def loop_layers(model) -> list:
    """The loop block's physical layers, in order.

    Reached through the stack rather than by scanning every module with a
    router: that scan would also return the head and tail blocks, and they must
    not be masked.
    """
    inner = getattr(model, "model", model)
    stack = getattr(inner, "stack", None)
    if stack is None or not hasattr(stack, "layers"):
        raise SystemExit("REFUSING: model has no `stack.layers`; not a looped-MoE model")
    return list(stack.layers)


@contextlib.contextmanager
def masked_routing(model, experts: Iterable[Expert]):
    """Zero the routing probability of `experts` for the duration of the block.

    Restores by deleting the instance attribute rather than reassigning the
    bound method: reassigning leaves a permanent shadow of the class method on
    the instance, so a later "unmasked" pass would still run through a
    patched-then-restored path instead of the original one.
    """
    layers = loop_layers(model)
    by_layer: dict[int, list[int]] = {}
    for layer, expert in experts:
        if not 0 <= layer < len(layers):
            raise SystemExit(
                f"REFUSING: layer {layer} outside 0..{len(layers) - 1} of the loop block"
            )
        by_layer.setdefault(layer, []).append(expert)

    restore: list[tuple[object, bool]] = []
    for layer_idx, expert_ids in by_layer.items():
        router = layers[layer_idx].router
        width = router.gate.weight.shape[0]
        bad = [e for e in expert_ids if not 0 <= e < width]
        if bad:
            raise SystemExit(
                f"REFUSING: experts {bad} outside 0..{width - 1} of layer {layer_idx}"
            )
        index = torch.tensor(sorted(expert_ids), dtype=torch.long)

        def make(router=router, index=index):
            def forward(x):
                logits = router.gate(x)
                probs = torch.softmax(logits.float(), dim=-1)
                probs = probs.index_fill(-1, index.to(probs.device), 0.0)
                top_scores, top_experts = torch.topk(probs, k=router.num_active, dim=-1)
                top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)
                return logits, probs, top_scores, top_experts
            return forward

        restore.append((router, "forward" in router.__dict__))
        router.forward = make()
    try:
        yield {"masked": sorted(by_layer.items()), "n_masked": sum(map(len, by_layer.values()))}
    finally:
        for router, had_own in restore:
            if not had_own:
                del router.forward

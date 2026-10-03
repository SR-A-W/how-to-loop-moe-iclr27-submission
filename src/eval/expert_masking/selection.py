"""Shares, target sets and random controls for the expert-masking evaluation.

Procedure version 1 (see README, Expert masking).

Everything here is pure: it takes counts and returns sets, so the parts that
decide WHICH experts get masked can be checked by hand on a four-expert example
without a GPU. The forward passes live in `run.py`.

A physical expert is a `(layer, expert)` pair. The loop block's layer is reused
once per pass, so a pass is not a layer: merging the passes back onto the layer
is the whole point of section 2, and keying on the expert id alone would merge
DIFFERENT layers' experts into one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

Expert = tuple[int, int]  # (layer_idx, expert_id)


def shares_by_physical_expert(counts: dict[int, np.ndarray]) -> dict[Expert, float]:
    """Per-`(layer, expert)` share, normalised as section 2 specifies.

    `counts[layer]` is the number of times each expert of that layer was chosen
    into the top-k, already summed over the loop passes.

    Two normalisations, in this order: within a layer the shares sum to 1, then
    every layer is divided by the layer count D so the whole block sums to 1.
    Dividing by D matters as soon as D > 1 -- without it a layer's shares would
    each be D times too large, and since the target set is chosen by a CUMULATIVE
    threshold over all layers, the set would stop early and be too small. With
    one layer (S4) the division is a no-op, which is exactly why the pilot never
    exposed this.
    """
    if not counts:
        raise ValueError("no layers given; cannot compute shares over nothing")
    n_layers = len(counts)
    shares: dict[Expert, float] = {}
    for layer, per_expert in sorted(counts.items()):
        arr = np.asarray(per_expert, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"layer {layer}: counts must be a non-empty 1-D array")
        if (arr < 0).any():
            raise ValueError(f"layer {layer}: negative selection counts")
        total = arr.sum()
        if total <= 0:
            raise ValueError(f"layer {layer}: no expert was ever selected")
        for expert, value in enumerate(arr / total / n_layers):
            shares[(layer, expert)] = float(value)
    return shares


@dataclass(frozen=True)
class MaskSet:
    """One set of experts to mask, with what it actually came to."""

    name: str
    experts: tuple[Expert, ...]
    actual_share: float
    per_layer_counts: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n_experts": len(self.experts),
            "actual_share": self.actual_share,
            # Members are recorded, not just the count: two sets with the same
            # size and share can be different sets, and the comparison this
            # evaluation exists for is between sets.
            "experts": [list(e) for e in self.experts],
            "per_layer_counts": {str(k): v for k, v in sorted(self.per_layer_counts.items())},
        }


def _accumulate_nearest(
    ordered: Sequence[Expert], shares: dict[Expert, float], target: float
) -> tuple[list[Expert], float]:
    """Take a prefix of `ordered` whose cumulative share is nearest `target`.

    Nearest, not first-to-exceed: overshooting can land further from the target
    than stopping one short, and for the random controls that difference is the
    whole point -- a control that removes more traffic than the target set makes
    the comparison unfair in the direction that flatters the hypothesis.
    """
    best_k, best_gap, cumulative, running = 0, abs(target), 0.0, 0.0
    for k, expert in enumerate(ordered, start=1):
        running += shares[expert]
        gap = abs(running - target)
        if gap < best_gap:
            best_k, best_gap, cumulative = k, gap, running
    return list(ordered[:best_k]), cumulative


def target_set(shares: dict[Expert, float], percent: float) -> MaskSet:
    """`T(x)`: lowest-share experts accumulated to nearest `percent`%."""
    if not 0 < percent <= 100:
        raise ValueError(f"percent must be in (0, 100], got {percent}")
    ordered = sorted(shares, key=lambda e: (shares[e], e))
    experts, actual = _accumulate_nearest(ordered, shares, percent / 100.0)
    return MaskSet(f"T({percent:g}%)", tuple(sorted(experts)), actual, _per_layer(experts))


def random_control(
    shares: dict[Expert, float], target: MaskSet, seed: int, name: str | None = None
) -> MaskSet:
    """`R_j(x)`: random experts from OUTSIDE `target`, to the same actual share.

    Excluding the target's own members is required by the procedure and is not a
    detail: a control allowed to reuse them would be partly the target set, and
    the two would be compared against each other with an overlap nobody records.
    """
    pool = sorted(set(shares) - set(target.experts))
    if not pool:
        raise ValueError("no experts left outside the target set to draw a control from")
    rng = np.random.default_rng(seed)
    ordered = [pool[i] for i in rng.permutation(len(pool))]
    experts, actual = _accumulate_nearest(ordered, shares, target.actual_share)
    return MaskSet(name or f"R{seed}({target.name})", tuple(sorted(experts)), actual,
                   _per_layer(experts))


def _per_layer(experts: Iterable[Expert]) -> dict[int, int]:
    out: dict[int, int] = {}
    for layer, _ in experts:
        out[layer] = out.get(layer, 0) + 1
    return out


# --- Per-layer feasibility ---------------------------------------------------
#
# A masking set must leave every layer at least `top_k` usable experts. With
# fewer, top-k has to pick a masked expert (zero weight, or for a fully masked
# layer a division by zero and a NaN loss). Random draws that violate this are
# skipped deterministically and the next seed is used; a target that violates
# it makes the tier undefined on that model.


def layer_sizes(shares: dict[Expert, float]) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for layer, _ in shares:
        sizes[layer] = sizes.get(layer, 0) + 1
    return sizes


def per_layer_min_remaining(mask: MaskSet, sizes: dict[int, int]) -> int:
    """Fewest usable experts any layer is left with under `mask`."""
    return min(sizes[layer] - mask.per_layer_counts.get(layer, 0) for layer in sizes)


def leaves_enough(mask: MaskSet, sizes: dict[int, int], top_k: int) -> bool:
    return per_layer_min_remaining(mask, sizes) >= top_k


def random_controls(
    shares: dict[Expert, float], target: MaskSet, n: int, top_k: int,
    seeds=None,
) -> tuple[list[MaskSet], list[dict]]:
    """The first `n` feasible random controls, and the seeds that were skipped.

    Seeds are tried in order 0, 1, 2, ...; a seed whose draw leaves some layer
    with fewer than `top_k` usable experts is skipped and recorded. The choice
    is a function of (shares, target, top_k) only, so a later run with a larger
    `n` extends this list rather than reshuffling it.
    """
    sizes = layer_sizes(shares)
    if not leaves_enough(target, sizes, top_k):
        raise ValueError(
            f"target {target.name} leaves a layer with fewer than {top_k} usable experts "
            f"(min remaining {per_layer_min_remaining(target, sizes)}); undefined on this model")
    controls, skipped, seed = [], [], 0
    while len(controls) < n:
        if seed > 100 * max(n, 1):
            raise ValueError(f"could not find {n} feasible controls in {seed} seeds")
        c = random_control(shares, target, seed, name=f"R{seed}({target.name})")
        if leaves_enough(c, sizes, top_k):
            controls.append(c)
        else:
            skipped.append({"seed": seed, "reason": "leaves a layer with fewer than top_k usable experts",
                            "per_layer_counts": {str(k): v for k, v in sorted(c.per_layer_counts.items())},
                            "min_remaining": per_layer_min_remaining(c, sizes)})
        seed += 1
    return controls, skipped

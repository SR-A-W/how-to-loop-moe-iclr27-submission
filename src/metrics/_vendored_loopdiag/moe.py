# ---------------------------------------------------------------------------
# VENDORED FILE — DO NOT EDIT BELOW THIS HEADER.
#
# Source: an earlier, separate diagnostics code base (read-only reference);
# original module path metrics/moe.py.
#
# WHY vendored instead of imported: this is the repo that ships to the
# training cluster; that code base does not and is not a
# submodule. Training-time metrics must run standalone on the training
# cluster, so the metric DEFINITIONS this project already verified/tested are
# copied verbatim rather than reimplemented (reuse, don't reinvent).
#
# Everything from here to end-of-file is code-identical to the source path
# above, except where marked "deviation from upstream".
# In the development repo a sync test (not shipped here) compares this file's
# code with the source file and fails on drift.
# ---------------------------------------------------------------------------

"""MoE routing diagnostics.

Pure functions over router scores / top-k assignments. No model code
imported; nothing here depends on HF `transformers` or any specific MoE
implementation.

Two input flavors recur throughout this module, always explicit via an
`input_type` or `is_*` flag (never silently guessed --
the pre/post-softmax ambiguity in HF `router_logits` bit the original
research; we refuse to repeat that mistake here):

  - "logits": raw, pre-softmax gate outputs, shape (..., n_experts).
  - "probs":  already-normalized routing probabilities, shape (..., n_experts).
  - "indices": already-computed top-k expert indices, shape (..., k), integer.

RESOLVED: the pre/post-softmax question is settled by two
independent verification paths that agree -- an empirical
probe (transformers 4.55.0 and 5.6.2, across Mixtral/Qwen3-MoE/OLMoE: row
sums != 1, negative entries present) and a collector
(hooks the router `nn.Linear` module directly, whose output is `gate(x)` by
definition). **HF `router_logits` is PRE-softmax**; the "post-softmax" wording
in `transformers/utils/modeling_outputs.py` (lines ~390/419, which also
misspells "logits" as "logtis") is a stale/incorrect docstring, not a
contract. `input_type="logits"` remains every function's default in this
module for that reason -- it was already the safe default chosen before this
was confirmed (per "don't bet on the ambiguous one" guidance), and now it is
additionally the *empirically verified correct* one, not merely the cautious
guess. `input_type="probs"` remains fully supported for callers who
pre-normalize.

WARNING -- do not assume this module's softmax-based entropy applies to every
gating mechanism: DeepSeek-V3's router uses
**sigmoid + group-limited top-k, not softmax**, and Ouro's `early_exit_gate`
is a `Linear -> sigmoid` producing an independent per-step probability, not a
categorical distribution over positions. Applying this module's
`_to_probs`/softmax-entropy machinery to either of those would be silently
wrong (a sigmoid output does not sum to 1 across anything, and there is
nothing to "normalize" it against). Sigmoid-gate diagnostics belong in
`exit_gate.py`, not here -- see that module's docstring for why the two are
not interchangeable.

KNOWN UNAVAILABLE from this collection path:
**token drop rate is NOT implemented here, and deliberately so.**
HF `output_router_logits` was confirmed to return routing
*scores*, not the actual expert assignment executed by the model, and that
capacity-limited drops are invisible in those scores -- a dropped token's
router logits look exactly like a kept token's. An earlier version of this
module included a `token_drop_rate` function that *simulated* a capacity
policy on top of a synthetic top-1 assignment; that was removed
because a number computed that way describes a
hypothetical policy the model may never have run, not an observed drop rate,
and a plausible-looking-but-fabricated number is worse than no number in a
"fast exploration" workflow where docstring caveats are easy to skip past.
**If real drop information is ever needed, it must come from the
collector's own capacity-tracking logic (a real drop mask captured at
execution time), not be reconstructed here from router scores.** That is a
collection-layer feature request, not something this module can provide.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_probs(scores: torch.Tensor, input_type: str) -> torch.Tensor:
    if input_type == "logits":
        return F.softmax(scores, dim=-1)
    if input_type == "probs":
        return scores
    raise ValueError(f"input_type must be 'logits' or 'probs', got {input_type!r}")


def topk_from_scores(
    scores: torch.Tensor, k: int, input_type: str = "logits"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared top-k helper: normalize (if needed) then take top-k experts.

    Returns (probs, topk_vals, topk_idx) where probs is the full
    (..., n_experts) distribution, topk_vals/topk_idx are (..., k).
    `input_type` must be explicit -- see module docstring.
    """
    probs = _to_probs(scores, input_type)
    topk_vals, topk_idx = probs.topk(k, dim=-1)
    return probs, topk_vals, topk_idx


def expert_load_distribution(
    topk_idx: torch.Tensor, n_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert token-slot counts and the load coefficient of variation.

    Definition
    ----------
    Flatten `topk_idx` (any leading shape, last dim or all dims are expert
    ids -- pass e.g. (n_tokens, k)) into a 1D list of expert ids and
    bincount:

        load(e) = number of (token, k-slot) pairs assigned to expert e

    Note this counts *assignment slots*, not tokens: a token counts once per
    k it activates, so `sum(load) == topk_idx.numel()`, not `n_tokens`. This
    matches how aux load-balancing losses (Switch Transformer, Mixtral) are
    normally computed.

    CV = std(load) / mean(load), population standard deviation (divide by
    n_experts, not n_experts - 1) since we treat the n_experts counts as the
    full population being described, not a sample estimating some larger
    population. CV = 0 means perfectly uniform load; CV grows without bound
    as load concentrates on fewer experts (dead experts push CV up sharply).

    Known-answer check (see tests): uniform assignment (every expert gets
    exactly the same count) gives CV = 0.

    Args:
        topk_idx: integer tensor of expert ids, any shape.
        n_experts: total number of experts (bincount minlength).

    Returns:
        (load, cv): load is a 1D tensor of length n_experts (integer
        counts as float), cv is a 0-d tensor.
    """
    flat = topk_idx.reshape(-1)
    load = torch.bincount(flat, minlength=n_experts).float()
    mean = load.mean()
    std = load.std(unbiased=False)
    cv = std / mean.clamp_min(1e-12)
    return load, cv


def router_entropy(
    scores: torch.Tensor,
    input_type: str = "logits",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-token entropy of the full router distribution over all experts.

    Definition: same Shannon entropy as `attention.attention_entropy`
    (H = -sum_e p_e log p_e), applied to the *entire* (..., n_experts)
    router distribution -- not just the activated top-k. This measures how
    decisive routing is overall: near-uniform routing (entropy ~ log(n_experts))
    signals the router isn't discriminating between experts; near-zero
    entropy signals a very peaked (near-one-hot) routing decision.

    Reduction: none baked in, returns per-token entropy (shape
    `scores.shape[:-1]`); average yourself for a summary scalar.

    `input_type` must be given explicitly (see module docstring on the
    pre/post-softmax ambiguity).

    Args:
        scores: (..., n_experts) logits or probs.
        input_type: "logits" (softmax applied internally) or "probs"
            (used as-is).
        eps: log-clamp floor (same guarded-zero handling as
            `attention.attention_entropy`).

    Returns:
        Tensor of shape scores.shape[:-1], entropy in nats.
    """
    p = _to_probs(scores, input_type)
    plogp = torch.where(p > 0, p * torch.log(p.clamp_min(eps)), torch.zeros_like(p))
    return -plogp.sum(dim=-1)


def topk_confidence(
    scores: torch.Tensor,
    input_type: str = "logits",
    ks: tuple[int, ...] = (1, 2),
) -> dict[str, torch.Tensor]:
    """Per-token top-1 / top-2 routing-confidence distributions.

    Definition
    ----------
    For each token, sort router probabilities descending: p_(1) >= p_(2) >= ...
    For each requested k in `ks`:

        confidence_k(token) = sum_{i=1..k} p_(i)      (cumulative mass of the k largest)

    So `confidence_1` is just the max probability (standard "top-1
    confidence"), and **`confidence_2` is the sum of the two largest
    probabilities (cumulative mass captured by the top-2 experts), not the
    second-highest probability alone.** This is a deliberate definition
    choice: "top-2 confidence" is ambiguous in casual usage between
    (a) mass captured by choosing 2 experts (this function, relevant to a
    top-2-routed MoE's actual behavior) and (b) the probability of the
    second-place expert alone (answers "how close is the runner-up",
    relevant to routing-decision stability/ties). We report (a) as the
    named `confidence_k` because it directly answers "how much of the
    routing decision does keeping only the top-k experts explain", matching
    how top-k MoE inference actually works. To get definition (b), take
    `sorted_probs[..., 1]` from the returned raw tensor instead.

    No reduction is baked in for the per-token arrays (the "distribution" in
    "top-1/top-2 confidence distribution" *is* this
    per-token array -- histogram it directly), but a mean/std summary is
    also included per k for convenience.

    Args:
        scores: (..., n_experts) logits or probs.
        input_type: "logits" or "probs" (see module docstring).
        ks: which cumulative-top-k confidences to compute.

    Returns:
        Dict with keys:
          - "sorted_probs": (..., n_experts) probs sorted descending.
          - "confidence_{k}": (...,) cumulative top-k mass, for each k in ks.
          - "confidence_{k}_mean" / "confidence_{k}_std": scalars.
    """
    probs = _to_probs(scores, input_type)
    sorted_probs, _ = probs.sort(dim=-1, descending=True)
    out: dict[str, torch.Tensor] = {"sorted_probs": sorted_probs}
    for k in ks:
        conf = sorted_probs[..., :k].sum(dim=-1)
        out[f"confidence_{k}"] = conf
        out[f"confidence_{k}_mean"] = conf.mean()
        out[f"confidence_{k}_std"] = conf.std(unbiased=False)
    return out


def expert_coactivation_matrix(
    topk_idx: torch.Tensor,
    n_experts: int,
    normalize: str | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Symmetric expert co-activation counts/rates.

    Definition
    ----------
    Input `topk_idx`: (n_tokens, k) integer expert ids (the top-k experts
    chosen for each token; indices within a token's row are assumed
    distinct, as produced by `torch.topk`). Build a per-token multi-hot
    membership matrix `M`: (n_tokens, n_experts), M[t, e] = 1 if expert e is
    among token t's top-k. Then:

        coact = M^T @ M       # (n_experts, n_experts)

    `coact[e, e]` is expert e's total load (matches `expert_load_distribution`'s
    `load`); `coact[e, f]` (e != f) is the number of tokens for which both e
    and f were simultaneously activated -- direct evidence of routing
    redundancy (two experts consistently chosen together suggest they may
    be functionally correlated / a candidate for merging).

    `normalize`:
      - None (default): raw co-occurrence counts.
      - "conditional": row-normalize by the diagonal, giving
        `P(f active | e active) = coact[e,f] / load(e)`. **Asymmetric** --
        this is a conditional probability, not a similarity measure; do not
        expect `result[e,f] == result[f,e]`.
      - "jaccard": `coact[e,f] / (load(e) + load(f) - coact[e,f])`,
        the symmetric Jaccard index of the two experts' token sets. This is
        the recommended normalization for comparing co-activation strength
        across expert pairs with very different overall loads (raw counts
        are dominated by high-load experts).

    Args:
        topk_idx: (n_tokens, k) integer expert ids.
        n_experts: total number of experts.
        normalize: None, "conditional", or "jaccard".
        eps: denominator floor for the normalized variants.

    Returns:
        (n_experts, n_experts) tensor.
    """
    n_tokens = topk_idx.shape[0]
    # deviation from upstream: device fix -- same class of bug as
    # expert_mass_distribution's `mass` tensor below in this file: without
    # device=, scatter_ raises "Expected all tensors to be on the same
    # device" the moment topk_idx lives on a CUDA device.
    membership = torch.zeros(n_tokens, n_experts, dtype=torch.float32, device=topk_idx.device)
    membership.scatter_(1, topk_idx.long(), 1.0)
    coact = membership.T @ membership
    if normalize is None:
        return coact
    load = coact.diagonal()
    if normalize == "conditional":
        return coact / load.clamp_min(eps).unsqueeze(1)
    if normalize == "jaccard":
        union = load.unsqueeze(0) + load.unsqueeze(1) - coact
        return coact / union.clamp_min(eps)
    raise ValueError(f"normalize must be None, 'conditional', or 'jaccard', got {normalize!r}")


def dead_expert_count(
    topk_idx: torch.Tensor | None,
    n_experts: int,
    load: torch.Tensor | None = None,
    threshold: int = 0,
) -> dict[str, torch.Tensor]:
    """Count experts whose load is at or below `threshold` (default: never used).

    Definition: `dead(e) = load(e) <= threshold`. Pass either raw
    `topk_idx` (counts computed internally via the same bincount as
    `expert_load_distribution`) or a precomputed `load` vector directly
    (e.g. aggregated across a whole eval set without keeping every
    `topk_idx` around) -- exactly one of the two must be given.

    `threshold=0` (default) means "literally never selected"; raising it
    (e.g. `threshold=n_tokens * 1e-4`) answers a softer question ("selected
    on fewer than X% of tokens"), left to the caller.

    Args:
        topk_idx: integer tensor of expert ids, any shape (mutually
            exclusive with `load`).
        n_experts: total number of experts.
        load: precomputed 1D load vector of length n_experts (mutually
            exclusive with `topk_idx`).
        threshold: dead iff load <= threshold.

    Returns:
        Dict with "count" (0-d), "mask" (bool, length n_experts), "load"
        (the load vector used, for inspection).
    """
    if (topk_idx is None) == (load is None):
        raise ValueError("exactly one of topk_idx or load must be given")
    if load is None:
        load, _ = expert_load_distribution(topk_idx, n_experts)
    mask = load <= threshold
    return {"count": mask.sum(), "mask": mask, "load": load}


def cross_loop_step_overlap(
    data: torch.Tensor,
    input_type: str = "scores",
    k: int | None = None,
    n_experts: int | None = None,
    meta: object | None = None,
) -> torch.Tensor:
    """★ Core novelty metric: pairwise top-k routing overlap across loop steps.

    Background / adaptation
    ------------------------
    OLMoE (Muennighoff et al. 2024, arXiv 2409.02060) defines "router
    saturation" between a training checkpoint t and the final checkpoint T:

        Saturation(t) = (1/N) * sum_i |E_i^(t) ∩ E_i^(T)| / k

    where E_i^(t) is the set of top-k experts token i is routed to at
    checkpoint t. This is a measure of routing-decision *convergence over
    training time*.

    **This project's adaptation** (the novelty target): in a looped model,
    the *same* MoE block is invoked K times (once per loop step) rather than K different checkpoints existing
    across training. We replace "checkpoint" with "loop step" and ask the
    exact same structural question: does token i get routed to the same
    experts on loop step a as on loop step b?

        Overlap(a, b) = (1/N) * sum_i |E_i^(a) ∩ E_i^(b)| / k

    High overlap across all step pairs -> the loop is not producing
    functional differentiation in routing (experts do "the same job" every
    iteration). Overlap that decays with |a - b|, or that separates into
    early-loop vs late-loop clusters, is direct evidence of "the loop
    implements staged computation".

    This function generalizes OLMoE's single (t, T) comparison to the full
    S x S matrix of pairwise loop-step comparisons (S = number of loop
    steps), since with only a handful of loop steps (typically single
    digits) computing every pair is cheap and strictly more informative
    than only comparing against the last step. To recover the original
    OLMoE-style "vs final iterate" curve exactly, take `result[:, -1]`
    (or `result[:, -1]` renamed Saturation(t) with t = loop step, T = final
    loop step).

    Reuse beyond MoE experts: nothing here is MoE-specific -- "n_experts" is
    just "n_categories" for whatever discrete top-k choice is made once per
    loop step. The same function is intended to be reused verbatim for any
    other per-loop-step discrete decision a model makes (e.g. a future
    top-1 argmax-style routing decision), as long as it can be expressed as
    "each (step, token) activates a k-subset of a fixed, shared category
    set". It does NOT apply to a per-step independent scalar probability
    that isn't a choice among shared categories (e.g. Ouro's
    `early_exit_gate` sigmoid output, which is a per-step Bernoulli
    "continue/exit" signal, not a subset of a shared expert set) -- see
    `exit_gate.py` for that case instead.

    Input handling
    --------------
    `input_type="scores"` (default): `data` is (n_steps, ..., n_experts)
    raw routing scores (logits or probs -- doesn't matter, since we only
    need the top-k *set*, which softmax as a monotonic transform per row
    does not change); `k` is required, and top-k is computed per
    (step, ...-index) internally.

    `input_type="indices"`: `data` is already (n_steps, ..., k) integer
    expert indices (e.g. precomputed top-k); `k` is inferred from
    `data.shape[-1]` if not given, `n_experts` is required (to size the
    one-hot membership tensor).

    Middle dims / block axis: any number of axes are allowed between the
    leading loop-step axis and the trailing experts/k axis (e.g. the
    collector's `(K, B, S, E)` slice for one fixed block, or a flattened
    `(K, n_tokens, E)`) -- they are flattened internally into a single
    "token" axis before computing overlap. **If your tensor still has a
    block/layer axis in there (e.g. the collector's full `(K, L, B, S, E)`
    with L un-sliced), flattening will silently mix tokens from different,
    structurally distinct MoE modules into one overlap number.** Slice a
    single block index first (e.g. `data[:, block_idx]`) and call this
    function once per block if you need a per-block overlap matrix -- this
    function does not know which axis (if any) is a block axis and will not
    guard against that mistake for you.

    Computation: build a boolean membership tensor
    `onehot: (n_steps, n_tokens, n_experts)` (one-hot union of the k chosen
    experts per (step, token), after flattening middle dims into
    `n_tokens`), then

        Overlap(a, b) = mean_i [ sum_e onehot[a,i,e] * onehot[b,i,e] ] / k

    computed for all (a, b) at once via `einsum('ane,bne->abn', ...)` then
    averaged over tokens and divided by k. This is exact (no sampling), and
    vectorized rather than looping over pairs.

    Known-answer checks (see tests): identical routing at two steps gives
    Overlap = 1.0 for that pair; completely disjoint expert sets (k <=
    n_experts / 2, constructed so no expert is shared) give Overlap = 0.0;
    the diagonal `result[s, s]` is always exactly 1.0 (a step's routing
    trivially fully overlaps with itself).

    Args:
        data: (n_steps, ..., n_experts) scores, or (n_steps, ..., k)
            indices -- see `input_type`. Any middle dims are flattened
            into a token axis (see "Middle dims / block axis" above).
        input_type: "scores" or "indices".
        k: top-k size (required for "scores"; inferred for "indices" if
            omitted).
        n_experts: total expert count (required for "indices"; inferred
            from `data.shape[-1]` for "scores").
        meta: optional run metadata (e.g. a `TraceMeta`-like dict/object
            with `n_loops`/`block_paths`/etc.); when given, the return
            value is wrapped as `{"overlap": <matrix>, "meta": meta}`
            instead of the bare tensor, so a logged result is never
            separated from which run/block produced it. Default `None`
            preserves the old bare-tensor return for backward
            compatibility.

    Returns:
        (n_steps, n_steps) symmetric tensor, entries in [0, 1], diagonal = 1
        (or a `{"overlap": ..., "meta": ...}` dict if `meta` is given).
    """
    if input_type == "scores":
        if k is None:
            raise ValueError("k is required when input_type='scores'")
        n_experts = data.shape[-1] if n_experts is None else n_experts
        _, topk_idx = data.topk(k, dim=-1)
    elif input_type == "indices":
        if n_experts is None:
            raise ValueError("n_experts is required when input_type='indices'")
        topk_idx = data
        k = data.shape[-1] if k is None else k
    else:
        raise ValueError(f"input_type must be 'scores' or 'indices', got {input_type!r}")

    n_steps = topk_idx.shape[0]
    topk_idx = topk_idx.reshape(n_steps, -1, topk_idx.shape[-1])  # (S, n_tokens, k)
    n_tokens = topk_idx.shape[1]
    # deviation from upstream: device fix -- same class of bug as
    # expert_coactivation_matrix's `membership` and expert_mass_
    # distribution's `mass` tensors in this file.
    onehot = torch.zeros(n_steps, n_tokens, n_experts, dtype=torch.float32, device=topk_idx.device)
    onehot.scatter_(2, topk_idx.long(), 1.0)
    intersection = torch.einsum("ane,bne->abn", onehot, onehot)  # (S, S, n_tokens)
    overlap = intersection.mean(dim=-1) / k
    return overlap if meta is None else {"overlap": overlap, "meta": meta}


# ---------------------------------------------------------------------------
# "Silent" (under-specialization) collapse diagnostics -- sources:
# Li, Jin, Han, Liu (2026-06-09) "The death of
# z-loss in modern LLMs"; Jin, Li, Han, Liu, Wen, Zhao, Yin, Huang
# (2026-07-27) "Saving Lower Layer MoE Experts".
#
# Why these exist alongside expert_load_distribution/dead_expert_count:
# every count-based statistic above (load, dead-expert counts,
# entropy_ratio_vs_uniform in expert_collapse.py) can look perfectly healthy
# -- balanced token counts, no dead experts, near-uniform entropy -- while
# several experts have functionally collapsed into near-duplicates that
# happen to split traffic evenly. The source papers document this "silent"/
# under-specialization failure mode directly: token counts stay balanced
# (MaxVio_count near zero) while the actual output CONTRIBUTION mass
# concentrates sharply (MaxVio_mass far from zero) -- a gap invisible to any
# count-only metric. `topk_margin` is the complementary per-decision signal:
# a near-zero margin means the top-1 and k-th selected expert were nearly
# tied in the router's own scoring, i.e. functionally interchangeable for
# that token, regardless of how balanced their aggregate counts are.
#
# Scale caveat (must travel with any use of these on this project's data):
# the source papers measure 512-768-expert, 50-60B-parameter sparse MoE
# models. This project's seedvar models have 8 experts and are 0.2-0.5B
# parameters -- three orders of magnitude smaller. We are borrowing the
# PAPERS' DIAGNOSTIC DEFINITIONS, not their reported findings or thresholds
# -- nothing here implies those papers' conclusions transfer to an 8-expert
# model, that is a separate, unverified question a report using these
# numbers must state explicitly.
# ---------------------------------------------------------------------------


def topk_margin(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Per-token margin between the top-1 and k-th (weakest selected) router
    score: `margin_t = r_{t,top1} - r_{t,topk}`.

    Definition (Li et al. 2026-06-09, "The death of z-loss in modern LLMs"):
    the raw difference between the strongest and k-th-strongest score among
    the values the router actually selected. Computed directly on whatever
    `scores` is passed -- **no internal softmax/normalization is applied**,
    unlike most of this module's other functions (which take an explicit
    `input_type`). This is deliberate: the source paper computes margin on
    raw pre-softmax logits, and that is the value comparable to their
    reported numbers. Passing post-softmax probabilities instead computes a
    different, also-legitimate quantity -- a probability-space margin -- NOT
    the one the paper defines; which scale you pass is the caller's
    responsibility and changes the number's meaning, not just its units
    (margin is not scale-invariant under softmax, a strictly monotonic but
    nonlinear transform).

    A small margin means the top-1 and k-th selected expert were nearly tied
    in the router's own scoring: the router had little preference among the
    k experts it picked, so they are functionally near-interchangeable for
    this token's decision -- redundant capacity, even if their aggregate
    token counts (`expert_load_distribution`) look perfectly balanced.

    ⚠️ **That reading holds only at k == 2.** For k > 2 this quantity spans
    the WHOLE selected block, and a router that spreads a token across its k
    experts roughly equally -- which is what top-k routing is designed to do
    -- produces a SMALL margin. Measured on this project's pipeline: block
    specialization gives rho 0.248 at (E=64, k=8) and 0.084 at (E=8, k=2),
    while peaked specialization gives 1.100 and 1.567; and on OLMoE's real
    scores (E=64, k=8) a whole-batch token shuffle returns rho_cal to 0.9997
    while the real scores sit at 0.678, confirming the low value is genuine
    routing structure rather than a measurement artefact. So at k > 2 a small
    margin indicates BLOCK specialization, not its absence, and the
    direction of any "under-specialization" reading inverts. (This was
    predicted analytically, estimating rho ~ 0.2-0.3 for perfect block
    specialization.)

    Known-answer check (see tests): all experts tied (constant scores) ->
    margin = 0 for every token; one expert's score offset by delta above a
    tied field of k -> margin = delta.

    Args:
        scores: (..., n_experts) tensor, e.g. raw router logits.
        k: how many top scores define "selected" (this project's `top_k`).

    Returns:
        (...,) tensor, one margin value per token (last dim reduced away).
    """
    top_scores, _ = scores.topk(k, dim=-1)
    return top_scores[..., 0] - top_scores[..., -1]


def expert_mass_distribution(
    topk_idx: torch.Tensor, topk_weights: torch.Tensor, n_experts: int
) -> torch.Tensor:
    """Per-expert summed COMBINE-WEIGHT MASS (not token/slot count).

    Definition: for each (token, k-slot) pair, `topk_weights` is the
    combine weight (renormalized top-k softmax probability, as returned by
    e.g. `LoopedTrace.topk()` -- `w / w.sum(dim=-1, keepdim=True)`, summing
    to 1 per token) that will actually scale that expert's contribution to
    the token's output. This function sums those weights per expert:

        mass(e) = sum over (token, k-slot) pairs assigned to e of their combine weight

    This is the quantity `expert_load_distribution`'s `load` does NOT
    capture: `load(e)` counts assignment SLOTS regardless of how much each
    one actually contributes; `mass(e)` weights each slot by its actual
    contribution. Two experts can have identical `load` (same token counts)
    while one has much higher `mass` (its selections consistently get
    higher combine weight, i.e. actually matter more to the output) --
    exactly the gap the source paper (Jin et al. 2026-07-27) found aux-loss
    balancing hides: token counts stayed flat while mass concentration grew
    unchecked.

    Args:
        topk_idx: integer tensor of expert ids, any shape (..., k).
        topk_weights: combine weights, same shape as `topk_idx`.
        n_experts: total number of experts.

    Returns:
        (n_experts,) tensor of summed mass (floats sum to `topk_weights.sum()`,
        NOT to `topk_idx.numel()` like `load` does -- these are different
        units, do not compare a `mass` vector to a `load` vector directly;
        use `maxvio` on each separately, see that function).
    """
    flat_idx = topk_idx.reshape(-1).long()
    flat_w = topk_weights.reshape(-1).to(torch.float32)
    # deviation from upstream: device fix -- without device=, this landed
    # on CPU regardless of where topk_idx/topk_weights actually lived, and
    # scatter_add_ then raised "Expected all tensors to be on the same
    # device" the first time this ran against a real CUDA model (never
    # triggered by this project's CPU-only toy tests, since all tensors
    # happened to already be on CPU there).
    mass = torch.zeros(n_experts, dtype=torch.float32, device=flat_idx.device)
    mass.scatter_add_(0, flat_idx, flat_w)
    return mass


def maxvio(per_expert_quantity: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """MaxVio (Maximum Violation) load-imbalance statistic:

        MaxVio = (max_i(X_i) - mean(X)) / mean(X)

    where `X` is some per-expert aggregated quantity (the source papers use
    this on token COUNTS, calling the result `MaxVio_count`; applying the
    identical formula to combine-weight MASS instead gives `MaxVio_mass` --
    same formula, two different `X`, see `expert_load_distribution` (count)
    vs `expert_mass_distribution` (mass), and this module's `maxvio` doesn't
    care which one you feed it -- **callers must record which was used**
    (an `aggregated_quantity` field, not just a bare number) since the two
    are not comparable and mixing them up silently is exactly the kind of
    error to be prevented (same shape as the
    `0.125` vs `0.25` share-definition mixup this project already hit once).

    0 means every expert got exactly the mean quantity (perfectly
    balanced); grows without bound as the busiest expert pulls further
    above the mean.

    Known-answer check (see tests): uniform `X` -> MaxVio = 0; one expert at
    2x the others' (equal) value -> MaxVio = (2m - m_bar)/m_bar for the
    corresponding mean m_bar, verified numerically in tests rather than
    algebra here to keep this docstring an assertion, not a derivation.

    Args:
        per_expert_quantity: (n_experts,) tensor (count OR mass -- caller's
            choice, this function is agnostic).
        eps: mean-clamp floor to avoid divide-by-zero on an all-zero input
            (e.g. a domain slice with zero assignment slots).

    Returns:
        0-d tensor.
    """
    mean = per_expert_quantity.mean()
    return (per_expert_quantity.max() - mean) / mean.clamp_min(eps)

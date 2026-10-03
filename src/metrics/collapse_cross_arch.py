"""Cross-architecture MoE expert-collapse metrics.

Two router behaviours, each reported as a number that means the same thing at
different `E` and `k` so that architectures can be compared directly:

* **Metric A, load concentration** -- replaces MaxVio. Built from combine-weight
  MASS, never token counts: a count share is capped at `1/k`, which puts `k`
  back into the metric, and mass is what actually decides how much gradient an
  expert receives.
* **Metric B, selected-weight concentration** -- replaces the top-k margin.
  Built from the entropy of the weights WITHIN the selected set, so it does not
  involve `E`, and divided by `log k`, so it does not involve `k`.

Both are 0 when the phenomenon is absent. Neither says anything about whether
experts learned different functions; they describe router behaviour only,
which is why metric B is named for what it measures -- concentration of
weight among the selected experts -- rather than for "under-specialization".

## The one place this file deviates from the spec, and why it reports both

The spec defines `L = 1 - E_eff/E`, whose range is `[0, 1-1/E]` and whose value
is the *fraction of experts effectively starved*: `m` of `E` experts sharing all
the mass reads `1 - m/E` at every `E`. A version divided by `(1 - 1/E)` was
also requested so the range is exactly `[0, 1]`.

Those two requirements are incompatible, and the divisor is the reason: it
contains `E`, so dividing by it makes the result depend on `E` again. Half the
experts starved reads 0.5000 under the spec at any `E`, but 0.5714 at `E=8` and
0.5010 at `E=512` once scaled -- and this project's own archives are `E=16`
(0.5333) and `E=64` (0.5079), the exact comparison the spec exists to make.

So both are returned under names that say which is which. `*_starved_fraction`
is the spec's quantity and the one that is comparable across architectures;
`*_unit_scaled` has the tidy range and must only be compared within one `E`.
Choosing silently between them was the one option not available: a consumer
reading either name gets a number whose meaning is stated.
"""

from __future__ import annotations

from typing import Any

import numpy as np


#: The spec's floor for the indifferent-router reference. Enforced rather than
#: documented: a smaller draw still returns a number, and that number is wrong
#: by an amount nobody can see from the output -- `C_adj` is a difference
#: against it, so a noisy reference moves every adjusted value.
C_NULL_MIN_TOKENS = 100_000


def _effective_experts(q: np.ndarray) -> tuple[float, float]:
    """`(E_eff_2, E_eff_inf)` for a share vector `q` that sums to 1.

    Both are "how many experts working uniformly would look like this": order 2
    weighs the whole distribution, order infinity looks only at the largest
    share. Neither is defined for an all-zero `q`, which cannot occur here --
    tokens with no mass are dropped before this is called.
    """
    return 1.0 / float(np.sum(q**2)), 1.0 / float(np.max(q))


def _concentration(e_eff: float, n_experts: int) -> tuple[float, float]:
    """`(starved_fraction, unit_scaled)` -- see this module's docstring.

    Returned together, always, so that neither can be recorded without the
    other being available to check it against.
    """
    starved = 1.0 - e_eff / n_experts
    return starved, starved / (1.0 - 1.0 / n_experts)


def null_confidence(
    *,
    n_experts: int,
    top_k: int,
    sigma: float,
    n_tokens: int,
    seed: int,
) -> float:
    """`C` for a router with NO preference at all: iid `N(0, sigma^2)` logits.

    Metric B does not read 0 for an indifferent router. Picking the largest `k`
    of `E` independent draws leaves a gap between them by construction, and that
    gap grows with `E` and shrinks with `k`, so an untrained router scores some
    positive `C` that is a property of the arithmetic rather than of the model.
    Subtracting this reference is what makes `C_adj` comparable across
    architectures.

    Depends on `E`, `k` and `sigma` alone -- no model, no data. `sigma` is the
    empirical spread of the real logits, so the reference is matched to the cell
    it will be subtracted from; using a fixed sigma would compare a cell against
    a router whose scale it does not have.
    """
    if n_tokens < C_NULL_MIN_TOKENS:
        raise ValueError(
            f"n_tokens={n_tokens} is below the spec's floor of "
            f"{C_NULL_MIN_TOKENS} for the null reference. A smaller draw does "
            "not fail, it returns a noisy C_null -- and since C_adj is measured "
            "against it, that noise lands in every adjusted value with nothing "
            "in the output to show it."
        )
    if top_k < 2:
        raise ValueError(
            f"top_k={top_k}: the within-selection entropy is divided by log(k), "
            "which is 0 at k=1. The quantity is undefined for this "
            "configuration rather than zero, so callers must skip it."
        )
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, sigma, size=(int(n_tokens), int(n_experts)))
    # Only the k largest matter, and only their differences.
    top = np.sort(logits, axis=1)[:, -top_k:]
    top = top - top.max(axis=1, keepdims=True)  # shift-invariant; avoids overflow
    w = np.exp(top)
    s = w / w.sum(axis=1, keepdims=True)
    s_log = np.zeros_like(s)
    nz = s > 0
    s_log[nz] = s[nz] * np.log(s[nz])
    entropy = -np.sum(s_log, axis=1)
    return float(np.mean(1.0 - entropy / np.log(top_k)))


def collapse_metrics(
    *,
    topk_idx: np.ndarray,
    topk_weight: np.ndarray,
    router_logits: np.ndarray,
    n_experts: int,
    c_null_tokens: int,
    c_null_seed: int,
) -> dict[str, Any]:
    """Both metrics for ONE `(layer, loop_step)` cell.

    Args:
        topk_idx: `(N, k)` selected expert ids.
        topk_weight: `(N, k)` combine weights actually used in the forward pass.
            Need not sum to 1 per token; every quantity here normalises first,
            so a renormalised and a raw router give the same answer.
        router_logits: `(N, E)` raw pre-softmax router output. Used only for
            `sigma_hat` and the reference `margin_mean`, never for the metrics
            themselves -- those come from the weights the forward pass really
            applied, which is the point of the spec.
        n_experts: `E` for this layer, routed experts only.
        c_null_tokens: draws for the indifferent-router reference.
        c_null_seed: seed for that reference, recorded so it can be recomputed.

    Every argument is required. There is no default `E`, no default sample
    size, and no default seed: each of them changes the numbers, and a default
    would let two runs differ without their command lines differing.

    Reduction order is fixed and reported: per-token normalisation first, then
    the mean over tokens. Averaging cells or layers together is deliberately
    NOT done here -- how to aggregate is undecided, and doing it
    early would destroy the split this measurement exists to show.
    """
    topk_idx = np.asarray(topk_idx)
    weight = np.asarray(topk_weight, dtype=np.float64)
    logits = np.asarray(router_logits, dtype=np.float64)
    if topk_idx.shape != weight.shape:
        raise ValueError(
            f"topk_idx {topk_idx.shape} and topk_weight {weight.shape} must have "
            "the same shape; they index the same selections."
        )
    if logits.shape[0] != topk_idx.shape[0]:
        raise ValueError(
            f"router_logits has {logits.shape[0]} tokens but topk_idx has "
            f"{topk_idx.shape[0]}; they must describe the same tokens."
        )
    if logits.shape[1] != n_experts:
        raise ValueError(
            f"router_logits has {logits.shape[1]} experts but n_experts={n_experts}."
        )
    if c_null_tokens < C_NULL_MIN_TOKENS:
        raise ValueError(
            f"c_null_tokens={c_null_tokens} is below the spec's floor of "
            f"{C_NULL_MIN_TOKENS}. Checked here as well as in "
            "`null_confidence`, because at k=1 the reference is never drawn -- "
            "so a bad value would pass silently on those cells and only "
            "surface on the next configuration."
        )
    n_total, top_k = topk_idx.shape

    # Tokens whose weights are all zero carry no mass and would divide by zero.
    # Counted and reported rather than dropped quietly: a run where many tokens
    # are in this state is a finding about the router, not a detail.
    row_sum = weight.sum(axis=1)
    valid = row_sum > 0
    n_zero_mass = int((~valid).sum())
    weight, idx, row_sum = weight[valid], topk_idx[valid], row_sum[valid]
    sel_logits = np.take_along_axis(logits[valid], idx, axis=1)
    n_tokens = int(weight.shape[0])
    if n_tokens == 0:
        raise ValueError(
            "every token had zero combine mass; there is nothing to measure. "
            "This is a broken forward pass, not a collapsed router."
        )

    # --- Metric A: load concentration -------------------------------------
    w_norm = weight / row_sum[:, None]  # each token contributes exactly 1 unit
    q_mass = np.bincount(idx.ravel(), weights=w_norm.ravel(), minlength=n_experts) / n_tokens
    e_eff_2, e_eff_inf = _effective_experts(q_mass)
    l2_starved, l2_scaled = _concentration(e_eff_2, n_experts)
    linf_starved, linf_scaled = _concentration(e_eff_inf, n_experts)

    # The count version exists only to be DIFFERENCED against the mass version.
    # On its own it is not comparable across architectures (its share is capped
    # at 1/k), but "counts look balanced while mass is concentrated" is the
    # blog's central observation, and that shows up as the gap between the two.
    q_count = np.bincount(idx.ravel(), minlength=n_experts).astype(np.float64) / (
        n_tokens * top_k
    )
    e_eff_2_count, e_eff_inf_count = _effective_experts(q_count)
    l2_starved_count, _ = _concentration(e_eff_2_count, n_experts)
    linf_starved_count, _ = _concentration(e_eff_inf_count, n_experts)

    out: dict[str, Any] = {
        "N": n_tokens,
        "n_zero_mass_tokens": n_zero_mass,
        "E": int(n_experts),
        "k": int(top_k),
        # --- A, mass (the comparable ones) ---
        "L2_starved_fraction": l2_starved,
        "Linf_starved_fraction": linf_starved,
        "L2_unit_scaled": l2_scaled,
        "Linf_unit_scaled": linf_scaled,
        "E_eff_2": e_eff_2,
        "E_eff_inf": e_eff_inf,
        # --- A, count (same architecture only) ---
        "L2_starved_fraction_count": l2_starved_count,
        "Linf_starved_fraction_count": linf_starved_count,
        # Positive means mass is more concentrated than counts are -- the
        # "counts balanced, mass collapsed" pattern. Only meaningful within one
        # architecture, because the count version's ceiling depends on k.
        "L2_mass_minus_count": l2_starved - l2_starved_count,
        "maxvio_mass": float(n_experts * np.max(q_mass) - 1.0),
        "maxvio_count": float(n_experts * np.max(q_count) - 1.0),
        "sigma_hat": float(np.std(logits[valid])),
    }

    # --- Metric B: concentration of weight among the selected --------------
    if top_k < 2:
        # Not zero, and not an error: at k=1 there is one selected expert and
        # log k = 0, so the quantity has no definition. Recorded as a stated
        # status so a consumer sees "this configuration has no such number"
        # rather than finding the field missing and guessing why.
        out["metric_b_status"] = "not_applicable_k_eq_1"
        return out

    s = w_norm  # already normalised within the selected set: others are zero
    # log is evaluated only where s > 0. `np.where` alone would still compute
    # log(0) first and raise a warning on every call, and a warning that fires
    # on correct input is one people learn to ignore.
    s_log = np.zeros_like(s)
    nz = s > 0
    s_log[nz] = s[nz] * np.log(s[nz])
    entropy = -np.sum(s_log, axis=1)
    c_t = 1.0 - entropy / np.log(top_k)
    c_mean = float(np.mean(c_t))
    c_null = null_confidence(
        n_experts=n_experts,
        top_k=top_k,
        sigma=out["sigma_hat"],
        n_tokens=c_null_tokens,
        seed=c_null_seed,
    )
    out.update(
        {
            "metric_b_status": "ok",
            "C_mean": c_mean,
            "C_p25": float(np.percentile(c_t, 25)),
            "C_p50": float(np.percentile(c_t, 50)),
            "C_p75": float(np.percentile(c_t, 75)),
            "k_eff_mean": float(np.mean(np.exp(entropy))),
            "C_null": c_null,
            "C_adj": (c_mean - c_null) / (1.0 - c_null),
            "c_null_tokens": int(c_null_tokens),
            "c_null_seed": int(c_null_seed),
            # The blog's original quantity, kept for same-architecture
            # cross-checking only: it reads the largest and smallest selected
            # logit and ignores the ones between, so at k>2 it can rank a tie
            # for first above a single dominant expert.
            "margin_mean": float(np.mean(sel_logits.max(axis=1) - sel_logits.min(axis=1))),
        }
    )
    out.update(raw_score_stats(logits[valid], top_k=top_k))
    return out


#: Bin count for the per-position score histogram. Edges are data-derived and
#: written out beside the counts, because a fixed range would clip: the scale of
#: the raw scores is exactly what these fields exist to show.
LOGIT_HIST_BINS = 64


def raw_score_stats(logits: np.ndarray, *, top_k: int) -> dict[str, Any]:
    """Statistics of the RAW router scores -- nothing normalised by scale.

    Every quantity here is a difference
    or a probability read straight off the scores; no sigma, no synthetic
    reference. The scale itself is the object of interest, so dividing it out
    would remove the measurement.

    * `margin_raw` -- 1st minus 2nd. NOT the same as `margin_mean` above, which
      is 1st minus k-th over the SELECTED set: the two coincide only at k=2.
    * `decision_gap` -- k-th minus (k+1)-th: how close this token came to
      selecting a different expert. **This is a boundary quantity and the (k+1)-th
      score has never existed in any stored artifact** -- it is computable here
      only because the full E-dimensional scores are in hand during the forward.
    * `top1_prob` -- softmax over ALL E experts, largest entry. Softmax is
      scale-dependent, so this keeps the scale information rather than removing it.

    Percentiles rather than only means: a mean gap says nothing about how many
    tokens sat near the boundary, which is the question these fields are for.
    """
    x = np.asarray(logits, dtype=np.float64)
    n_experts = x.shape[1]
    # Partial sort is enough and is O(nE) rather than O(nE log E); `top_k + 1`
    # entries are needed because the decision gap straddles the boundary.
    need = min(top_k + 1, n_experts)
    part = np.partition(x, -need, axis=1)[:, -need:]
    part.sort(axis=1)              # ascending within the kept slice
    top1 = part[:, -1]
    out: dict[str, Any] = {
        "margin_raw_mean": float(np.mean(top1 - part[:, -2])) if n_experts >= 2 else None,
        "margin_raw_p01": float(np.percentile(top1 - part[:, -2], 1)) if n_experts >= 2 else None,
    }
    if n_experts > top_k:
        gap = part[:, -top_k] - part[:, -(top_k + 1)]
        out.update({
            "decision_gap_mean": float(np.mean(gap)),
            "decision_gap_min": float(np.min(gap)),
            "decision_gap_p01": float(np.percentile(gap, 1)),
            "decision_gap_p50": float(np.percentile(gap, 50)),
            "decision_gap_status": "ok",
        })
    else:
        # k == E: every expert is selected, so there is no boundary to measure.
        # Recorded as a stated status, never as 0 -- a zero would read as "every
        # token was on a knife edge", the opposite of what this configuration is.
        out.update({"decision_gap_status": "not_applicable_k_eq_E"})
    # Softmax over all E, shift-invariant form.
    z = x - x.max(axis=1, keepdims=True)
    w = np.exp(z)
    p1 = (w / w.sum(axis=1, keepdims=True)).max(axis=1)
    out.update({
        "top1_prob_mean": float(np.mean(p1)),
        "top1_prob_p01": float(np.percentile(p1, 1)),
        "top1_prob_p50": float(np.percentile(p1, 50)),
        "top1_prob_p99": float(np.percentile(p1, 99)),
    })
    counts, edges = np.histogram(x, bins=LOGIT_HIST_BINS)
    out.update({
        "logit_hist_counts": [int(c) for c in counts],
        "logit_hist_edges": [float(e) for e in edges],
        "logit_hist_bins": LOGIT_HIST_BINS,
    })
    return out

"""Physical-layer metrics for looped MoE.

Everything here is computed offline from the routing dump -- no model, no
forward pass.

## The distinction this file exists for

A looped model sends every token through the same physical layer `R` times.
v1 measured each `(layer, pass)` cell on its own, which answers "at this depth
position, in this single forward, is the load balanced". It does not answer
"is this set of physical expert parameters being used". Those differ, and they
can point opposite ways: with `E=4`, a layer that uses experts {1,2} uniformly
on pass 1 and {3,4} uniformly on pass 2 reads `L=0.5` on every pass while
every expert is fully used, so the physical-layer answer is `0`.

So both granularities are reported, and **the granularity is part of every
field name**. `_eff` averages the per-pass values; `_phys` pools the passes
before measuring. For `R=1` they are identically equal, which is asserted in
the tests rather than assumed.

## What is NOT a collapse metric here

`D`, `D_E` and `Cov` describe whether a token meets different experts across
passes. A collapsed model where every token uses the same
6 experts and a well-specialised one read the SAME `D`. They are reported
because they answer a different question, and the field docs say so.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _effective_experts(q: np.ndarray) -> tuple[float, float]:
    """`(E_eff_2, E_eff_inf)` for a share vector summing to 1."""
    return 1.0 / float(np.sum(q**2)), 1.0 / float(np.max(q))


def _shares(idx: np.ndarray, weight: np.ndarray, n_experts: int) -> tuple[np.ndarray, np.ndarray]:
    """`(mass share, count share)` for one pass, each summing to 1.

    Per-token normalisation first, so every token contributes exactly one unit
    of mass regardless of how the router scaled its weights.
    """
    w = np.asarray(weight, dtype=np.float64)
    row = w.sum(axis=1)
    keep = row > 0
    w, ids, row = w[keep], np.asarray(idx)[keep], row[keep]
    n = w.shape[0]
    w_norm = w / row[:, None]
    mass = np.bincount(ids.ravel(), weights=w_norm.ravel(), minlength=n_experts) / n
    count = np.bincount(ids.ravel(), minlength=n_experts).astype(np.float64) / (n * ids.shape[1])
    return mass, count


def load_by_granularity(
    passes: "list[tuple[np.ndarray, np.ndarray]]", *, n_experts: int
) -> dict[str, Any]:
    """Load concentration and utilisation at both granularities for ONE physical layer.

    `passes` is `[(topk_idx, topk_weight), ...]` in pass order -- every pass of
    this one physical layer.

    `_phys` averages the per-pass SHARES and then measures. It does
    not average the per-pass metric values: `E_eff` is not linear, so averaging
    `L` per pass is the `_eff` quantity and gives a different answer. Both are
    returned so nobody has to reconstruct which was which.
    """
    if not passes:
        raise ValueError("a physical layer needs at least one pass")
    mass = [];  count = []
    for idx, weight in passes:
        m, c = _shares(idx, weight, n_experts)
        mass.append(m); count.append(c)
    mass_arr = np.stack(mass); count_arr = np.stack(count)
    n_passes = len(passes)

    def block(q: np.ndarray, tag: str) -> dict[str, float]:
        e2, einf = _effective_experts(q)
        return {
            f"L2_{tag}": 1.0 - e2 / n_experts,
            f"Linf_{tag}": 1.0 - einf / n_experts,
            f"L2_{tag}_unit_scaled": (1.0 - e2 / n_experts) / (1.0 - 1.0 / n_experts),
            f"Linf_{tag}_unit_scaled": (1.0 - einf / n_experts) / (1.0 - 1.0 / n_experts),
            f"U2_{tag}": e2 / n_experts,
            f"Uinf_{tag}": einf / n_experts,
            f"E_eff_2_{tag}": e2,
            f"E_eff_inf_{tag}": einf,
        }

    out: dict[str, Any] = {"n_passes": n_passes, "E": int(n_experts)}
    # phys: pool the passes, then measure.
    out.update(block(mass_arr.mean(axis=0), "phys"))
    out.update(
        {
            "L2_count_phys": 1.0 - _effective_experts(count_arr.mean(axis=0))[0] / n_experts,
            "E_eff_2_count_phys": _effective_experts(count_arr.mean(axis=0))[0],
        }
    )
    # eff: measure each pass, then average the values.
    per_pass = [block(m, "eff") for m in mass_arr]
    for key in per_pass[0]:
        out[key] = float(np.mean([p[key] for p in per_pass]))
    out["L2_count_eff"] = float(
        np.mean([1.0 - _effective_experts(c)[0] / n_experts for c in count_arr])
    )
    out["E_eff_2_count_eff"] = float(np.mean([_effective_experts(c)[0] for c in count_arr]))
    out["per_pass_L2_eff"] = [float(p["L2_eff"]) for p in per_pass]
    return out


def cross_loop_diversity(
    idx_passes: "list[np.ndarray]", *, n_experts: int
) -> dict[str, Any]:
    """Does a token meet different experts on different passes.

    NOT a collapse metric. A model where every token uses the same `k` experts
    on every pass and a model that spreads work well can read the same `D`;
    `U2_phys` is what separates them.

    `R=1` returns `not_applicable` rather than 0: the denominator
    `min(Rk, E) - k` is zero, and 0 would read as "never changes experts",
    which is a measurement a single-pass model cannot make.
    """
    n_passes = len(idx_passes)
    top_k = idx_passes[0].shape[1]
    out: dict[str, Any] = {"n_passes": n_passes, "k": int(top_k)}
    if n_passes < 2:
        out["diversity_status"] = "not_applicable_single_pass"
        return out

    n_tokens = idx_passes[0].shape[0]
    # Union size per token, via a per-token boolean occupancy of the experts.
    seen = np.zeros((n_tokens, n_experts), dtype=bool)
    rows = np.arange(n_tokens)[:, None]
    for idx in idx_passes:
        seen[rows, idx.astype(np.int64)] = True
    d_t = seen.sum(axis=1).astype(np.float64)

    denom_cross = min(n_passes * top_k, n_experts) - top_k
    out.update(
        {
            "diversity_status": "ok",
            "d_mean": float(d_t.mean()),
            "D": float(((d_t - top_k) / denom_cross).mean()),
            "D_E": float(((d_t - top_k) / (n_experts - top_k)).mean()),
            "d_min": int(d_t.min()),
            "d_max": int(d_t.max()),
        }
    )
    # Adjacent-pass overlap, reported per boundary: it says WHERE reuse
    # happens, which a single averaged number cannot.
    overlaps = []
    for a, b in zip(idx_passes, idx_passes[1:]):
        inter = np.array(
            [len(np.intersect1d(x, y, assume_unique=False)) for x, y in zip(a[:5000], b[:5000])],
            dtype=np.float64,
        )
        overlaps.append(float((inter / top_k).mean()))
    out["adjacent_overlap_by_pass"] = overlaps
    out["adjacent_overlap_tokens_sampled"] = min(5000, n_tokens)
    return out


def sequence_coverage(
    idx_passes: "list[np.ndarray]",
    *,
    n_experts: int,
    tokens_per_document: int,
    seq_len: int,
) -> dict[str, Any]:
    """Within one short sequence, how much of the layer gets used.

    `Cov_rand` is subtracted and reported because a model with more passes gets
    more draws and would win on that alone -- the comparison people want is
    against chance at the same number of draws, not the raw coverage.

    Windows are non-overlapping and never cross a document boundary: a window
    spanning two documents would mix unrelated text and answer a question about
    the corpus layout rather than about the router.
    """
    n_tokens = idx_passes[0].shape[0]
    top_k = idx_passes[0].shape[1]
    n_passes = len(idx_passes)
    if n_tokens % tokens_per_document:
        raise ValueError(
            f"{n_tokens} tokens is not a whole number of {tokens_per_document}-token "
            "documents; the document index derived from the row order would be wrong."
        )
    n_docs = n_tokens // tokens_per_document
    per_doc = tokens_per_document // seq_len  # whole windows only

    stacked = np.stack([i.astype(np.int64) for i in idx_passes])  # (R, N, k)
    # (n_windows, seq_len * k * R) expert ids, one row per window.
    keep = per_doc * seq_len
    view = stacked.reshape(n_passes, n_docs, tokens_per_document, top_k)[:, :, :keep, :]
    view = view.reshape(n_passes, n_docs * per_doc, seq_len * top_k)
    view = np.moveaxis(view, 0, 1).reshape(n_docs * per_doc, n_passes * seq_len * top_k)

    n_seq = np.array([np.unique(row).size for row in view], dtype=np.float64)
    cov = float((n_seq / n_experts).mean())
    cov_rand = 1.0 - (1.0 - top_k / n_experts) ** (seq_len * n_passes)
    return {
        "seq_len": int(seq_len),
        "n_sequences": int(n_seq.size),
        "n_draws_per_sequence": int(seq_len * n_passes * top_k),
        "Cov": cov,
        "Cov_rand": cov_rand,
        "Cov_minus_rand": cov - cov_rand,
        "n_seq_mean": float(n_seq.mean()),
    }


def choose_seq_len(n_experts: int, top_k: int) -> int:
    """`T` with `T*k ~= E`, at least 1."""
    return max(1, round(n_experts / top_k))

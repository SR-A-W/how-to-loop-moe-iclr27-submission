# ---------------------------------------------------------------------------
# VENDORED FILE — DO NOT EDIT BELOW THIS HEADER.
#
# Source: an earlier, separate diagnostics code base (read-only reference);
# original module path metrics/looped.py.
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

"""Looped-model iterate-space diagnostics (specific to looped models).

Based on the "two-scale latent dynamics" picture (Pappone, Crisostomi &
Rodolà, arXiv 2509.23314): within one looped block, consecutive loop-step
iterates form a small-scale *refinement* trajectory; across different
looped blocks, representations undergo a larger-scale *drift*. This module
gives both scales the same underlying step-norm / angle machinery, applied
along different axes by the caller.

Shape convention: every function takes a tensor whose axis `step_dim`
indexes the sequence being differenced (loop steps for step_norm /
consecutive_step_angle / second_order_diff; looped-block index for
cross_block_drift) and whose last axis is d_model. All other axes
(batch, seq position, ...) are preserved and reduced only by the `reduce`
argument.
"""

from __future__ import annotations

import torch


def _diff_along(h: torch.Tensor, step_dim: int) -> torch.Tensor:
    """h[k+1] - h[k] along step_dim, one fewer entry along that axis."""
    idx_next = [slice(None)] * h.dim()
    idx_curr = [slice(None)] * h.dim()
    step_dim = step_dim % h.dim()
    n = h.shape[step_dim]
    idx_next[step_dim] = slice(1, n)
    idx_curr[step_dim] = slice(0, n - 1)
    return h[tuple(idx_next)] - h[tuple(idx_curr)]


def _reduce(x: torch.Tensor, reduce: str, keep_dim: int) -> torch.Tensor:
    if reduce == "none":
        return x
    if reduce == "mean":
        dims = [d for d in range(x.dim()) if d != (keep_dim % x.dim())]
        return x.mean(dim=dims) if dims else x
    raise ValueError(f"reduce must be 'none' or 'mean', got {reduce!r}")


def _maybe_wrap(value, key: str, meta: object | None):
    """Wrap `value` as {key: value, "meta": meta} when meta is given.

    Shared convention across this module (and `moe.cross_loop_step_overlap`):
    `meta=None` (default) preserves the old bare-tensor return exactly, so
    existing callers/tests are unaffected. Passing a `meta` object (e.g. a
    `TraceMeta`-like dict identifying which run/loop-config/block produced
    this tensor) switches the return to a small dict that carries it along
    -- added specifically so a logged number is never separated from which
    run produced it, without forcing every caller who doesn't need that to thread
    metadata through call sites that don't have any yet.
    """
    return value if meta is None else {key: value, "meta": meta}


def step_norm(
    h: torch.Tensor,
    step_dim: int = 0,
    ord: int = 2,
    reduce: str = "mean",
    meta: object | None = None,
) -> torch.Tensor:
    """‖h_{k+1} − h_k‖ as a function of loop step k.

    Definition: for consecutive loop-step iterates h_k (all other axes --
    batch, seq position -- held fixed), compute the step vector
    ``delta_k = h_{k+1} - h_k`` and its norm ``||delta_k||_ord`` (default
    L2). `reduce="mean"` (default) averages over every axis except
    `step_dim`, giving a 1D curve over k of length (n_steps - 1);
    `reduce="none"` returns the full per-(batch, seq, ...) tensor.

    This is the "small-scale refinement magnitude" half of the two-scale
    picture: large step norms mean the loop iterate is still moving a lot
    at step k; step norms shrinking toward 0 with k is the empirical
    signature (Pappone et al.) motivating convergence-based halting.

    ★ CONFIRMED external anchor (verified by reading this
    function's implementation, not by its own docstring's claim about
    itself): with `ord=2` (the default) this is algebraically identical to
    PLT's (Pappone/Looped Transformers) published `delta^(r) =
    ‖h^(r) - h^(r-1)‖_2` -- unnormalized absolute L2, no scaling. This is
    this project's first metric with both a published-paper number AND a
    verified-identical definition to compare against (Pappone et al.'s own
    paper doesn't ship code, so every loop-dynamics metric before this had
    been "defined by us, implemented by us, verified by us" with no outside
    reference point). See `consecutive_step_angle` for the matching
    confirmation on the angle side.

    Not the same quantity as Huginn's `LatentDiffExitEvaluator` (checked
    against Huginn's source): Huginn's is
    a **relative** step size `‖Δh_k‖ / ‖h_{k+1}‖` (normalized, no eps guard,
    reduced over the token axis to one scalar per batch element for an
    online stop/go decision); this function is the **absolute**, unnormalized
    `‖Δh_k‖`, reduced over batch/seq while keeping the step axis (a curve
    over k, for diagnosis rather than a per-step decision). Different scale,
    different reduction axis, different threshold semantics -- don't compare
    a Huginn 0.03 relative threshold against this function's absolute output.

    Args:
        h: tensor with a loop-step axis at `step_dim` and last axis
            d_model, e.g. (n_steps, batch, seq, d_model).
        step_dim: which axis indexes loop steps.
        ord: norm order (default 2).
        reduce: "mean" (default, average over non-step axes) or "none".
        meta: optional run metadata; see `_maybe_wrap` docstring. When
            given, returns `{"step_norm": <tensor>, "meta": meta}` instead
            of the bare tensor.

    Returns:
        Tensor of shape (n_steps - 1,) if reduce="mean", else
        h.shape with step_dim reduced by 1 and the last (feature) dim
        removed. (Or a dict wrapping it if `meta` is given.)
    """
    delta = _diff_along(h, step_dim)
    norms = torch.linalg.vector_norm(delta, ord=ord, dim=-1)
    return _maybe_wrap(_reduce(norms, reduce, step_dim), "step_norm", meta)


def consecutive_step_angle(
    h: torch.Tensor,
    step_dim: int = 0,
    eps: float = 1e-12,
    reduce: str = "mean",
    meta: object | None = None,
) -> torch.Tensor:
    """cos(Δh_k, Δh_{k+1}): directional alignment of consecutive loop steps.

    Definition: with delta_k = h_{k+1} - h_k as in `step_norm`, compute

        angle_k = cos_sim(delta_k, delta_{k+1}) = (delta_k . delta_{k+1}) / (||delta_k|| ||delta_{k+1}||)

    Values near +1 mean the loop keeps moving in the same direction
    (monotonic, "one-directional" refinement); near -1 means it is
    oscillating/overshooting back and forth; near 0 means successive
    updates are roughly orthogonal (Pappone et al.'s reported finding for
    trained models: steps become smaller *and* more mutually orthogonal
    with more training, i.e. angle -> 0, consistent with local
    fine-grained corrections rather than a single sustained direction).

    ★ CONFIRMED external anchor (verified against implementation,
    not docstring claims): this is algebraically identical to PLT's
    published `cos(theta)^(r) = <h^(r)-h^(r-1), h^(r-1)-h^(r-2)> /
    (||...|| ||...||)` -- both compare consecutive first-difference
    vectors. See `step_norm`'s matching confirmation for `delta^(r)`; the
    two together are this project's first external, published-number
    anchor for its loop-dynamics metrics (see that docstring for why that
    matters).

    ★ Not the same quantity as Huginn's `CosineExitEvaluator` (checked
    against Huginn's source; "same name, different thing" -- deliberately
    not reconciled, both kept): Huginn
    compares the two **raw latent states** `cos(h_k, h_{k+1})`, which stays
    close to 1 whenever the latent has a large iteration-invariant "DC
    component", regardless of whether the *update direction* is stabilizing
    -- it answers "will the output still change, can we stop yet", an
    online engineering question. This function compares the two **step
    (first-difference) vectors** `cos(Δh_{k-1,k}, Δh_{k,k+1})`, which
    subtracts that shared component out and is what actually tracks Pappone
    et al.'s "do refinement steps become mutually orthogonal with more
    training" question. Not an implementation disagreement -- do not use
    one to "validate" the other, and do not compare their numeric ranges.

    Args:
        h: tensor with a loop-step axis at `step_dim` and last axis
            d_model, e.g. (n_steps, batch, seq, d_model). Needs at least 3
            steps along `step_dim` to produce one angle.
        step_dim: which axis indexes loop steps.
        eps: norm floor.
        reduce: "mean" (default) or "none".
        meta: optional run metadata; see `_maybe_wrap` docstring.

    Returns:
        Tensor of shape (n_steps - 2,) if reduce="mean", else per-sample.
    """
    delta = _diff_along(h, step_dim)
    step_dim_pos = step_dim % h.dim()
    idx_next = [slice(None)] * delta.dim()
    idx_curr = [slice(None)] * delta.dim()
    n = delta.shape[step_dim_pos]
    idx_next[step_dim_pos] = slice(1, n)
    idx_curr[step_dim_pos] = slice(0, n - 1)
    d_next = delta[tuple(idx_next)]
    d_curr = delta[tuple(idx_curr)]
    num = (d_curr * d_next).sum(dim=-1)
    denom = d_curr.norm(dim=-1).clamp_min(eps) * d_next.norm(dim=-1).clamp_min(eps)
    cos = num / denom
    return _maybe_wrap(_reduce(cos, reduce, step_dim_pos), "angle", meta)


def raw_state_cosine_similarity(
    h: torch.Tensor,
    step_dim: int = 0,
    eps: float = 1e-12,
    reduce: str = "mean",
    meta: object | None = None,
) -> torch.Tensor:
    """cos(h_k, h_{k+1}): how similar are RAW consecutive-step states.

    Definition: ``cos_k = <h_k, h_{k+1}> / (||h_k|| ||h_{k+1}||)`` on the
    undifferenced state itself (not the step/first-difference vector -- see
    `consecutive_step_angle` for that, a different quantity: this measures
    whether the state itself changed, that measures whether the *direction
    of change* is consistent across steps).

    Why this exists (reinjection-vs-no-reinjection experiment): the specific
    literature claim under test is that per-step input re-injection lets a constant `h_pre` dominate the residual stream
    and "suppress iteration-to-iteration differentiation" -- i.e. makes
    `h_{k+1}` look like `h_k`. That is a direct claim about the RAW state's
    similarity across consecutive steps, not about whether the refinement
    *direction* stays consistent (which is what `consecutive_step_angle`
    tracks -- using that function for this question would answer a related
    but different thing). Values near +1 mean consecutive states are nearly
    identical (low differentiation); values further from 1 (toward 0 or
    negative) mean more differentiation.

    No `latent_norm_state` gate (unlike `evaluators.huginn_raw_state_cosine`,
    the Huginn-formula-specific sibling of this function): this is a plain
    cosine similarity between whatever tensor is passed in, deliberately
    generic so the exact same call works on raw residual-stream states from
    every model family in this project (Ouro/LoopLM/Raven) without first
    requiring each one's final-norm weight to be available -- the
    reinjection question is about the residual stream itself, which this
    is. If you specifically need Huginn's own post-norm exit-evaluator
    formula (their `CosineExitEvaluator`), use
    `evaluators.huginn_raw_state_cosine` instead, not this.

    Args:
        h: tensor with a loop-step axis at `step_dim` and last axis
            d_model, e.g. (n_steps, batch, seq, d_model).
        step_dim: which axis indexes loop steps.
        eps: norm-clamp floor.
        reduce: "mean" (default, average over non-step axes) or "none".
        meta: optional run metadata; see `_maybe_wrap` docstring.

    Returns:
        Tensor of shape (n_steps - 1,) if reduce="mean", else per-sample
        (h.shape with step_dim reduced by 1 and the feature dim removed).
    """
    step_dim_pos = step_dim % h.dim()
    n = h.shape[step_dim_pos]
    idx_next = [slice(None)] * h.dim()
    idx_curr = [slice(None)] * h.dim()
    idx_next[step_dim_pos] = slice(1, n)
    idx_curr[step_dim_pos] = slice(0, n - 1)
    h_next = h[tuple(idx_next)]
    h_curr = h[tuple(idx_curr)]
    num = (h_curr * h_next).sum(dim=-1)
    denom = h_curr.norm(dim=-1).clamp_min(eps) * h_next.norm(dim=-1).clamp_min(eps)
    cos = num / denom
    return _maybe_wrap(_reduce(cos, reduce, step_dim_pos), "raw_state_cosine_similarity", meta)


def relative_step_norm(
    h: torch.Tensor,
    step_dim: int = 0,
    eps: float = 1e-12,
    reduce: str = "mean",
    meta: object | None = None,
) -> torch.Tensor:
    """‖h_{k+1} − h_k‖ / ‖h_{k+1}‖: normalized (unitless) step size.

    Companion to `raw_state_cosine_similarity` for the same reinjection
    experiment -- magnitude side rather than direction side of
    "how much did the state change this step". Unlike `step_norm` (this
    module's absolute, unnormalized ``‖Δh_k‖``), dividing by ``‖h_{k+1}‖``
    makes this comparable across model families with different d_model /
    activation scale, which an absolute norm is not.

    Definition matches the numerator/denominator convention of
    `evaluators.huginn_latent_diff_relative` (denominator is the LATER
    step's norm, not the earlier one's) -- but, like
    `raw_state_cosine_similarity` above, has no `latent_norm_state` gate:
    this is the same relative-step-size quantity computed on whatever raw
    tensor is passed in, so it needs no final-norm weight to run on every
    model family in this project. Use
    `evaluators.huginn_latent_diff_relative` instead if you specifically
    need Huginn's own post-norm formula.

    Args:
        h: tensor with a loop-step axis at `step_dim` and last axis
            d_model.
        step_dim: which axis indexes loop steps.
        eps: norm-clamp floor.
        reduce: "mean" (default) or "none".
        meta: optional run metadata; see `_maybe_wrap` docstring.

    Returns:
        Tensor of shape (n_steps - 1,) if reduce="mean", else per-sample.
    """
    step_dim_pos = step_dim % h.dim()
    n = h.shape[step_dim_pos]
    delta = _diff_along(h, step_dim)
    idx_next = [slice(None)] * h.dim()
    idx_next[step_dim_pos] = slice(1, n)
    h_next = h[tuple(idx_next)]
    rel = delta.norm(dim=-1) / h_next.norm(dim=-1).clamp_min(eps)
    return _maybe_wrap(_reduce(rel, reduce, step_dim_pos), "relative_step_norm", meta)


def cross_block_drift(
    h_blocks: torch.Tensor,
    block_dim: int = 0,
    ord: int = 2,
    reduce: str = "mean",
    meta: object | None = None,
) -> torch.Tensor:
    """Large-scale representation displacement across looped blocks.

    Definition: identical computation to `step_norm` (‖h_{i+1} − h_i‖),
    but the caller is expected to pass the representation *at a fixed
    point within each block's loop* (e.g. always the final loop-step
    iterate, or always the block's input) across the sequence of looped
    blocks, rather than across loop steps within one block. We give this
    its own name/docstring rather than reusing `step_norm` silently,
    because the two-scale-dynamics picture this project is built around
    treats "within-block loop movement" and "across-block movement" as
    conceptually distinct (small-scale refinement vs large-scale drift,
    Pappone et al.) even though the arithmetic is identical -- naming
    disambiguates which scale a given number refers to when both are
    logged side by side.

    Args:
        h_blocks: tensor with a block-index axis at `block_dim` and last
            axis d_model, e.g. (n_blocks, batch, seq, d_model) where each
            slice along block_dim is, e.g., the final iterate of that
            looped block.
        block_dim: which axis indexes looped blocks.
        ord: norm order (default 2).
        reduce: "mean" (default) or "none".
        meta: optional run metadata; see `_maybe_wrap` docstring. Note
            `step_norm`'s wrap key is "step_norm" even when called from
            here (same underlying computation); rewrap yourself with a
            different key if "step_norm" is confusing at a call site that
            is conceptually about cross-block drift.

    Returns:
        Tensor of shape (n_blocks - 1,) if reduce="mean", else per-sample.
    """
    return step_norm(h_blocks, step_dim=block_dim, ord=ord, reduce=reduce, meta=meta)


def second_order_diff(
    h: torch.Tensor,
    step_dim: int = 0,
    ord: int = 2,
    reduce: str = "mean",
    meta: object | None = None,
) -> torch.Tensor:
    """‖h_{k+1} − 2h_k + h_{k-1}‖: loop-iterate "acceleration".

    Definition: the discrete second derivative along the loop-step axis,

        accel_k = h_{k+1} - 2*h_k + h_{k-1}

    with its norm taken (default L2). This is the quantity Pappone et al.
    use to derive a halting rule: once acceleration stays small for two
    consecutive steps ("two-hit check", see `halting_signal`), the
    iterate's trajectory has stopped curving and further loop steps are
    unlikely to change the answer -- a cheaper, more direct signal than
    Geiping et al.'s KL-divergence-based early-exit criterion (reported
    ~580ms -> ~360ms/token from switching to this style of rule).

    Args:
        h: tensor with a loop-step axis at `step_dim` and last axis
            d_model. Needs at least 3 steps.
        step_dim: which axis indexes loop steps.
        ord: norm order (default 2).
        reduce: "mean" (default) or "none".
        meta: optional run metadata; see `_maybe_wrap` docstring.

    Returns:
        Tensor of shape (n_steps - 2,) if reduce="mean", else per-sample.
    """
    step_dim_pos = step_dim % h.dim()
    n = h.shape[step_dim_pos]

    def sl(a, b):
        idx = [slice(None)] * h.dim()
        idx[step_dim_pos] = slice(a, b)
        return h[tuple(idx)]

    accel = sl(2, n) - 2 * sl(1, n - 1) + sl(0, n - 2)
    norms = torch.linalg.vector_norm(accel, ord=ord, dim=-1)
    return _maybe_wrap(_reduce(norms, reduce, step_dim_pos), "accel", meta)


def halting_signal(
    accel_norms: torch.Tensor,
    threshold: float,
    hits_required: int = 2,
) -> torch.Tensor:
    """First loop step index at which acceleration stays below threshold.

    Definition ("two-hit check"): given a sequence of per-step acceleration
    norms along the **last** axis (e.g. the `reduce="mean"` output of
    `second_order_diff` for the 1D case, or a per-token curve of shape
    `(n_tokens, n_steps)` for the batched case), find the first index k
    such that `accel_norms[..., k], ..., accel_norms[..., k+hits_required-1]`
    are *all* below `threshold` -- i.e. `hits_required` (default 2,
    matching Pappone et al.'s "two-hit") consecutive small-acceleration
    steps, rather than a single below-threshold reading (which could be
    noise). Result is -1 (per leading-index) where no such run of
    consecutive hits exists.

    Batched: this is now vectorized over arbitrary leading dims (added
    when the exit-gate analysis needed to compare a per-token acceleration-
    based halting step against a per-token exit-gate halting step -- see
    `exit_gate.halting_agreement`). A plain 1D input (the original use
    case) still returns a 0-d tensor exactly as before.

    Implementation note (tie-safety): the first-hit index is computed via
    `(~all_hit).cumprod(dim=-1).sum(dim=-1)` rather than `argmax` on a
    boolean/float mask. `cumprod` of "not yet hit" stays 1 for every
    leading position before the first hit and drops to (and stays at) 0
    from the first hit onward, so its sum equals exactly the count of
    positions before the first hit -- i.e. the first hit's index -- with
    no dependence on `argmax`'s tie-breaking behavior for equal values.

    Args:
        accel_norms: tensor of acceleration norms, loop-step axis last.
        threshold: below this counts as a "hit".
        hits_required: consecutive hits needed to trigger halting.

    Returns:
        Integer tensor of shape `accel_norms.shape[:-1]` (0-d for a plain
        1D input): index of the first step in the qualifying run, or -1
        where none found.
    """
    below = accel_norms < threshold
    n_steps = below.shape[-1]
    if n_steps < hits_required:
        # deviation from upstream: device fix -- without device=, this
        # early-return tensor landed on CPU regardless of accel_norms'
        # actual device, which would silently mismatch whatever the caller
        # does with it next on a CUDA input (lower risk than the other three
        # spots since this is a single early-exit branch, but the same class
        # of bug).
        return torch.full(below.shape[:-1], -1, dtype=torch.int64, device=accel_norms.device)
    windows = below.unfold(-1, hits_required, 1)  # (..., n_windows, hits_required)
    all_hit = windows.all(dim=-1)  # (..., n_windows)
    any_hit = all_hit.any(dim=-1)
    first_idx = (~all_hit).to(torch.int64).cumprod(dim=-1).sum(dim=-1)
    return torch.where(any_hit, first_idx, torch.full_like(first_idx, -1))

"""Training-time metrics: JSONL logging, free-tier (metrics #1/#2/#3/#5/#6/#7)
and probe-tier (metrics #4/#8).

Every reduction here is a direct call into `_vendored_loopdiag`'s functions
(themselves verbatim copies of the earlier diagnostics code base's tested definitions) or a
documented composition of them -- this module does not reimplement any
metric formula. What it *does* own is: which raw tensors go into which
call, how the results are flattened into JSONL rows, and the JSONL file
layout/reduction ORDER (spelled out explicitly, since a metric
name alone is not a complete definition -- see each function's docstring).

## File layout

Two JSONL files, each line self-contained (parseable without reference to
any other line):

- `train_metrics.jsonl` -- free tier, written every N steps (config value,
  no default -- see `pretrain/metrics/README` / training config). Rows are
  routed by a `"kind"` field:
    - `"router"`   -- one row per (step, layer_idx, loop_step): #1/#2/#3.
    - `"residual"` -- one row per (step, kind=within_step|step_boundary,
      ...): #5.
    - `"state_cosine"` -- one row per (step, layer_idx, loop_step_pair): #6.
    - `"loss"` -- one row per step (every step, not gated by N): #7. Carries
      an OPTIONAL `grad_norm` field that appears only on torchtitan's own
      log steps -- see `loss_record`'s docstring for why it is sparse.
- `probe_metrics.jsonl` -- cheap tier, written every M steps (M > N; a
  separate, independent `no_grad` forward on the fixed probe corpus, see
  `probe_corpus.py`). Rows routed the same way:
    - `"overlap"` -- one row per (step, layer_idx): #4.
    - `"repeat_stratified_loss"` -- one row per (step, bucket): #8.

⚠️ **KNOWN GAP (not yet fixed)**:
under the FSDP2 production training path, the online trace (`on_trace_step`'s
`sink`) is not currently populated -- `"router"`/`"residual"`/`"state_cosine"`
rows are therefore ABSENT from a production run's `train_metrics.jsonl`; only
`"loss"` rows land every step. Router/residual analysis for a production run
must be recomputed OFFLINE from the trajectory checkpoints instead (density =
however many trajectory points that run has, ~14-16, not the N-step cadence
described above -- see `src/pretrain/checkpointing/schedule.py`). Do not assume
`train_metrics.jsonl` has router/residual rows when writing analysis code
against a real run; the toy/CPU path this module's own tests exercise is
unaffected (the gap is FSDP2-specific).

## Reduction-order convention (applies to every function below)

Every tensor argument's expected shape is `(batch, seq, ...)` with any
expert/vocab axis last, matching training's natural forward-pass layout
(NOT the `_vendored_loopdiag` collection-layer convention of a leading
`(loop_step, block_idx)` pair on every tensor -- this module is called once
per (loop_step, layer_idx) combination from the training loop's own hook,
so those two axes are already fixed by the call site, not present in the
tensor itself). Every function reduces `batch` and `seq` together into a
single scalar via `.mean()` unless documented otherwise -- this matches
`_vendored_loopdiag`'s own `reduce="mean"` default.
"""

from __future__ import annotations

import hashlib

from collections.abc import Sequence

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.metrics._vendored_loopdiag import looped as _looped
from src.metrics._vendored_loopdiag import moe as _moe


class MetricNotApplicable(Exception):
    """A metric is STRUCTURALLY undefined for this configuration -- not a
    failure, and not a value of zero.

    Two different situations both used to raise `ValueError` from this module:

    * the metric does not exist for this shape (top-k margin at k=1, where
      top-1 and the k-th expert are the same one, so the margin is
      identically 0), and
    * something is actually wrong (a 3-D tensor where 2-D was required,
      degenerate all-tied scores, fewer null draws than an interval needs).

    A producer that wants to record "not applicable" for the first kind had
    only `except ValueError` available, which would also swallow all of the
    second kind and label real defects as normal. Hence a distinct type.

    ⚠️ It deliberately inherits from `Exception`, NOT from `ValueError`.
    Inheriting from `ValueError` would let a single `except ValueError`
    catch it together with the genuine failures again, which is the exact
    outcome this type exists to prevent -- the inheritance would route
    around the distinction rather than express it. The cost is that an
    existing `except ValueError` no longer catches k=1; that is intended,
    because such a handler would be treating a structural fact as an error
    and should be updated to say which of the two it means.
    """


# --- controlled vocabularies for cross-model router metadata ----------------
#
# Defined here because `collector.py` is the single source every producer reads
# from: the probes of this project's own models, the open-source adapters, and
# the frontend contract.
# Spellings are the unabbreviated ones. The adapters used to write short
# forms ("aux_loss", "ffn_experts"). Artifacts written by those older adapter
# versions carry the short forms; they map one-to-one
# (aux_loss -> auxiliary_loss, ffn_experts -> feedforward_experts) and are not
# rewritten, so a reader of an old file needs this note.
#
# Two producers inventing their own spelling is not a cosmetic problem -- the criteria gate a comparison on these values being
# EQUAL, so a spelling difference reads as "different mechanism" and silently
# withdraws a comparison that was actually valid.

LOAD_BALANCING_MECHANISMS: tuple[str, ...] = (
    "auxiliary_loss",     # a load-balancing term in the loss
    "probability_bias",   # a bias added to the router PROBABILITIES
    "none",               # the model has no load-balancing mechanism
    "unknown",            # not established -- see below, this is a real answer
)
"""How this model balances expert load.

`probability_bias` is distinguished from `auxiliary_loss` because it also
flattens the combine weights, so mass-concentration metrics are not comparable
across models that differ on this field.

⚠️ `unknown` is a FULL MEMBER, not a placeholder. It must never be spelled as
`None` or `""`: "nobody filled this in" and "we looked and could not establish
it" are different states, the first a defect and the second a finding the
criteria act on (they declare no conclusion). Collapsing them into one falsy
value destroys exactly the distinction the downstream rule needs.
"""

ROUTER_MODULE_KINDS: tuple[str, ...] = (
    "feedforward_experts",  # experts are FFN blocks -- the comparable case
    "attention_experts",    # experts are attention heads/blocks
    "unknown",
)
"""What this router routes over. Same `unknown` rule as above."""


def validate_router_metadata(
    *, load_balancing_mechanism: str, router_module_kind: str
) -> dict[str, str]:
    """Check both values against their vocabularies and return them as a dict.

    Raises on anything outside the vocabulary rather than passing it through:
    an unrecognised value is a producer bug, and the artifact it would land in
    is read by a comparison that treats unequal values as "different
    mechanism". Failing here costs one run; passing it through costs a
    withdrawn comparison that looks like a finding.
    """
    for name, value, allowed in (
        ("load_balancing_mechanism", load_balancing_mechanism, LOAD_BALANCING_MECHANISMS),
        ("router_module_kind", router_module_kind, ROUTER_MODULE_KINDS),
    ):
        if value not in allowed:
            hint = (
                " (use 'unknown' -- it is a real member of this vocabulary; do "
                "not pass None or an empty string)"
                if value in (None, "")
                else ""
            )
            raise ValueError(
                f"{name}={value!r} is not one of {allowed}{hint}"
            )
    return {
        "load_balancing_mechanism": load_balancing_mechanism,
        "router_module_kind": router_module_kind,
    }


def drop_first_token_per_document(
    scores: Tensor,
    input_ids: Tensor,
    *,
    tokens_per_doc: int,
    special_ids: "Sequence[int]",
) -> tuple[Tensor, int, dict[str, Any]]:
    """Drop position 0 of every document window, uniformly, and report what went.

    ## Why position 0 and not "wherever a special token is"

    On this project's probe corpus the Llama-3 tokenizer adds a start marker at
    position 0 of every document -- but only on probe seed 0, whose window is
    the tokenizer's own truncation (the first `seq` tokens). Seeds 1-3 draw a
    random window and, measured on these 40 documents, none started at 0, so
    they carry no start marker at all.

    Excluding "wherever a special token is" would therefore shorten seed 0's
    documents to `seq - 1` and leave the others at `seq`. Every document-scoped
    operation downstream -- the within-document shuffle, the per-document
    statistics, the document bootstrap, the position bins -- reshapes a flat
    matrix with ONE `tokens_per_doc`, so ragged documents are not a slower
    path, they are not a supported one. Worse, equal-length documents are what
    makes "mean over all tokens" identical to "each document weighted 1/40",
    and percentiles have no per-document-averaged definition at all, so the
    ragged case needs a new definition of the reported quantity, not new code.

    Dropping position 0 from EVERY window keeps all documents the same length,
    changes no downstream code, and removes the seed-0/other-seeds asymmetry
    rather than merely the marker: after this, all four seeds are alike in
    having lost their first token. The cost is the ~0.2% of tokens that were
    ordinary text in seeds 1-3.

    Returns `(scores, tokens_per_doc - 1, stats)`. `stats` records
    `special_tokens_excluded` -- how many of the dropped tokens really were
    special, which is the number the criteria ask for and is NOT the number of
    documents: it is 40 for seed 0 and 0 for the others here.
    """
    n_tokens, n_experts = scores.shape
    if n_tokens % tokens_per_doc:
        raise ValueError(
            f"scores has {n_tokens} rows, not a multiple of tokens_per_doc="
            f"{tokens_per_doc}; document blocks must be equal-sized."
        )
    n_docs = n_tokens // tokens_per_doc
    ids = input_ids.reshape(-1)
    if ids.numel() != n_tokens:
        raise ValueError(
            f"input_ids has {ids.numel()} tokens but scores has {n_tokens} rows. "
            "The count of excluded special tokens would describe different text "
            "than the scores it is filed with."
        )
    if tokens_per_doc < 2:
        raise ValueError(
            f"tokens_per_doc={tokens_per_doc} leaves nothing after dropping the "
            "first token of each document."
        )

    dropped_positions = torch.arange(n_docs, device=ids.device) * tokens_per_doc
    dropped_ids = ids[dropped_positions]
    special = torch.tensor(list(special_ids), device=ids.device, dtype=dropped_ids.dtype)
    n_special = int(torch.isin(dropped_ids, special).sum()) if special.numel() else 0

    kept = scores.view(n_docs, tokens_per_doc, n_experts)[:, 1:, :].reshape(-1, n_experts)
    stats = {
        "first_token_dropped": True,
        "special_tokens_excluded": n_special,
        "n_tokens_dropped": n_docs,
        "tokens_per_doc_before": int(tokens_per_doc),
        "tokens_per_doc_after": int(tokens_per_doc - 1),
        "n_documents": int(n_docs),
        # Recorded per document so a consumer can check equal weighting holds
        # rather than assume it; all equal is the point, not an accident.
        "slots_per_document": [int(tokens_per_doc - 1)] * n_docs,
    }
    return kept, int(tokens_per_doc - 1), stats


class JsonlWriter:
    """Append-only JSONL writer. Each `write()` call is one self-contained
    line, flushed immediately -- long jobs write their output incrementally (a
    background training process has non-zero odds of dying silently; losing
    at most one unflushed line is the acceptable failure mode, not losing an
    unbounded buffer)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)  # line-buffered

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# #1/#2/#3 -- router-derived metrics for one (step, layer_idx, loop_step).
# ---------------------------------------------------------------------------


def router_layer_loop_record(
    *,
    step: int,
    layer_idx: int,
    loop_step: int,
    aggregation_scope: str = "rank_local",
    rank: int = 0,
    world_size: int = 1,
    router_logits: Tensor,
    topk_idx: Tensor,
    topk_weights: Tensor,
    n_experts: int,
    top_k: int,
) -> dict[str, Any]:
    """One `kind="router"` JSONL row: expert share/entropy/effective-count
    (#1), MaxVio_count & MaxVio_mass (#2), top-k margin (#3) -- computed
    together because they share the same input tensors (one MoE forward's
    router output for this (layer, loop_step)), not because they are one
    metric.

    Args:
        router_logits: `(batch, seq, n_experts)` raw pre-softmax router
            output for this (layer_idx, loop_step). Passed to
            `moe.topk_margin` as-is (that function requires raw logits, not
            probabilities -- see its docstring on why scale matters).
        topk_idx: `(batch, seq, top_k)` integer expert ids actually
            selected (the real routing decision, not recomputed from
            `router_logits` here -- if capacity-limited dropping is ever
            implemented upstream, `topk_idx` is where that would show up;
            recomputing top-k from `router_logits` would silently ignore
            drops, per `_vendored_loopdiag/moe.py`'s explicit warning about
            this).
        topk_weights: `(batch, seq, top_k)` renormalized combine weights
            (same convention as `LoopedTrace.topk()`: softmax weights
            re-normalized to sum to 1 over the top_k slots).
        n_experts, top_k: routing config for this layer.
        aggregation_scope, rank, world_size: whose tokens these numbers
            cover. Always `"rank_local"` today -- there is no cross-rank
            reduction, so under data parallelism this row describes one
            process's shard, roughly `1/world_size` of the step. Recorded on
            every row, single-process included, because maxvio is an extremal
            statistic and a rank-local reading is a LOWER BOUND on the global
            one.

    Reduction order (metric definitions specify the reduction order):
        1. `topk_idx`/`topk_weights` flattened over (batch, seq, top_k) into
           a flat list of (token, k-slot) assignment-slot pairs --
           `moe.expert_load_distribution` / `moe.expert_mass_distribution`.
        2. Per-expert load (count) and mass (combine-weight sum) computed
           over that flattened pool -- NOT per-token, per-expert aggregates
           already collapse the token axis.
        3. `entropy_ratio_vs_uniform`, `effective_expert_count`,
           `maxvio_count`, `maxvio_mass` are all functions of that one
           per-expert load/mass vector (n_experts-length), not of the raw
           per-token distributions.
        4. `margin_mean`/`margin_min` are the ONE exception: `topk_margin`
           is evaluated per-token first (`(batch, seq)` shape) then reduced
           by `.mean()` / `.min()` over all of batch*seq -- a genuinely
           different reduction axis from steps 1-3, which is why the two
           are computed by separate calls.

    `effective_expert_count` derivation: NOT a
    new metric. `entropy_ratio_vs_uniform = entropy_nats / log(n_experts)`
    (from `moe.router_entropy` applied to the per-expert share vector, same
    call `_vendored_loopdiag/expert_collapse.py::expert_collapse_stats`
    would make -- not vendored here since only this one derived scalar is
    needed, see that file for the fuller stats this project chose not to
    log at N-step density). `effective_expert_count := n_experts **
    entropy_ratio_vs_uniform`, algebraically `exp(entropy_nats)` -- the same
    "exp(Shannon entropy)" form `_vendored_loopdiag/representation.py`
    already uses for PLT's `erank` (effective rank of a singular-value
    spectrum), applied here to the expert-share distribution instead. A
    uniform share vector gives `effective_expert_count == n_experts`; total
    concentration on one expert gives `effective_expert_count -> 1`.
    """
    # Reduce on CPU copies. `expert_mass_distribution` accumulates with
    # `scatter_add_` and `expert_load_distribution` with `bincount`; both use
    # atomics on CUDA, so the accumulation ORDER varies between runs and two
    # identical runs disagree in the last float bits (adapter measured ~3e-6
    # relative drift in `maxvio_mass`). Nothing here is large enough to make a
    # GPU worthwhile -- these are per-cell reductions over one probe batch --
    # so the determinism is free.
    #
    # Done here rather than at each call site so this project's model probes and the
    # open-source adapters cannot drift apart on it; the vendored functions
    # themselves are left untouched.
    topk_idx = topk_idx.cpu()
    topk_weights = topk_weights.cpu()
    load, _cv = _moe.expert_load_distribution(topk_idx, n_experts)
    mass = _moe.expert_mass_distribution(topk_idx, topk_weights, n_experts)

    total = load.sum()
    share = load / total.clamp_min(1e-12)
    entropy_nats = _moe.router_entropy(share.unsqueeze(0), input_type="probs")[0]
    max_entropy_nats = math.log(n_experts)
    entropy_ratio = float((entropy_nats / max_entropy_nats).item())
    effective_expert_count = float(n_experts**entropy_ratio)

    dead = _moe.dead_expert_count(topk_idx=None, n_experts=n_experts, load=load, threshold=0)
    maxvio_count = float(_moe.maxvio(load).item())
    maxvio_mass = float(_moe.maxvio(mass).item())
    ratio = maxvio_mass / maxvio_count if maxvio_count > 1e-6 else None

    margin = _moe.topk_margin(router_logits, top_k)

    return {
        "kind": "router",
        "step": step,
        "layer_idx": layer_idx,
        "loop_step": loop_step,
        "n_experts": n_experts,
        "top_k": top_k,
        "entropy_ratio_vs_uniform": entropy_ratio,
        "effective_expert_count": effective_expert_count,
        "hard_dead_count": int(dead["count"].item()),
        # WHOSE tokens these numbers describe. Under data parallelism each
        # rank sees only its own shard of the batch, so every number in this
        # row is computed over 1/world_size of the step's tokens.
        #
        # The label exists because the row merely APPEARING is a convincing
        # false signal that the multi-GPU gap is closed. It is not. maxvio is
        # an extremal statistic, so a rank-local reading is a LOWER BOUND on
        # the global one and the gap widens with world_size; entropy ratio and
        # effective expert count are distribution-shape quantities and degrade
        # more gracefully, but they are still not the global values either.
        #
        # "rank_local" is written even at world_size == 1, where it happens to
        # coincide with the global value. A field that shows up only when
        # something is wrong reads as an alarm rather than as provenance, and
        # its absence would have to be special-cased by every consumer.
        "aggregation_scope": aggregation_scope,
        "rank": rank,
        "world_size": world_size,
        "maxvio_count": maxvio_count,
        "maxvio_mass": maxvio_mass,
        # Recorded so a consumer can tell a run that used the deterministic
        # path from an older artifact that did not, without dating the file.
        "mass_reduction_device": "cpu",
        "maxvio_mass_to_count_ratio": ratio,
        "margin_mean": float(margin.mean().item()),
        "margin_min": float(margin.min().item()),
    }


# ---------------------------------------------------------------------------
# #9 -- router z-loss state sensor (mean z_t, sign-flip rate).
# ---------------------------------------------------------------------------


def zloss_state_record(
    *,
    step: int,
    layer_idx: int,
    loop_step: int,
    router_logits: Tensor,
) -> dict[str, Any]:
    """One `kind="zloss_state"` row: the distribution of the router
    log-partition `z_t` for this (layer_idx, loop_step).

    ## What this measures and why it is not redundant with `loss_record`

    `loss_record` already logs the scalar `z_loss` the model returns, which
    is `mean(z_t**2)` averaged over the unrolled depth. That scalar is a
    *magnitude* -- it cannot distinguish `z_t = +0.2` from `z_t = -0.2`,
    and the sign is the entire mechanism of interest here:

    Differentiating `L_z = c_z * mean(z_t**2)` with respect to two router
    logits `i, j` with `r_i > r_j` gives, for the margin `m_ij = r_i - r_j`
    under one gradient step with learning rate `eta`:

        d(m_ij) = -2 * eta * c_z * z_t * (p_i - p_j)

    `p_i > p_j` holds by construction, so the SIGN of the update is set
    entirely by `sign(z_t)`:

    * `z_t > 0` -> the margin SHRINKS -> routing softens toward uniform.
      This is the behavior z-loss is usually assumed to have.
    * `z_t < 0` -> the margin WIDENS -> routing sharpens toward
      concentration, i.e. it actively drives over-specialization.

    So `frac_z_negative` is a direct readout of what fraction of tokens the
    z-loss is currently pushing toward over-specialization, and it is
    available strictly before any expert-norm collapse becomes visible.
    Source of the derivation and of the reference values below:
    `The death of z-loss in modern LLMs` (Li et al. 2026, sections 3-4),
    <https://alltoall.notion.site/z-loss-in-llms-a-story-of-expert-collapse-and-specialization>.
    Their reported sweep at E=512/top-8, step 40k, is `mean(z_t)`
    7.45 -> 1.72 -> 0.20 -> 0.03 with `Pr[z_t<0]` 0% -> 4.6% -> 21.7% ->
    41.5% for `c_z` 0 / 1e-5 / 1e-4 / 1e-3.

    ⚠️ **Those reference numbers are NOT thresholds for this project.**
    They come from a 512-expert / top-8 model; this project runs E in
    {8, 16, 64} with top-2, and the same reference explicitly scopes the
    collapse it documents to "the wide ones (more than 300 routed
    experts)". They are recorded here as the provenance of *why this row
    exists*, not as a pass/fail line.

    Args:
        router_logits: `(batch, seq, n_experts)` raw pre-softmax router
            output for this (layer_idx, loop_step) -- the SAME tensor
            `router_layer_loop_record` takes. Probabilities would be wrong
            input: `logsumexp` of a softmax output is not `z_t`.

    Reduction order (metric definitions specify the reduction order):
        1. `z_t = logsumexp(router_logits, dim=-1)` computed **per token**,
           in float32 (`.float()`), matching how the model's own z-loss
           term is computed in `modeling_loop_lm.MoELayer.forward` -- doing
           the reduction in bf16 would change the low-order digits of a
           quantity whose SIGN is the thing being measured.
        2. Only then reduced over `batch * seq`: `z_mean` is
           `z_t.mean()`, `frac_z_negative` is `(z_t < 0).float().mean()`.
           These two steps do not commute, and step 2 must never be applied
           before step 1.
        3. `z_sq_mean` is `(z_t**2).mean()`, i.e. exactly this layer's
           un-weighted contribution to the model's own `z_loss` scalar.
           It is emitted so an offline reader can cross-check this row
           against a training-time `loss_record` from the same step rather
           than having to trust that the two agree.

    `frac_z_negative` uses a strict `< 0` comparison: exact zeros (possible
    only if every router logit for a token is exactly `-inf`-free and the
    partition is exactly 1.0) count as non-negative, matching the
    derivation above, where `z_t == 0` produces no margin update at all.
    """
    if router_logits.ndim != 3:
        raise ValueError(
            f"router_logits must be (batch, seq, n_experts); got shape "
            f"{tuple(router_logits.shape)}. Passing an already-reduced or "
            "already-softmaxed tensor here would silently produce a "
            "meaningless z_t."
        )

    z = torch.logsumexp(router_logits.float(), dim=-1)  # (batch, seq)
    n_tokens = int(z.numel())
    if n_tokens == 0:
        raise ValueError(
            "router_logits contains zero tokens -- refusing to emit a "
            "zloss_state row whose statistics would be NaN."
        )

    return {
        "kind": "zloss_state",
        "step": step,
        "layer_idx": layer_idx,
        "loop_step": loop_step,
        "n_experts": int(router_logits.shape[-1]),
        "n_tokens": n_tokens,
        "z_mean": float(z.mean().item()),
        "z_std": float(z.std(unbiased=True).item()) if n_tokens > 1 else 0.0,
        "z_min": float(z.min().item()),
        "z_max": float(z.max().item()),
        "z_sq_mean": float((z**2).mean().item()),
        "frac_z_negative": float((z < 0).float().mean().item()),
    }


# ---------------------------------------------------------------------------
# #10 -- top-k margin against a per-checkpoint null baseline, plus quantiles.
# ---------------------------------------------------------------------------


def _column_shuffle(centered: Tensor, generator: torch.Generator, tokens_per_doc: int | None):
    """One per-expert column shuffle of an ALREADY per-token-centered `(T, E)`
    matrix, scoped either to the whole batch or to within each document.

    `shuffled[i, e] = centered[perm_e[i], e]`, with the permutation drawn
    independently per expert column. This preserves each expert's marginal
    score distribution exactly (so the null keeps the distribution SHAPE)
    while destroying the within-token structure across experts -- i.e.
    destroying routing preference, the thing the null must not have.

    `tokens_per_doc` selects the scope. Whole-batch shuffling moves a token's
    score to any other token in the probe batch, including tokens from a
    different document; within-document shuffling keeps it inside its own
    document. The two ask different questions, and for a router that
    specializes purely BY DOMAIN they disagree: whole-batch treats
    "expert 3 fires on the biology document" as real structure, while
    within-document calls the same thing null, because it only asks whether
    the router discriminates BETWEEN TOKENS OF ONE DOCUMENT. Neither is
    wrong; they are different null hypotheses, which is why the scope is
    recorded in every row rather than assumed.

    Documents are required to be equal-sized contiguous blocks -- which is
    what the fixed probe batch produces -- so the shuffle is one reshape and
    one argsort rather than a Python loop over documents.
    """
    if tokens_per_doc is None:
        noise = torch.rand(centered.shape, generator=generator, device=centered.device)
        return torch.gather(centered, 0, noise.argsort(dim=0))
    n_tokens, n_experts = centered.shape
    blocked = centered.view(n_tokens // tokens_per_doc, tokens_per_doc, n_experts)
    noise = torch.rand(blocked.shape, generator=generator, device=centered.device)
    return torch.gather(blocked, 1, noise.argsort(dim=1)).reshape(n_tokens, n_experts)


STATISTICS: tuple[str, ...] = ("mean", "p25", "p50", "p75")
"""The margin summaries that each get their own null distribution.

`mean` is the one the published figures report as a line; `p25`/`p50`/`p75`
are the ribbon around it. Each needs its OWN null: a null distribution is a
distribution OF A STATISTIC, so the mean's null cannot be divided into a
quantile -- that would be standardising one quantity by another's baseline.
"""



def selection_gap(scores: Tensor, top_k: int) -> Tensor:
    """Per-token `r_(k) - r_(k+1)`: the SELECTED set against the rest.

    The companion to `topk_margin`, which measures spread WITHIN the selected
    block (`r_(1) - r_(k)`). The two answer different questions and are not
    substitutes:

    * within-selection asks "do the k chosen experts differ from each other?"
    * selection separation asks "was this group chosen FOR this token?"

    The second exists because the first misreads block specialization. When a
    token has a dedicated group of k experts that share its work, the group is
    near-equivalent internally BY DEFINITION, so the within-selection margin is
    small and rho falls below 1 -- which reads as "interchangeable, therefore
    under-specialized" for a router that is in fact strongly specialized.
    A measured rho of 0.04-0.07 for perfect block specialization falls
    firmly in "refutes the hypothesis" territory.

    Costs no extra forward pass: same scores, one more order statistic.

    Raises `MetricNotApplicable` when `top_k == n_experts`, where there is no
    (k+1)-th expert to separate from -- a structural fact about the
    configuration, not a failure, which is why it is not a ValueError.
    """
    n_experts = scores.shape[-1]
    if top_k >= n_experts:
        raise MetricNotApplicable(
            f"selection separation needs an unselected expert, but top_k={top_k} "
            f"of n_experts={n_experts} selects them all. Nothing is left to "
            "separate from; the quantity is undefined for this configuration, "
            "not zero."
        )
    top = scores.topk(top_k + 1, dim=-1).values
    return top[..., top_k - 1] - top[..., top_k]


#: The two quantities the criteria pair. Keys are the names that appear in
#: artifacts; values are the per-token functions.
QUANTITIES: dict[str, Any] = {
    "within_selection": _moe.topk_margin,
    "selection_separation": selection_gap,
}


def _margin_statistics(margin: Tensor) -> Tensor:
    """`[mean, p25, p50, p75]` of a per-token margin vector, in `STATISTICS`
    order. Quantiles are of the PER-TOKEN values, never of an aggregate."""
    q = torch.quantile(margin, torch.tensor([0.25, 0.5, 0.75], device=margin.device))
    return torch.stack([margin.mean(), q[0], q[1], q[2]])


def _null_margin_stats(
    centered: Tensor,
    top_k: int,
    n_shuffles: int,
    generator: torch.Generator,
    tokens_per_doc: int | None,
    quantity_name: str = "within_selection",
) -> Tensor:
    """`(n_shuffles, len(STATISTICS))` -- each row is one null draw's four
    summaries. Computing all four costs one extra `torch.quantile` per draw
    on a vector the shuffle already produced; the shuffle dominates."""
    quantity = QUANTITIES[quantity_name]
    out = torch.empty(n_shuffles, len(STATISTICS), dtype=torch.float32, device=centered.device)
    for i in range(n_shuffles):
        shuffled = _column_shuffle(centered, generator, tokens_per_doc)
        out[i] = _margin_statistics(quantity(shuffled, top_k))
    return out


def _null_margin_means(
    centered: Tensor,
    top_k: int,
    n_shuffles: int,
    generator: torch.Generator,
    tokens_per_doc: int | None,
) -> Tensor:
    """`n_shuffles` null draws of `topk_margin(...).mean()`.

    Deliberately NOT a row permutation and NOT an expert relabeling: margin
    depends only on the sorted order within a token, so both of those leave
    every margin value bit-identical and have zero test power.
    """
    out = torch.empty(n_shuffles, dtype=torch.float32, device=centered.device)
    for i in range(n_shuffles):
        out[i] = _moe.topk_margin(_column_shuffle(centered, generator, tokens_per_doc), top_k).mean()
    return out


def _structureless_surrogate(centered: Tensor, generator: torch.Generator) -> Tensor:
    """A matrix with the same VALUE DISTRIBUTION as `centered` but no
    within-token structure at all: every entry is drawn independently, with
    replacement, from the pooled centered scores of this cell.

    Used to measure how much of an observed ratio is structural rather than
    real specialization -- see `margin_null_record`'s calibration section.
    """
    pool = centered.reshape(-1)
    idx = torch.randint(0, pool.numel(), centered.shape, generator=generator, device=centered.device)
    return pool[idx]




#: The criteria's position bins, as CUT POINTS on the post-drop position
#: axis: [0, 64), [64, 256), [256, end). Uneven on purpose -- the question is
#: whether a result is an artefact of where in a document the tokens sat, and
#: the interesting structure is concentrated near the start, so equal-width
#: bins would blur exactly the region being checked.
#:
#: ⚠️ Positions are counted AFTER the corpus layer drops each document's first
#: token, so position 0 here is the document's SECOND token. Recorded in the
#: artifact as `position_axis` rather than left implicit: a reader comparing
#: against a pre-drop artifact would otherwise be off by one with nothing to
#: show it.
POSITION_BIN_EDGES: tuple[int, ...] = (0, 64, 256)


def _quantity_block(
    centered: Tensor,
    *,
    top_k: int,
    n_experts: int,
    n_shuffles: int,
    tokens_per_doc: int | None,
    shuffle_seed: int,
    quantity_name: str,
) -> dict[str, Any]:
    """The full ratio apparatus for one per-token quantity.

    Same construction as the within-selection path above -- per-token
    centering, per-expert column shuffle, and the structureless surrogate for
    the zero point -- so the two quantities are comparable to each other rather
    than each having its own conventions.

    Its own generator, seeded identically, so the two quantities see the SAME
    sequence of shuffles. That makes their ratio difference attributable to the
    quantity rather than to draw noise, which matters because the criteria read
    the two side by side in a four-cell table.
    """
    quantity = QUANTITIES[quantity_name]
    generator = torch.Generator(device=centered.device)
    generator.manual_seed(shuffle_seed)

    observed = _margin_statistics(quantity(centered, top_k))
    null = _null_margin_stats(
        centered, top_k, n_shuffles, generator, tokens_per_doc, quantity_name
    ).median(dim=0).values

    surrogate = _structureless_surrogate(centered, generator)
    surrogate_centered = surrogate - surrogate.mean(dim=-1, keepdim=True)
    sur_obs = _margin_statistics(quantity(surrogate_centered, top_k))
    sur_null = _null_margin_stats(
        surrogate_centered, top_k, n_shuffles, generator, tokens_per_doc, quantity_name
    ).median(dim=0).values

    degenerate = [
        name for j, name in enumerate(STATISTICS)
        if float(null[j].item()) <= 0 or float(sur_null[j].item()) <= 0
    ]
    if degenerate:
        raise ValueError(
            f"{quantity_name}: null median is <= 0 for statistic(s) {degenerate}; the "
            "ratio is undefined for those. Enough tokens are tied that the "
            "shuffled baseline collapses -- a finding to report, not a denominator."
        )

    analytic = math.sqrt(n_experts / (n_experts - 1))
    statistics: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(STATISTICS):
        obs = float(observed[j].item())
        rho = obs / float(null[j].item())
        ref = float(sur_obs[j].item()) / float(sur_null[j].item())
        statistics[name] = {
            "observed": obs,
            "null_median": float(null[j].item()),
            "rho": rho,
            "rho_ref": ref,
            "rho_cal": rho / ref,
            "rho_ref_over_analytic": ref / analytic,
        }
    mean = statistics["mean"]
    return {
        "quantity": quantity_name,
        "observed_mean": mean["observed"],
        "null_median": mean["null_median"],
        "rho": mean["rho"],
        "rho_ref": mean["rho_ref"],
        "rho_cal": mean["rho_cal"],
        "rho_ref_normal_prediction": analytic,
        "rho_ref_over_analytic": mean["rho_ref_over_analytic"],
        "statistics": statistics,
        "n_shuffles": int(n_shuffles),
    }


#: The four-cell reading the criteria use. Both quantities near 1
#: is under-specialization; either one high means a real, differently-shaped
#: specialization. The threshold is deliberately NOT a criteria threshold --
#: this label is descriptive, and the judgement uses `judgement_rho_cal`.
_CELL_HIGH = 1.10


def _specialization_cell(within: float, separation: float) -> str:
    """Which of the four cells this (within, separation) pair falls in.

    Named rather than numbered so a reader does not have to hold the table in
    their head, and returned alongside the numbers, never instead of them.
    """
    if within != within or separation != separation:  # NaN
        return "undetermined"
    hi_w, hi_s = within >= _CELL_HIGH, separation >= _CELL_HIGH
    if not hi_w and not hi_s:
        return "under_specialized"
    if not hi_w and hi_s:
        return "block_specialization"
    if hi_w and not hi_s:
        return "hierarchical_specialization"
    return "strong_specialization"


def margin_null_record(
    *,
    step: int,
    layer_idx: int,
    loop_step: int,
    router_logits: Tensor,
    top_k: int,
    tokens_per_doc: int | None = None,
    shuffle_scope: str = "batch",
    n_shuffles: int = 200,
    shuffle_seed: int = 0,
    n_bootstrap: int = 200,
    n_shuffles_bootstrap: int = 15,
    recalibrate_bootstrap: bool = True,
    # No default, deliberately. The criteria file carries `min_tokens: 50000`,
    # and a default here was a SECOND copy of that number maintained
    # independently -- and the criteria file's copy turned out never to have
    # been read by anything. Two numbers that are supposed to be the same, with only
    # one of them checked, is how a precondition silently stops matching the
    # criteria it claims to enforce.
    min_tokens: int,
    n_tokens_effective: int | None = None,
    doc_domains: list[str] | None = None,
    position_bin_edges: tuple[int, ...] = POSITION_BIN_EDGES,
) -> dict[str, Any]:
    """One `kind="margin_null"` row: the top-k margin for this
    (layer_idx, loop_step), reported as quantiles and as a dimensionless
    ratio against a null built from this checkpoint's OWN router scores.

    ## ⚠️ The direction of `rho` depends on `top_k`

    `margin = top1 - top_k` spans the whole SELECTED BLOCK. At k == 2 that is
    just the top-2 gap and specialization widens it, so a larger rho means
    more specialization. At k > 2 a router that spreads a token across its k
    experts roughly equally -- the behaviour top-k routing is built for --
    has a small within-block spread, while the column shuffle destroys the
    "exactly k high scores per token" structure and produces a WIDER
    baseline. So rho < 1 at large k indicates BLOCK specialization, not its
    absence.

    ⟹ **rho is comparable only between models with the same `top_k`.** The
    row records `top_k` and `margin_semantics` so a consumer can enforce
    that; `margin_readout` refuses a cross-k comparison outright.

    ## Why the raw margin is not reportable on its own

    `router_layer_loop_record` already emits `margin_mean`/`margin_min`.
    Those are NOT comparable across models, and the failure inverts
    conclusions rather than blurring them: the margin carries units (raw
    router score) whose scale is set by `d_model`, gate init and training;
    and under a NO-PREFERENCE null it is not 0 but the expected gap between
    the 1st and k-th order statistic of E scores, so it moves with E and k
    on its own. The published reference 0.23 reads as "under-specialized"
    at E=512/top-8, while this project's E=64/top-2 config yields 0.382
    under a completely random router.

    ## Three separate corrections live here; keep them distinct

    **1. The null (`rho`).** Per-token centering, then a per-expert column
    shuffle (`_column_shuffle`), `n_shuffles` times; `rho` is the observed
    mean over the median null draw. Centering must come first -- skipping
    it mixes one token's common mode into another's vector, inflates the
    null, and a structureless router then scores rho ~= 0.16, i.e. the
    answer points the wrong way. A regression test pins that ordering.

    **2. The structural bias (`rho_ref`, `rho_cal`).** Even a completely
    structureless router does not score rho == 1. The column shuffle
    destroys the within-token covariance that centering induced, so the
    null's spread is systematically smaller than the observation's, and the
    ratio sits above 1 by construction. For i.i.d. normal scores the offset
    is `sqrt(E/(E-1))` -- 1.069 at E=8, 1.033 at E=16, 1.008 at E=64.

    ⚠️ An earlier version of this docstring called that offset "finite-sample
    positive bias". **That was wrong**: it is structural, not a sampling
    artifact, and it does not shrink as tokens are added. It was found by
    matching a measured 1.008 at E=64 against `sqrt(64/63)`.

    The analytic form is NOT used as a correction, because it only holds for
    normal scores -- under uniform scores it is off by about -46%, and the
    real score distribution is trained, not known. Instead `rho_ref` is
    calibrated empirically per cell: pool this cell's centered scores, draw
    each entry independently from that pool (`_structureless_surrogate`) to
    get a matrix with the same value distribution and no structure at all,
    and run it through the identical centering + shuffle pipeline. Then
    `rho_cal = rho / rho_ref` is ~1 for a structureless router at ANY E and
    under any score distribution. `sqrt(E/(E-1))` is retained only as a
    shape check: `rho_ref_deviates_from_normal` flags a >= 3% gap, which
    says the scores are far from normal, not that anything is broken.

    **3. The shuffle scope (`shuffle_scope`).** `"batch"` shuffles across
    the whole probe batch; `"document"` shuffles only within each document
    (requires `tokens_per_doc`). For a router that specializes purely by
    domain the two disagree by construction -- see `_column_shuffle`. Both
    are legitimate; neither is the default answer, so the scope is recorded
    in the row and both are meant to be reported side by side.

    ## Sample size: raw vs effective

    `n_tokens` counts rows. `n_tokens_effective` is what the caller
    determined to be genuinely distinct text after deduplicating the
    windows several probe seeds drew from the same documents -- overlapping
    windows on short documents make the raw count a fiction. Preconditions
    are checked against the effective count when it is supplied.

    `qualified` is bookkeeping only. It never suppresses numbers and never
    raises: an under-powered cell is still worth plotting, and refusing to
    JUDGE it is the readout script's job, gated by the criteria file. The
    ratified gate is a precision requirement (`rho_ci_halfwidth`), not a
    token count -- token counts are a proxy for precision, and the
    bootstrap measures precision directly.

    ## Document-level bootstrap

    `rho_boot_lo/hi` resample DOCUMENTS with replacement (not tokens):
    tokens within a document are not independent, so a token-level
    bootstrap would report an interval several times too narrow.
    Requires `tokens_per_doc`.

    Args:
        router_logits: `(n_tokens, n_experts)` raw pre-softmax scores, float32.
            Two dimensions, not three: reaching the token precondition means
            concatenating several probe forwards, and making that the
            caller's explicit job keeps it visible.
        tokens_per_doc: document block size; documents must be equal-sized
            contiguous blocks. Required for `shuffle_scope="document"`, for
            per-document statistics and for the bootstrap.
        top_k: `top_k == 1` raises -- the margin is identically 0 there.

    Reduction order:
        1. `topk_margin` per token over the RAW scores -> the observation.
        2. `mean`/`p25`/`p50`/`p75`/`min` over that per-token vector.
           Quantiles are of PER-TOKEN margins, never of an aggregate.
        3. The null centers, then shuffles, then takes the MEAN per draw --
           so the null distribution is a distribution OF THE MEAN. It is
           comparable to `margin_mean` only, never to p25/p75.
    """
    if router_logits.ndim != 2:
        raise ValueError(
            f"router_logits must be (n_tokens, n_experts); got "
            f"{tuple(router_logits.shape)}. A (batch, seq, n_experts) tensor "
            "would be shuffled along the batch axis only, silently producing a "
            "null with the wrong number of draws."
        )
    if top_k < 2:
        raise MetricNotApplicable(
            f"top_k={top_k}: the top-k margin is identically 0 when k == 1 "
            "(top-1 and k-th are the same expert), so this row would be a "
            "column of zeros and a 0/0 ratio that looks like data. This is a "
            "structural property of the configuration, not a failure -- a "
            "producer may record it as not-applicable."
        )
    if shuffle_scope not in ("batch", "document"):
        raise ValueError(f"shuffle_scope must be 'batch' or 'document', got {shuffle_scope!r}.")
    n_tokens, n_experts = router_logits.shape
    if n_tokens == 0:
        raise ValueError("router_logits contains zero tokens -- statistics would be NaN.")
    if top_k > n_experts:
        raise ValueError(f"top_k={top_k} exceeds n_experts={n_experts}.")
    if n_shuffles < 2:
        raise ValueError(f"n_shuffles={n_shuffles}: a null INTERVAL needs at least 2 draws.")
    if tokens_per_doc is not None and n_tokens % tokens_per_doc:
        raise ValueError(
            f"n_tokens={n_tokens} is not a multiple of tokens_per_doc="
            f"{tokens_per_doc}; documents must be equal-sized contiguous blocks."
        )
    if shuffle_scope == "document" and tokens_per_doc is None:
        raise ValueError(
            "shuffle_scope='document' needs tokens_per_doc -- without document "
            "boundaries there is nothing to shuffle within."
        )

    scores = router_logits.float()
    margin = _moe.topk_margin(scores, top_k)
    q = torch.quantile(margin, torch.tensor([0.25, 0.5, 0.75], device=margin.device))
    margin_mean = float(margin.mean().item())

    centered = scores - scores.mean(dim=-1, keepdim=True)
    shuffle_blocks = tokens_per_doc if shuffle_scope == "document" else None

    generator = torch.Generator(device=centered.device)
    generator.manual_seed(shuffle_seed)
    # All four statistics share the same shuffles -- one null pass, not four.
    null_stats = _null_margin_stats(centered, top_k, n_shuffles, generator, shuffle_blocks)
    qs = torch.tensor([0.025, 0.5, 0.975], device=null_stats.device)
    null_q_all = torch.stack([torch.quantile(null_stats[:, j], qs) for j in range(len(STATISTICS))])
    null_means = null_stats[:, 0]
    null_lo, null_p50, null_hi = (float(v.item()) for v in null_q_all[0])
    if null_p50 <= 0.0:
        raise ValueError(
            f"null median margin is {null_p50} (<= 0) at layer_idx={layer_idx}, "
            f"loop_step={loop_step}: the scores are degenerate (every expert tied "
            "for every token). That is a finding to report, not a denominator."
        )
    rho = margin_mean / null_p50

    # --- structural-bias calibration -------------------------------------
    surrogate = _structureless_surrogate(centered, generator)
    surrogate_centered = surrogate - surrogate.mean(dim=-1, keepdim=True)
    sur_obs_all = _margin_statistics(_moe.topk_margin(surrogate_centered, top_k))
    sur_null_all = _null_margin_stats(
        surrogate_centered, top_k, n_shuffles, generator, shuffle_blocks
    ).median(dim=0).values
    # Same degeneracy, same treatment as the main path above. Returning NaN
    # here instead would let it flow into `rho_cal` and
    # `rho_ref_deviates_from_normal` with nothing in the row saying the
    # structural reference for this cell was undefined -- one kind of
    # degeneracy raising while an identical one passes silently.
    degenerate = [name for j, name in enumerate(STATISTICS) if not sur_null_all[j] > 0]
    if degenerate:
        raise ValueError(
            f"surrogate null margin is <= 0 for statistic(s) {degenerate} at "
            f"layer_idx={layer_idx}, loop_step={loop_step}: the calibration "
            "reference is undefined, so `rho_cal` would be meaningless. This "
            "means the pooled scores are degenerate (effectively all tied) -- "
            "a finding to report, not a denominator."
        )
    # Divide as Python floats, NOT as tensors. `sur_obs_all / sur_null_all` is a
    # float32/float32 divide whose QUOTIENT is rounded to float32 before `.item()`
    # ever widens it, so `rho_ref` -- alone among the ratios here -- used to land
    # exactly on the float32 grid while `rho` (computed at the `stat_rho` line
    # below from two already-`.item()`ed operands) did not. Same operands, same
    # value to ~1e-8; the only difference is where the rounding happens. Keep
    # this consistent with `stat_rho` so the two ratios cannot drift apart again.
    rho_ref_all = [
        float(sur_obs_all[j].item()) / float(sur_null_all[j].item())
        for j in range(len(STATISTICS))
    ]
    rho_ref = rho_ref_all[0]
    normal_prediction = math.sqrt(n_experts / (n_experts - 1))

    # --- document-level bootstrap ----------------------------------------
    by_domain: list[dict[str, Any]] | None = None
    by_position_bin: list[dict[str, Any]] | None = None
    boot_lo = boot_hi = cal_lo = cal_hi = None
    boot_corr = boot_cv_rho = boot_cv_ref = None
    cal_q_all = None
    n_degenerate_draws = None
    by_document: list[dict[str, float]] | None = None
    if tokens_per_doc is not None:
        blocked = margin.view(n_tokens // tokens_per_doc, tokens_per_doc)
        per_doc_mean = blocked.mean(dim=1)
        by_document = [
            {"doc_index": i, "margin_mean": float(v.item())}
            for i, v in enumerate(per_doc_mean)
        ]
        n_docs = per_doc_mean.numel()

    # The bootstrap is guarded SEPARATELY from the per-document values above.
    # `n_bootstrap <= 0` means "skip the interval", and it has to be honoured
    # here as well as on the no-documents path: it used to fall through with
    # zero draws and die inside torch.quantile on an empty tensor, so a
    # legitimate request came back as a crash with an unrelated message.
    #
    # Guarding the WHOLE block instead would have taken `by_document` with it,
    # and those per-document values are a requirement in their own right --
    # they are what the document-level bootstrap and the per-domain checks are
    # computed from, not a by-product of running the bootstrap.
    if tokens_per_doc is not None and n_bootstrap > 0:
        # Resample DOCUMENTS, not tokens: tokens inside one document are not
        # independent, so a token-level bootstrap reports an interval several
        # times too narrow and would make an under-powered cell look precise.
        pick = torch.randint(
            0, n_docs, (n_bootstrap, n_docs), generator=generator, device=per_doc_mean.device
        )
        boot_rho = per_doc_mean[pick].mean(dim=1) / null_p50
        bq = torch.quantile(boot_rho, torch.tensor([0.025, 0.975], device=boot_rho.device))
        boot_lo, boot_hi = float(bq[0].item()), float(bq[1].item())

        if recalibrate_bootstrap:
            # Build the interval on the CALIBRATED
            # scale by redoing the calibration inside every resample, rather
            # than rescaling the finished interval by a fixed `rho_ref`.
            #
            # Rescaling by a fixed `rho_ref` is exact for the point (it is a
            # per-cell constant divide) but treats `rho_ref` as if it were
            # known without error. Re-estimating it per draw lets the
            # surrogate move with the resampled documents, so the interval
            # also carries the calibration's own sampling uncertainty -- which
            # is what makes it honest for CROSS-MODEL comparison, where two
            # cells have different `rho_ref` estimates.
            #
            # `n_shuffles_bootstrap` is deliberately smaller than `n_shuffles`:
            # each draw needs only a null MEDIAN, and the bootstrap averages
            # over draws, so precision per draw matters far less than the
            # number of draws. Cost is n_bootstrap x 2 x n_shuffles_bootstrap
            # shuffles per cell, which is why it is not simply `n_shuffles`.
            blocked_scores = centered.view(n_docs, tokens_per_doc, n_experts)
            cal = torch.empty(n_bootstrap, dtype=torch.float32, device=centered.device)
            # Retained, not recomputed: `sub_rho` and `sur_rho` are already
            # produced by every draw below and were previously discarded in
            # favour of their ratio. Keeping them costs two arrays of length
            # n_bootstrap and no extra forward or shuffle.
            draw_rho = torch.empty_like(cal)
            draw_ref = torch.empty_like(cal)
            cal_all = torch.empty(
                n_bootstrap, len(STATISTICS), dtype=torch.float32, device=centered.device
            )
            inner = n_shuffles_bootstrap
            n_degenerate_draws = 0
            keep = torch.ones(n_bootstrap, dtype=torch.bool, device=centered.device)
            for i in range(n_bootstrap):
                sub = blocked_scores[pick[i]].reshape(-1, n_experts)
                sub_obs = _margin_statistics(_moe.topk_margin(sub, top_k))
                sub_null = _null_margin_stats(
                    sub, top_k, inner, generator, shuffle_blocks
                ).median(dim=0).values
                sur = _structureless_surrogate(sub, generator)
                sur = sur - sur.mean(dim=-1, keepdim=True)
                sur_obs = _margin_statistics(_moe.topk_margin(sur, top_k))
                sur_null = _null_margin_stats(
                    sur, top_k, inner, generator, shuffle_blocks
                ).median(dim=0).values
                # Per statistic: rho on this resample, divided by the
                # reference this resample produces for the SAME statistic.
                # A single degenerate resample must not silently pollute the
                # quantiles with NaN/Inf. Such draws are excluded and COUNTED,
                # so the row states how many were dropped rather than leaving
                # a reader to infer it.
                if bool((sub_null <= 0).any() or (sur_null <= 0).any()):
                    n_degenerate_draws += 1
                    keep[i] = False
                    continue
                cal_all[i] = (sub_obs / sub_null) / (sur_obs / sur_null)
                sub_rho = float((sub_obs[0] / sub_null[0]).item())
                sur_rho = float((sur_obs[0] / sur_null[0]).item())
                draw_rho[i], draw_ref[i] = sub_rho, sur_rho
                cal[i] = cal_all[i, 0]
            if not bool(keep.any()):
                raise ValueError(
                    f"all {n_bootstrap} bootstrap resamples were degenerate at "
                    f"layer_idx={layer_idx}, loop_step={loop_step}; no calibrated "
                    "interval can be formed."
                )
            bq = torch.tensor([0.025, 0.975], device=cal_all.device)
            cal_kept, cal_all_kept = cal[keep], cal_all[keep]
            cal, cal_all = cal_kept, cal_all_kept
            # Only the diagnostics and the calibrated interval exclude the
            # dropped draws; the uncalibrated interval never used the
            # surrogate, so it is unaffected and is left as computed.
            draw_rho, draw_ref = draw_rho[keep], draw_ref[keep]
            cal_q_all = torch.stack(
                [torch.quantile(cal_all[:, j], bq) for j in range(len(STATISTICS))]
            )
            cal_lo, cal_hi = float(cal_q_all[0][0].item()), float(cal_q_all[0][1].item())

            # Diagnostics that make "is the fixed-rho_ref rescale too narrow
            # or too wide here?" COMPUTABLE from the artifact, instead of a
            # remembered measurement. By the delta method,
            #   Var(rho/rho_ref)/R^2 ~ CV^2(rho) + CV^2(rho_ref)
            #                          - 2*corr*CV(rho)*CV(rho_ref)
            # and the fixed rescale keeps only the first term, so it is too
            # narrow exactly when CV(rho_ref) > 2*corr*CV(rho). The direction
            # is data-dependent -- measured both ways on this project's own
            # checkpoints -- which is why it must be derived per cell rather
            # than asserted.
            mr, mf = draw_rho.mean(), draw_ref.mean()
            boot_cv_rho = float((draw_rho.std(unbiased=True) / mr).item()) if mr != 0 else None
            boot_cv_ref = float((draw_ref.std(unbiased=True) / mf).item()) if mf != 0 else None
            stacked = torch.stack([draw_rho, draw_ref])
            cmat = torch.corrcoef(stacked)
            boot_corr = float(cmat[0, 1].item())

    # --- per-statistic block -------------------------------------------
    # Built FIRST, and the flat fields below are read out of it, so the
    # deliberate duplication cannot drift: `statistics["mean"][k]` and its
    # flat counterpart are the same Python float object, not two roundings
    # of one computation. Consumers rely on this as an invariant.
    observed_all = _margin_statistics(margin)
    # Same degeneracy, same treatment -- for EVERY statistic, not just the
    # mean. The main path's raise above covers only j=0; p25/p50/p75 have
    # their own null distributions and can degenerate independently (a
    # quantile's null collapses once enough tokens tie, while the mean's
    # stays positive). Leaving an `else float("nan")` here would reproduce
    # exactly the silent-NaN hole just closed for `rho_ref`, one level down.
    degenerate_null = [
        name for j, name in enumerate(STATISTICS) if not null_q_all[j][1] > 0
    ]
    if degenerate_null:
        raise ValueError(
            f"null median margin is <= 0 for statistic(s) {degenerate_null} at "
            f"layer_idx={layer_idx}, loop_step={loop_step}: the ratio is "
            "undefined for those. Enough tokens are tied that the shuffled "
            "baseline collapses -- a finding to report, not a denominator."
        )
    try:
        selection_block: dict[str, Any] = _quantity_block(
            centered, top_k=top_k, n_experts=n_experts, n_shuffles=n_shuffles,
            tokens_per_doc=shuffle_blocks, shuffle_seed=shuffle_seed,
            quantity_name="selection_separation",
        )
    except MetricNotApplicable as exc:
        # k == E: no unselected expert to separate from. A structural fact
        # about the configuration, recorded as such rather than as a failure
        # or as a zero -- the pair is simply one-sided here.
        selection_block = {"quantity": "selection_separation", "not_applicable": str(exc)}

    statistics: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(STATISTICS):
        n_lo, n_med, n_hi = (float(v.item()) for v in null_q_all[j])
        obs = float(observed_all[j].item())
        ref = rho_ref_all[j]
        stat_rho = obs / n_med  # n_med > 0 guaranteed by the check above
        statistics[name] = {
            "observed": obs,
            "null_median": n_med,
            "null_lo": n_lo,
            "null_hi": n_hi,
            "rho": stat_rho,
            "rho_ref": ref,
            "rho_cal": stat_rho / ref if ref and ref == ref else float("nan"),
            "rho_cal_boot_lo": None if cal_q_all is None else float(cal_q_all[j][0].item()),
            "rho_cal_boot_hi": None if cal_q_all is None else float(cal_q_all[j][1].item()),
        }
    mean_stat = statistics["mean"]
    _sep_rho_cal = selection_block.get("rho_cal", float("nan"))
    _within_rho_cal = mean_stat["rho_cal"]
    # NaN-safe: when the separation quantity does not apply (k == E) the
    # judgement falls back to the quantity that does, rather than becoming NaN
    # and quietly removing this cell from every downstream comparison.
    judgement_rho_cal = (
        _within_rho_cal if _sep_rho_cal != _sep_rho_cal
        else max(_within_rho_cal, _sep_rho_cal)
    )
    specialization_cell = _specialization_cell(_within_rho_cal, _sep_rho_cal)

    # --- grouped diagnostics -------------------------------------------
    # Not metrics: they exist so a difference can be checked for being an
    # artefact of WHICH text was probed (one domain dominating) or of WHERE
    # in the sequence a token sat, rather than of the router. They are
    # deliberately kept out of the metric rows for that reason.
    if tokens_per_doc is not None:
        blocked_margin = margin.view(n_tokens // tokens_per_doc, tokens_per_doc)

        if doc_domains is not None:
            if len(doc_domains) != blocked_margin.shape[0]:
                raise ValueError(
                    f"doc_domains has {len(doc_domains)} entries but there are "
                    f"{blocked_margin.shape[0]} document blocks. A mismatch would "
                    "silently attribute a document's tokens to the wrong domain."
                )
            per_doc = blocked_margin.mean(dim=1)
            groups: dict[str, list[float]] = {}
            for name, value in zip(doc_domains, per_doc.tolist()):
                groups.setdefault(name, []).append(value)
            by_domain = [
                {"domain": name, "n_docs": len(vals), "margin_mean": sum(vals) / len(vals)}
                for name, vals in sorted(groups.items())
            ]

        # Position bins are over the offset WITHIN a document, pooled across
        # documents: bin 0 is the first slice of every document. The bin count
        # travels with the readings because a 4-bin and an 8-bin mean are not
        # the same quantity -- the same reason `aggregation` is recorded.
        cuts = [c for c in position_bin_edges if c < tokens_per_doc]
        if cuts:
            bounds = list(cuts) + [tokens_per_doc]
            by_position_bin = [
                {
                    "bin_index": i,
                    "token_start": bounds[i],
                    "token_end": bounds[i + 1],
                    "n_position_bins": len(cuts),
                    # The axis these positions are counted on. Without it a
                    # reader cannot tell a post-drop bin from a pre-drop one,
                    # and the two are off by one token.
                    "position_axis": "post_first_token_drop",
                    "margin_mean": float(blocked_margin[:, bounds[i]:bounds[i + 1]].mean().item()),
                }
                for i in range(len(cuts))
            ]

    effective = n_tokens if n_tokens_effective is None else int(n_tokens_effective)
    return {
        "statistics": statistics,
        "kind": "margin_null",
        "step": step,
        "layer_idx": layer_idx,
        "loop_step": loop_step,
        "n_experts": int(n_experts),
        "top_k": int(top_k),
        # Descriptive, not a verdict: it says what the quantity IS at this k,
        # and lets the consumer derive the interpretive consequence. A
        # "small_k"/"large_k" label would bake in a threshold judgement
        # instead. At k == 2 the margin is exactly the top-2 gap; beyond that
        # it measures the spread across the whole selected block.
        "margin_semantics": "top2_gap" if top_k == 2 else "block_spread",
        # Constant, recorded so the constraint travels with the reading rather
        # than living only in a document.
        "rho_comparable_across_top_k": False,
        "n_tokens": int(n_tokens),
        # The criteria's name for the same count: rows actually
        # reduced over, BEFORE the overlap deduplication. Emitted next to
        # `n_tokens_effective` so the pair can be read without knowing that
        # `n_tokens` is the nominal one -- the whole point is that the
        # nominal figure looks like it clears the threshold when the effective
        # one does not. `n_tokens` stays for existing consumers.
        "n_tokens_nominal": int(n_tokens),
        "n_tokens_effective": effective,
        "tokens_per_doc": tokens_per_doc,
        "n_documents": None if tokens_per_doc is None else n_tokens // tokens_per_doc,
        "shuffle_scope": shuffle_scope,
        "qualified": bool(effective >= min_tokens),
        "min_tokens": int(min_tokens),
        "margin_mean": mean_stat["observed"],
        "margin_p25": float(q[0].item()),
        "margin_p50": float(q[1].item()),
        "margin_p75": float(q[2].item()),
        "margin_min": float(margin.min().item()),
        "null_margin_p50": mean_stat["null_median"],
        "null_margin_lo": mean_stat["null_lo"],
        "null_margin_hi": mean_stat["null_hi"],
        "rho": mean_stat["rho"],
        "rho_null_lo": null_lo / null_p50,
        "rho_null_hi": null_hi / null_p50,
        "rho_ref": mean_stat["rho_ref"],
        "rho_cal": mean_stat["rho_cal"],
        # The SECOND quantity the criteria pair with this one: k-th
        # minus (k+1)-th score, i.e. whether this group was chosen FOR this
        # token. Its own block rather than mixed in, or every field here would
        # need two spellings. Costs no extra forward pass -- same scores, one
        # more order statistic -- so it can also be recomputed over artifacts
        # that predate it.
        "selection_separation": selection_block,
        # The criteria's judgement quantity: max of the two calibrated ratios.
        # Reading only the within-selection ratio calls block specialization
        # "under-specialized" -- measured rho 0.04-0.07 for a perfectly
        # block-specialized router, i.e. firmly in "refutes" territory.
        "judgement_rho_cal": judgement_rho_cal,
        "specialization_cell": specialization_cell,
        "rho_ref_normal_prediction": normal_prediction,
        # The distribution-shape check as a NUMBER, not only as the boolean
        # below. The boolean answers "is the gap >= 3%?"; a consumer checking
        # whether this router's scores look normal at all needs the size and
        # DIRECTION of the gap -- measured deviations run both ways (uniform
        # scores come out below 1, normal ones above), and a flag cannot say
        # which. 1.0 means the empirical calibration reproduced sqrt(E/(E-1)).
        "rho_ref_over_analytic": (
            rho_ref / normal_prediction if rho_ref == rho_ref else float("nan")
        ),
        "rho_ref_deviates_from_normal": bool(
            rho_ref == rho_ref and abs(rho_ref - normal_prediction) / normal_prediction >= 0.03
        ),
        "rho_boot_lo": boot_lo,
        "rho_boot_hi": boot_hi,
        "rho_ci_halfwidth": None if boot_lo is None else (boot_hi - boot_lo) / 2.0,
        # The same interval expressed on the CALIBRATED scale. Both are emitted
        # because `rho` and `rho_cal` are different scales, and pairing a
        # `rho_cal` point with a `rho`-scale interval puts the point OUTSIDE its
        # own error bar for about two thirds of cells -- a plot that reads as
        # broken data when both numbers are in fact correct.
        #
        # Rescaling is EXACT, not an approximation: `rho_cal` is `rho` divided
        # by `rho_ref`, a single constant per cell, so the transform is monotone
        # linear and carries quantiles over unchanged.
        #
        # ⚠️ What it does NOT include: uncertainty in `rho_ref` itself. This
        # interval reflects resampling of DOCUMENTS only. `rho_ref` is estimated
        # once per cell from the surrogate, and propagating its error would mean
        # re-estimating it inside every bootstrap draw. For comparing cells that
        # share a `rho_ref` this makes no difference; for cross-model comparison
        # it is a known, unquantified omission.
        # Recalibrated per draw when `recalibrate_bootstrap` (the default);
        # otherwise the fixed-`rho_ref` rescale, which is exact for the point
        # but omits the calibration's own uncertainty.
        "rho_cal_boot_lo": mean_stat["rho_cal_boot_lo"] if cal_q_all is not None else (
            None if boot_lo is None or not rho_ref else boot_lo / rho_ref),
        "rho_cal_boot_hi": mean_stat["rho_cal_boot_hi"] if cal_q_all is not None else (
            None if boot_hi is None or not rho_ref else boot_hi / rho_ref),
        "rho_cal_ci_halfwidth": (
            (cal_hi - cal_lo) / 2.0 if cal_lo is not None else (
                None if boot_lo is None or not rho_ref else (boot_hi - boot_lo) / (2.0 * rho_ref))
        ),
        "bootstrap_recalibrated": bool(cal_lo is not None),
        # Explicit rather than inferable: how many resamples were dropped as
        # degenerate before the interval was formed.
        "n_bootstrap_degenerate_draws": n_degenerate_draws,
        # null on the fixed branch: not missing, INAPPLICABLE -- that path
        # never re-estimates rho_ref, so these quantities do not exist there.
        "boot_corr_rho_rhoref": boot_corr,
        "boot_cv_rho": boot_cv_rho,
        "boot_cv_rho_ref": boot_cv_ref,
        "n_shuffles_bootstrap": n_shuffles_bootstrap if cal_lo is not None else None,
        "by_document": by_document,
        "by_domain": by_domain,
        "by_position_bin": by_position_bin,
        "n_shuffles": int(n_shuffles),
        "n_bootstrap": int(n_bootstrap) if tokens_per_doc is not None else None,
        "shuffle_seed": int(shuffle_seed),
    }


# ---------------------------------------------------------------------------
# #5 -- residual-stream norms, within-step and step-boundary.
# ---------------------------------------------------------------------------



def _document_resample_indices(
    n_docs: int, n_bootstrap: int, generator: torch.Generator, device: Any
) -> Tensor:
    """`(n_bootstrap, n_docs)` document indices, drawn ONCE for a whole archive.

    Drawn here, and only here, so that every grid point of the archive is
    resampled with the SAME documents in the same draw. See
    `archive_document_bootstrap` for why that is the whole point.
    """
    return torch.randint(0, n_docs, (n_bootstrap, n_docs), generator=generator, device=device)


def _index_digest(pick: Tensor) -> str:
    """A short fingerprint of a resample-index matrix, recorded per cell.

    Exists so "every cell used the same documents" is CHECKABLE after the
    fact rather than a property of how the code happens to be written today.
    """
    flat = pick.reshape(-1).to(torch.int64).cpu().numpy().tobytes()
    return hashlib.sha256(flat).hexdigest()[:16]


def archive_document_bootstrap(
    cells: "Sequence[dict[str, Any]]",
    *,
    top_k: int,
    n_experts: int,
    n_bootstrap: int = 1000,
    n_shuffles_bootstrap: int = 15,
    tokens_per_doc: int,
    n_distinct_documents: int,
    shuffle_scope: str,
    shuffle_seed: int = 0,
    quantity_name: str = "within_selection",
) -> dict[str, Any]:
    """Document-level bootstrap of ARCHIVE-level statistics (MID).

    MID is `3 x` the largest document-level bootstrap standard deviation among
    this project's model checkpoints, and the statistic being bootstrapped is an archive
    level one -- the mean over the archive's grid points, and its worst point --
    not a single grid point.

    ## The one thing that must not be got wrong

    Each draw resamples DOCUMENTS ONCE and uses that same document set for
    EVERY grid point, then aggregates across points. Bootstrapping each point
    separately and combining the results afterwards discards the correlation
    between points -- they share documents, so their errors move together --
    and the combined deviation comes out TOO SMALL.

    That error is invisible in the output: the numbers look entirely normal,
    just tighter. And since MID is defined as three standard deviations, a
    deviation that is too small sets the threshold too low, so differences
    that should read as "no meaningful difference" come out as real ones. This
    function therefore draws the indices itself, records a digest of them on
    every cell's contribution, and asserts the digests agree.

    `cells` is one dict per grid point: `{"centered": (n_tokens, n_experts)
    tensor, "layer_idx": int, "loop_step": int}`. All must share `n_docs`.

    ## The resampling unit is a DOCUMENT, not a document slot

    `n_distinct_documents` is required because the two are not the same number
    whenever more than one probe seed is concatenated. Every seed reads the
    SAME 40 documents and differs only in which window of each it takes
    (`probe_corpus.build_probe_batch` slices `docs[:batch]` before the seed is
    consulted), so four seeds give 160 slots over 40 documents. Resampling the
    160 treats four windows of one document as four independent observations;
    they are not independent, the effective sample size is inflated fourfold,
    and the deviation comes out TOO SMALL -- which, since MID is three times
    that deviation, sets the threshold too low.

    A draw therefore selects DOCUMENTS, and a selected document enters the
    resample with ALL of its windows. Dropping to one window per document
    would align the units too, but by discarding three quarters of the
    measurement -- shrinking the sample for real instead of correcting how it
    is counted, an error in the same direction.

    Because the correction removes degrees of freedom that were never there,
    **the deviation it produces cannot be smaller than the slot-level one**.
    A smaller result means the implementation is wrong, not that the data
    disagreed; `test_document_clustering_cannot_lower_the_deviation` pins it.

    ## `shuffle_scope` is required, and it is not the same knob as `tokens_per_doc`

    `tokens_per_doc` says where the DOCUMENT boundaries are, and the bootstrap
    needs it no matter what -- documents are the thing being resampled. Which
    null the ratio is measured against is a separate question, and the two
    answers are not interchangeable: `"batch"` shuffles each expert's column
    across the whole batch, `"document"` shuffles only within a document, so a
    router that specializes by domain scores differently under them. There is
    no defensible default -- picking one here would be choosing the answer --
    so the caller has to say which null it means. Same values, same meaning,
    same reason as `margin_null_record`'s `shuffle_scope`.
    """
    if shuffle_scope not in ("batch", "document"):
        raise ValueError(f"shuffle_scope must be 'batch' or 'document', got {shuffle_scope!r}.")
    if not cells:
        raise ValueError("archive_document_bootstrap needs at least one grid point")

    slots_seen = {int(c["centered"].shape[0]) // tokens_per_doc for c in cells}
    if len(slots_seen) != 1:
        raise ValueError(
            f"grid points disagree on slot count {sorted(slots_seen)}; they must "
            "be resampled together, which is impossible if they hold different documents."
        )
    n_slots = slots_seen.pop()
    if n_distinct_documents <= 0 or n_slots % n_distinct_documents:
        raise ValueError(
            f"{n_slots} document slots do not divide into {n_distinct_documents} distinct "
            "documents. Every document must contribute the same number of windows, or "
            "there is no well-defined cluster to resample."
        )
    windows_per_document = n_slots // n_distinct_documents
    n_docs = n_distinct_documents
    device = cells[0]["centered"].device
    generator = torch.Generator(device=device)
    generator.manual_seed(shuffle_seed)

    # ONE index matrix for the whole archive.
    pick = _document_resample_indices(n_docs, n_bootstrap, generator, device)
    digest = _index_digest(pick)

    quantity = QUANTITIES[quantity_name]
    # `None` means "shuffle across the whole batch"; an int means "within each
    # document". Same convention as `_column_shuffle`, deliberately reused
    # rather than re-expressed, so the two code paths cannot drift.
    shuffle_blocks = tokens_per_doc if shuffle_scope == "document" else None
    per_cell = torch.empty(len(cells), n_bootstrap, dtype=torch.float64, device=device)
    used_digests: set[str] = set()

    for j, cell in enumerate(cells):
        # (windows_per_document, n_distinct_documents, tokens_per_doc, n_experts).
        # The leading axis is the SEED: the driver concatenates one seed's whole
        # 40-document batch before the next, so slot index `s*n_docs + d` is
        # document `d` seen through seed `s`. Resampling therefore indexes axis
        # 1, which takes a document together with every window of it.
        blocked = cell["centered"].view(
            windows_per_document, n_docs, tokens_per_doc, n_experts
        )
        used_digests.add(digest)
        for b in range(n_bootstrap):
            resampled = blocked[:, pick[b]].reshape(-1, n_experts)
            observed = float(quantity(resampled, top_k).mean().item())
            null = float(
                _null_margin_means(
                    resampled, top_k, n_shuffles_bootstrap, generator, shuffle_blocks
                ).mean().item()
            )
            # Recalibrated INSIDE the draw: holding rho_ref fixed
            # treats it as known without error and understates the spread,
            # which is the quantity MID is defined from.
            surrogate = _structureless_surrogate(resampled, generator)
            surrogate = surrogate - surrogate.mean(dim=-1, keepdim=True)
            sur_obs = float(quantity(surrogate, top_k).mean().item())
            sur_null = float(
                _null_margin_means(
                    surrogate, top_k, n_shuffles_bootstrap, generator, shuffle_blocks
                ).mean().item()
            )
            ref = sur_obs / sur_null if sur_null else float("nan")
            per_cell[j, b] = (observed / null) / ref if null and ref == ref else float("nan")

    if len(used_digests) != 1:
        raise AssertionError(
            f"grid points were resampled with different document sets ({used_digests}). "
            "Every point must see the same documents in a given draw; resampling them "
            "independently discards the correlation between points and makes the "
            "deviation -- and therefore MID -- too small."
        )

    archive_mean = per_cell.mean(dim=0)
    archive_worst = per_cell.min(dim=0).values
    out: dict[str, Any] = {
        "quantity": quantity_name,
        # ⚠️ `n_documents` below is the SLOT count and always was. Its name
        # answers a different question than it appears to, and that is exactly
        # how group A came to be bootstrapped as though four windows of one
        # document were four independent documents. Kept for readers of older
        # artifacts; `n_distinct_documents` is the resampling unit.
        "n_distinct_documents": int(n_docs),
        "n_document_slots": int(n_slots),
        "n_windows_per_document": int(windows_per_document),
        "shuffle_scope": shuffle_scope,
        "n_shuffles_bootstrap": int(n_shuffles_bootstrap),
        "tokens_per_doc": int(tokens_per_doc),
        "shuffle_seed": int(shuffle_seed),
        "n_bootstrap": int(n_bootstrap),
        "n_documents": int(n_slots),
        "n_grid_points": len(cells),
        "resample_index_digest": digest,
        "resample_shared_across_grid_points": True,
    }
    for name, draws in (("archive_mean", archive_mean), ("archive_worst_cell", archive_worst)):
        q = torch.quantile(draws, torch.tensor([0.025, 0.975], dtype=torch.float64, device=device))
        out[name] = {
            "point": float(draws.mean().item()),
            "sd": float(draws.std(unbiased=True).item()),
            "ci_lo": float(q[0].item()),
            "ci_hi": float(q[1].item()),
        }
    return out


def residual_norm_records(
    *,
    step: int,
    residual_by_loop_layer: dict[tuple[int, int], Tensor],
) -> list[dict[str, Any]]:
    """`kind="residual"` rows: the "within-step vs step-boundary" split
    (metric #5), both computed by `looped.step_norm` -- same function,
    different axis, per that function's own docstring ("these are two
    different quantities").

    Args:
        residual_by_loop_layer: `{(loop_step, layer_idx): tensor}`, each
            tensor `(batch, seq, d_model)` -- the block's raw output
            (`_vendored_loopdiag`'s `CAPTURE_BLOCK_OUTPUT` convention: no
            normalization applied). Must cover a full rectangular grid of
            `loop_step in [0, R)` x `layer_idx in [0, L)` for this to
            produce both kinds of rows -- a partial grid (e.g. only the
            last layer) still works but only yields `step_boundary` rows
            for the layers present and `within_step` rows for the loop
            steps present.

    Reduction order:
        - `within_step`: for a fixed `loop_step`, stack the L layers'
          tensors along a new axis 0 (`step_dim=0` in `looped.step_norm`
          terms) in `layer_idx` order, giving `(L, batch, seq, d_model)`;
          `step_norm` computes `||h_{i+1} - h_i||_2` between consecutive
          LAYERS at that fixed loop_step, then averages over
          `(batch, seq)` -- NOT over `layer_idx`, which remains as the
          output's leading axis (length L-1).
        - `step_boundary`: for a fixed `layer_idx`, stack the R loop-step
          tensors along axis 0 in `loop_step` order, giving
          `(R, batch, seq, d_model)`; same `step_norm` call, this time
          diffing consecutive LOOP STEPS at that fixed layer.
        Both use `ord=2` (L2, PLT `delta^(r)`-identical per
        `_vendored_loopdiag/looped.py`'s docstring), `reduce="mean"`.
    """
    if not residual_by_loop_layer:
        return []
    loop_steps = sorted({k[0] for k in residual_by_loop_layer})
    layer_idxs = sorted({k[1] for k in residual_by_loop_layer})
    records: list[dict[str, Any]] = []

    for loop_step in loop_steps:
        layers_present = [li for li in layer_idxs if (loop_step, li) in residual_by_loop_layer]
        if len(layers_present) < 2:
            continue
        stacked = torch.stack([residual_by_loop_layer[(loop_step, li)] for li in layers_present], dim=0)
        norms = _looped.step_norm(stacked, step_dim=0, ord=2, reduce="mean")
        for i, norm_val in enumerate(norms.tolist()):
            records.append(
                {
                    "kind": "residual",
                    "step": step,
                    "residual_kind": "within_step",
                    "loop_step": loop_step,
                    "layer_idx_pair": [layers_present[i], layers_present[i + 1]],
                    "norm_mean": float(norm_val),
                }
            )

    for layer_idx in layer_idxs:
        steps_present = [k for k in loop_steps if (k, layer_idx) in residual_by_loop_layer]
        if len(steps_present) < 2:
            continue
        stacked = torch.stack([residual_by_loop_layer[(k, layer_idx)] for k in steps_present], dim=0)
        norms = _looped.step_norm(stacked, step_dim=0, ord=2, reduce="mean")
        for i, norm_val in enumerate(norms.tolist()):
            records.append(
                {
                    "kind": "residual",
                    "step": step,
                    "residual_kind": "step_boundary",
                    "layer_idx": layer_idx,
                    "loop_step_pair": [steps_present[i], steps_present[i + 1]],
                    "norm_mean": float(norm_val),
                }
            )

    return records


# ---------------------------------------------------------------------------
# #6 -- raw-state cosine similarity across consecutive loop steps.
# ---------------------------------------------------------------------------


def state_cosine_records(
    *,
    step: int,
    residual_by_loop_layer: dict[tuple[int, int], Tensor],
    layer_idxs: list[int] | None = None,
) -> list[dict[str, Any]]:
    """`kind="state_cosine"` rows: `cos(h_k, h_{k+1})` on the RAW state
    (`looped.raw_state_cosine_similarity`, deliberately NOT
    `consecutive_step_angle` -- see that function's docstring on why the two
    answer different questions).

    Sampled at the same N-step frequency as #5 (same free-tier data source
    as residual norms, no separate probe forward needed -- it could be
    placed in the "M step" cheap tier, but that classification tracks cost,
    and this metric's cost is identical to #5's, not to the probe-forward
    metrics'). `layer_idxs`, if given, restricts which layers
    get a row -- purely a JSONL-volume control (e.g. "only the last layer"),
    not a cost decision; omit to cover every layer present in the input.

    Reduction order: for a fixed `layer_idx`, stack the R loop-step tensors
    along axis 0 in `loop_step` order (same construction as
    `residual_norm_records`'s `step_boundary` case); `raw_state_cosine_similarity`
    reduces `(batch, seq)` by mean, leaving one cosine value per consecutive
    `(loop_step, loop_step+1)` pair.
    """
    if not residual_by_loop_layer:
        return []
    loop_steps = sorted({k[0] for k in residual_by_loop_layer})
    all_layers = sorted({k[1] for k in residual_by_loop_layer})
    target_layers = layer_idxs if layer_idxs is not None else all_layers
    records: list[dict[str, Any]] = []

    for layer_idx in target_layers:
        steps_present = [k for k in loop_steps if (k, layer_idx) in residual_by_loop_layer]
        if len(steps_present) < 2:
            continue
        stacked = torch.stack([residual_by_loop_layer[(k, layer_idx)] for k in steps_present], dim=0)
        cos = _looped.raw_state_cosine_similarity(stacked, step_dim=0, reduce="mean")
        for i, cos_val in enumerate(cos.tolist()):
            records.append(
                {
                    "kind": "state_cosine",
                    "step": step,
                    "layer_idx": layer_idx,
                    "loop_step_pair": [steps_present[i], steps_present[i + 1]],
                    "cos_mean": float(cos_val),
                }
            )
    return records


# ---------------------------------------------------------------------------
# #7 -- loss components, every step.
# ---------------------------------------------------------------------------


def loss_record(
    *,
    step: int,
    task_loss: float,
    lb_loss: float,
    lb_loss_factor: float,
    z_loss: float,
    z_loss_factor: float,
    total_loss: float,
    grad_norm: float | None = None,
    lr: float | None = None,
) -> dict[str, Any]:
    """`kind="loss"` row, every step (metric #7 -- the one every-step metric,
    the basic health monitor). `lb_loss_factor`/`z_loss_factor` are logged
    alongside the loss VALUES, not just the values alone -- an aux-loss-vs-E
    ablation needs
    to know which coefficient produced a given loss trajectory, and a
    coefficient that isn't logged can't be reconstructed after the fact.
    All five loss-related fields are plain floats already computed by the
    training loop's own forward/loss function -- this function does no
    numeric work of its own, only shapes the JSONL row.

    `grad_norm` is a **SPARSE OPTIONAL field: present only on the steps
    torchtitan itself logs** (`step == 1 or step % metrics.log_freq == 0`),
    absent on every other step, and absent from every row written by older
    versions of this module. It is sparse by construction, not by accident: torchtitan
    computes the gradient norm as a GPU tensor local to its `train_step` and
    only ever moves it to the host (`grad_norm.item()`) inside its own
    log-step branch. Reusing that already-materialized float is free; making
    it dense would mean adding a per-step GPU->CPU sync to a training loop
    whose throughput we are actively defending. Consumers MUST treat the key
    as may-be-missing (`row.get("grad_norm")`), and anything counting
    "consecutive steps" over it is counting consecutive *logged* steps.

    `lr` is the learning rate actually in force on this step, added later:
    neither this file nor the stdout log carried it, so the training series
    streamed back from the training cluster could only ever report `lr: null`.
    It is READ from the live
    optimizer by the caller, never recomputed from the schedule here. A
    recomputation would agree with the real value right up until they diverged
    -- a resumed run whose scheduler state came back differently, say -- and the
    log would then describe a run that did not happen. Optional, because rows
    written before it was added do not have it; dense on every row written since.
    """
    rec = {
        "kind": "loss",
        "step": step,
        "task_loss": task_loss,
        "lb_loss": lb_loss,
        "lb_loss_factor": lb_loss_factor,
        "z_loss": z_loss,
        "z_loss_factor": z_loss_factor,
        "total_loss": total_loss,
    }
    if grad_norm is not None:
        rec["grad_norm"] = grad_norm
    if lr is not None:
        rec["lr"] = lr
    return rec


# ---------------------------------------------------------------------------
# #4 -- cross-loop-step expert overlap (probe tier).
# ---------------------------------------------------------------------------


def overlap_record(
    *,
    step: int,
    layer_idx: int,
    router_logits_by_loop: Tensor,
    top_k: int,
    input_ids_sha8: str,
) -> dict[str, Any]:
    """`kind="overlap"` row (metric #4), one per (step, layer_idx), from a probe
    forward on the fixed corpus (see `probe_corpus.py`).

    Args:
        router_logits_by_loop: `(n_loops, batch, seq, n_experts)` -- this
            layer's router logits at every loop step, from the SAME probe
            forward pass (all `n_loops` values are needed at once, unlike
            the free-tier router metrics which are called once per
            (layer, loop_step)).
        top_k: passed straight to `moe.cross_loop_step_overlap` (which does
            its own top-k internally from raw scores -- see that function
            for why it re-derives top-k rather than accepting `topk_idx`,
            input_type="scores" is used here for exactly that reason).
        input_ids_sha8: the probe batch's input-id hash (from
            `probe_corpus.build_probe_batch`'s `token_source`) -- carried on
            every row so overlap numbers from different training runs are
            only compared once this hash is confirmed equal (changing the
            corpus loses cross-checkpoint comparability).

    Reduction order: `moe.cross_loop_step_overlap` flattens `(batch, seq)`
    into one token axis internally, top-k's each `(step, token)` slice, then
    for every pair of loop steps `(a, b)` computes mean-over-tokens overlap
    (`|top-k(a) ∩ top-k(b)| / k`). Output is the full `(n_loops, n_loops)`
    matrix, NOT reduced further (the matrix is the only carrier of this
    information; compressing it is equivalent to not collecting it).
    """
    overlap = _moe.cross_loop_step_overlap(router_logits_by_loop, input_type="scores", k=top_k)
    return {
        "kind": "overlap",
        "step": step,
        "layer_idx": layer_idx,
        "n_loops": int(overlap.shape[0]),
        "overlap_matrix": overlap.tolist(),
        "input_ids_sha8": input_ids_sha8,
    }


# ---------------------------------------------------------------------------
# #8 -- repetition-rate-stratified probe loss (probe tier).
# ---------------------------------------------------------------------------


def repeat_stratified_loss_records(
    *,
    step: int,
    per_token_loss: Tensor,
    doc_buckets: list[int],
    seq: int,
    n_buckets: int,
    input_ids_sha8: str,
) -> list[dict[str, Any]]:
    """`kind="repeat_stratified_loss"` rows (metric #8), one per repetition-rate
    bucket present in this probe batch.

    Args:
        per_token_loss: `(batch, seq)` per-token cross-entropy loss from the
            probe forward, `batch` in the same document order as
            `doc_buckets`.
        doc_buckets: length-`batch` list, `doc_buckets[i]` = bucket index of
            document i (from `probe_corpus.compute_repeat_rate_buckets`,
            precomputed once at training start -- NOT recomputed here, see
            that function's docstring).
        seq: must match `per_token_loss.shape[1]` (defensive check).
        n_buckets: total bucket count (for `mean over documents in this
            bucket` denominators even when a bucket happens to have 0 docs
            in some slice -- not expected with the fixed 40-doc corpus but
            asserted rather than silently producing a shorter row list).
        input_ids_sha8: same provenance-carrying convention as `overlap_record`.

    Reduction order: mean over ALL tokens (seq axis) AND all documents
    (batch axis) whose bucket matches, in one flat mean -- not a per-document
    mean-of-means. A bucket with more/longer-effective-loss documents
    contributes proportionally more weight; this matches how the un-stratified
    LM loss is normally reported (mean over all tokens) and keeps the
    per-bucket numbers comparable to that baseline.
    """
    if per_token_loss.shape[1] != seq:
        raise ValueError(
            f"per_token_loss seq dim {per_token_loss.shape[1]} != seq={seq} -- "
            "doc_buckets was computed for a different seq length."
        )
    if len(doc_buckets) != per_token_loss.shape[0]:
        raise ValueError(
            f"doc_buckets has {len(doc_buckets)} entries, per_token_loss has "
            f"batch={per_token_loss.shape[0]} -- must be one bucket id per document, "
            "in the same order the probe batch was built."
        )
    records: list[dict[str, Any]] = []
    for bucket in range(n_buckets):
        idxs = [i for i, b in enumerate(doc_buckets) if b == bucket]
        if not idxs:
            continue
        losses = per_token_loss[idxs]
        records.append(
            {
                "kind": "repeat_stratified_loss",
                "step": step,
                "repeat_rate_bucket": bucket,
                "n_buckets": n_buckets,
                "n_docs": len(idxs),
                "mean_loss": float(losses.mean().item()),
                "input_ids_sha8": input_ids_sha8,
            }
        )
    return records

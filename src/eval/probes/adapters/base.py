"""Router-score adapters for open-source HuggingFace MoE models.

Purpose (infra outline 2.4(2), "open-model adapter layer" gap): put this
project's expert-collapse metrics onto someone else's MoE. The metric
formulas live in `src.metrics.collector` (and, beneath it,
`_vendored_loopdiag`) and are NOT reimplemented here. What this module owns
is only:

1. locating the router module inside a third-party model and capturing, via
   a forward hook, the exact tensors that model used -- the pre-softmax
   router scores, the expert indices it actually dispatched to, and the
   combine weights it actually multiplied by;
2. a **self-check** (hard requirement of the task brief) that recomputes the
   model's own top-k from the captured scores and reconciles it against the
   captured decision token by token, raising on any disagreement that is
   not an exact tie -- so a wrong tensor, a wrong unpack order, a
   softmax-before/after mix-up, or a normalized/unnormalized weight mix-up
   can never produce a plausible-looking number;
3. an explicit, all-fields-required record (`RoutingSpec`) of the other
   model's routing and load-balancing mechanism, because it decides which
   metrics are valid there (an expert-bias balancing mechanism makes
   MaxVio_mass blind).

## Contract with `collector.py`

- `router_layer_loop_record` receives `topk_weights` **renormalized to sum
  to 1 over the k slots** -- that is the `TracePoint.topk_weights`
  convention and keeps MaxVio_mass comparable to this project's own models.
  For a model whose forward multiplies by UN-normalized weights (OLMoE,
  `norm_topk_prob=False`) the mass computed from the weights the model
  really used is reported separately in a `kind="adapter_router_extra"`
  row, never by changing the `router` row's meaning.
- `zloss_state_record` receives the pre-bias affinity (the quantity the
  model's own z-loss penalizes); `router_layer_loop_record` /
  the collector's router records receive the post-bias scores (the ones that decided
  the top-k). For every model in the first batch the two are the SAME
  tensor (no expert-bias mechanism, verified from source), and `variant`
  is `"not_applicable"`; the two slots exist so an expert-bias model can be
  plugged in later without touching the pipeline. `variant` uses the
  controlled list `pre_bias` / `post_bias` / `not_applicable`.
- `loop_step` is always 0: these are non-looped models, and `0` means "this
  axis has one value", not "this axis does not apply".
- Existing kinds' fields are not changed; rows written by the adapter carry
  three ADDITIONAL keys (`model_id`, `revision`, `variant`) so every line
  stays self-contained.

## Precision

Probe forwards run in float32: this project's own
probes compute router scores in float32, so float32 here is alignment, not
a favour. bf16 has an 8-bit mantissa; router scores of magnitude 1-10 are
then spaced 0.004-0.04 apart, the same order as the margin itself, and exact
ties become common. `RoutingSpec.forward_dtype` records what was used.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from src.metrics import collector as _collector
from src.metrics._vendored_loopdiag import moe as _moe

VARIANTS = ("pre_bias", "post_bias", "not_applicable")
# A stated coefficient's source must OPEN with one of these, so the authority is declared in a
# fixed place rather than inferred from prose. Colons included: "paper not consulted" cannot
# accidentally satisfy "paper:".
COEFFICIENT_AUTHORITIES = ("paper:", "report:", "model card:")

SCORING_ORDERS = ("softmax_all_then_topk", "topk_then_softmax_selected")
# Controlled vocabularies. The single source of
# truth is `src.metrics.collector`; nothing is mirrored here, so a
# spelling can never drift between producer and consumer.
LOAD_BALANCING_MECHANISMS = _collector.LOAD_BALANCING_MECHANISMS
ROUTER_MODULE_KINDS = _collector.ROUTER_MODULE_KINDS


class RouterReconciliationError(RuntimeError):
    """The captured routing decision does not match what the captured scores
    imply. Raised, never logged: a mismatch means the wrong tensor was
    captured or unpacked, and every downstream number would be quietly
    wrong."""


class RouterCoverageError(RuntimeError):
    """Router modules found/fired do not match the model's declared depth."""


@dataclass(frozen=True)
class RoutingSpec:
    """Explicit description of the other model's routing + balancing, plus the
    run-side facts the comparison requires. Every field is required (no defaults):
    an omitted field would become a silent guess. Unknown training-time
    facts are the STRING `"unknown"`, never `0` (asserts "not used") and
    never `None`.

    `sources` maps EVERY other field name to where its
    value was actually read from -- `config.json:<key>`, a paper section, a
    library default, a file header, or a run setting. A field without a
    source raises. Aux-loss / z-loss coefficients must cite the paper, never
    a branch config (rule one: OLMoE's branch configs contradict each other)."""

    model_id: str
    revision: str
    revision_commit: str
    forward_dtype: str            # == probe_dtype: dtype of the parameters actually loaded
    probe_dtype: str
    probe_dtype_requested: str    # the --probe-dtype flag as typed (shared probe_dtype contract)
    release_dtype: str            # == model_native_dtype: dtype in the released weight files
    model_native_dtype: str
    n_experts: int
    top_k: int
    num_moe_layers: int
    num_layers: int
    num_stacks: int
    scoring_order: str
    scores_are_probs: bool
    gate_bias: bool
    expert_bias_balancing: bool
    load_balancing_mechanism: str
    shared_experts: int
    combine_weights_normalized_by_model: bool
    aux_loss_coef: float | str
    z_loss_coef: float | str
    z_loss_coefficient: float | str
    dropless: bool | str
    router_module_class: str
    router_module_kind: str
    router_module_paths: tuple
    tokenizer: str
    n_tokens_total: int
    special_tokens_excluded: int | str
    transformers_version: str
    variant: str
    sources: dict

    def __post_init__(self) -> None:
        for f in fields(self):
            if getattr(self, f.name) is None:
                raise ValueError(f"RoutingSpec.{f.name} is None; every field must be stated explicitly.")
        if self.scoring_order not in SCORING_ORDERS:
            raise ValueError(f"scoring_order={self.scoring_order!r} not in {SCORING_ORDERS}")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant={self.variant!r} not in {VARIANTS}")
        # collector's own validator: raises on anything outside its vocabularies
        _collector.validate_router_metadata(
            load_balancing_mechanism=self.load_balancing_mechanism, router_module_kind=self.router_module_kind,
        )
        for name in ("aux_loss_coef", "z_loss_coef", "z_loss_coefficient", "dropless", "special_tokens_excluded"):
            v = getattr(self, name)
            if isinstance(v, str) and v != "unknown":
                raise ValueError(f"{name}={v!r}: a string value must be exactly 'unknown'.")
        if self.z_loss_coefficient != self.z_loss_coef or self.probe_dtype != self.forward_dtype \
                or self.model_native_dtype != self.release_dtype or self.num_layers != self.num_moe_layers:
            raise ValueError("alias fields disagree (z_loss_coefficient/z_loss_coef, probe_dtype/forward_dtype, "
                             "model_native_dtype/release_dtype, num_layers/num_moe_layers)")
        if self.top_k < 1 or self.n_experts < self.top_k:
            raise ValueError(f"top_k={self.top_k} n_experts={self.n_experts} inconsistent")
        if len(self.router_module_paths) != self.num_moe_layers:
            raise ValueError(f"router_module_paths has {len(self.router_module_paths)} entries, num_moe_layers={self.num_moe_layers}")
        missing = [f.name for f in fields(self) if f.name != "sources" and f.name not in self.sources]
        if missing:
            raise ValueError(f"RoutingSpec.sources lacks an entry for: {missing} (every field needs its source)")
        # A training coefficient must say where its number came from. The check is on the
        # source's DECLARED authority, not on which words appear in its prose.
        #
        # Testing for words failed in both directions. It rejected "unknown: not stated in
        # repo config or model card" -- a sentence saying the config does NOT carry the
        # value -- because the word "config" occurs in it. And it accepted "...; paper not
        # consulted for a number", a sentence saying no paper was consulted, because the
        # word "paper" occurs in it. The second direction is the dangerous one: it does not
        # crash, so a number with no support behind it travels on wearing a checked-source
        # label.
        #
        # So: a value of "unknown" cites nothing and there is nothing to police, but its
        # source has to admit that. A real number's source must OPEN with the authority it
        # rests on, which a sentence denying that authority cannot do.
        #
        # The limit, stated rather than papered over: this checks that a source declares an
        # authority, not that the declaration is true. "paper: not consulted" would pass.
        # That is a false statement by the adapter's author, which no string test can catch;
        # what is caught is the far commoner case of prose that never declares one at all.
        for name in ("aux_loss_coef", "z_loss_coef", "z_loss_coefficient"):
            value, src = getattr(self, name), str(self.sources[name]).strip()
            if value == "unknown":
                if not src.lower().startswith("unknown"):
                    raise ValueError(
                        f"{name} is 'unknown' but sources[{name}]={self.sources[name]!r} does not say so. "
                        "A source that reads as a citation while the value is unknown invites a later "
                        "reader to treat the number as sourced. Open it with 'unknown'."
                    )
                continue
            if not src.lower().startswith(COEFFICIENT_AUTHORITIES):
                raise ValueError(
                    f"sources[{name}]={self.sources[name]!r}: a stated coefficient ({value!r}) must open with "
                    f"one of {COEFFICIENT_AUTHORITIES} naming where the number came from. A branch config is "
                    "an inference-time artifact, not a training fact, and prose that merely mentions a paper "
                    "does not cite one."
                )

    def to_row(self) -> dict[str, Any]:
        row = {"kind": "adapter_meta", **{f.name: getattr(self, f.name) for f in fields(self)}}
        row["router_module_paths"] = list(self.router_module_paths)
        return row


@dataclass(frozen=True)
class RouterCapture:
    """What one router module produced for one forward, reshaped to
    `(batch, seq, ...)`. All tensors detached, on the model's device.

    `scores_pre_bias` / `scores_post_bias` are the same tensor object for a
    model without an expert-bias mechanism (see module docstring)."""

    layer_idx: int
    module_path: str
    scores_pre_bias: Tensor
    scores_post_bias: Tensor
    topk_idx: Tensor
    topk_weights_model: Tensor
    topk_weights_renorm: Tensor
    score_kplus1: Tensor
    """[B, S] the (k+1)-th largest post-bias score per token (R7: "selection
    separation" = k-th - (k+1)-th needs it). Same tensor, one more order
    statistic; zero extra forward."""
    tie_token_count: int

    @property
    def router_logits(self) -> Tensor:
        return self.scores_post_bias


class HFRouterAdapter:
    """Base class. A subclass states, for one HF model family:

    - `router_class_name`: the router module's class name (located by class,
      never by attribute path -- attribute names drift across transformers
      versions, class names have not);
    - `router_path_filter`: optional substring the module path must contain
      (JetMoE reuses one gating class for attention experts AND MLP experts;
      only `.mlp.` routers are wanted);
    - `unpack(output)`: the hook output tuple -> `(logits, topk_idx,
      topk_weights_model)` in that model's own order;
    - `recompute(logits, k)`: that model's own selection rule ->
      `(topk_idx, topk_weights_model)`;
    - `router_weight(module)`: the gate's weight matrix, for the
      captured-scores-came-from-this-input check;
    - `routing_spec(...)`: the `RoutingSpec` for that family.
    """

    router_class_name: str = ""
    router_path_filter: str | None = None
    variant: str = "not_applicable"

    # ---- subclass surface -------------------------------------------------
    def unpack(self, output: tuple) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def recompute(self, logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def router_weight(self, module: nn.Module) -> Tensor:
        return module.weight

    def top_k(self, config: Any) -> int:
        return int(config.num_experts_per_tok)

    # ---- locating routers -------------------------------------------------
    def find_routers(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        found = [
            (name, m)
            for name, m in model.named_modules()
            if type(m).__name__ == self.router_class_name
            and (self.router_path_filter is None or self.router_path_filter in name)
        ]
        expected = int(model.config.num_hidden_layers)
        if len(found) != expected:
            raise RouterCoverageError(
                f"found {len(found)} x {self.router_class_name}"
                f"{'' if self.router_path_filter is None else ' under ' + self.router_path_filter!r}"
                f" but config.num_hidden_layers={expected}; refusing to guess which layers are MoE."
            )
        return found

    # ---- capture + self-check ---------------------------------------------
    @torch.no_grad()
    def capture(self, model: nn.Module, input_ids: Tensor, *, doc_chunk: int | None = None) -> list[RouterCapture]:
        """One `RouterCapture` per router, in layer order.

        `doc_chunk` splits the batch into groups of that many DOCUMENTS, runs one
        forward per group, and concatenates the captures. It is a memory strategy and
        nothing else: the captures are joined back into the same tensors a single
        forward would have produced, and every metric is computed once over the joined
        result -- never per chunk and averaged, which would be a different statistic.

        It exists because one forward over the 500-document corpus is 256,000 tokens and
        runs out of memory on an 80GB card, while the captures themselves are a fraction
        of that -- the activations are what does not fit, and they are freed per chunk.

        Raises `RouterCoverageError` / `RouterReconciliationError`.
        """
        n_docs = input_ids.shape[0]
        if doc_chunk is None or doc_chunk >= n_docs:
            return self._capture_batch(model, input_ids)
        if doc_chunk < 1:
            raise ValueError(f"doc_chunk={doc_chunk} must be at least 1")
        parts = [self._capture_batch(model, input_ids[i:i + doc_chunk])
                 for i in range(0, n_docs, doc_chunk)]
        return self._join(parts)

    @staticmethod
    def _join(parts: list[list[RouterCapture]]) -> list[RouterCapture]:
        """Concatenate per-chunk captures back into whole-batch ones, document-wise."""
        first = parts[0]
        for other in parts[1:]:
            if [c.module_path for c in other] != [c.module_path for c in first]:
                raise RouterCoverageError(
                    "chunks saw different routers; the layer grid must not depend on the batch"
                )
        joined = []
        for i, c0 in enumerate(first):
            cs = [p[i] for p in parts]
            scores = torch.cat([c.scores_post_bias for c in cs], dim=0)
            pre = torch.cat([c.scores_pre_bias for c in cs], dim=0)
            joined.append(RouterCapture(
                layer_idx=c0.layer_idx,
                module_path=c0.module_path,
                # Same object when the family has no expert-bias mechanism, as in a
                # single-forward capture; rebuilt here rather than assumed.
                scores_pre_bias=scores if all(c.scores_pre_bias is c.scores_post_bias for c in cs) else pre,
                scores_post_bias=scores,
                topk_idx=torch.cat([c.topk_idx for c in cs], dim=0),
                topk_weights_model=torch.cat([c.topk_weights_model for c in cs], dim=0),
                topk_weights_renorm=torch.cat([c.topk_weights_renorm for c in cs], dim=0),
                score_kplus1=torch.cat([c.score_kplus1 for c in cs], dim=0),
                tie_token_count=sum(c.tie_token_count for c in cs),
            ))
        return joined

    def _capture_batch(self, model: nn.Module, input_ids: Tensor) -> list[RouterCapture]:
        """One forward; returns one `RouterCapture` per router, in layer
        order. Raises `RouterCoverageError` / `RouterReconciliationError`."""
        model.eval()
        routers = self.find_routers(model)
        batch, seq = input_ids.shape
        k = self.top_k(model.config)
        hits: dict[int, list[tuple[Tensor, tuple]]] = {i: [] for i in range(len(routers))}

        def make_hook(i: int):
            def hook(mod: nn.Module, inp: tuple, out: tuple) -> None:
                hits[i].append((inp[0].detach(), tuple(o.detach() if torch.is_tensor(o) else o for o in out)))
            return hook

        handles = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(routers)]
        try:
            model(input_ids=input_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        caps: list[RouterCapture] = []
        for i, (name, mod) in enumerate(routers):
            if len(hits[i]) != 1:
                raise RouterCoverageError(f"router {name} fired {len(hits[i])} times in one forward (expected exactly 1)")
            hidden, out = hits[i][0]
            logits, idx, w_model = self.unpack(out)
            caps.append(self._reconcile(name, mod, hidden, logits, idx, w_model, k, batch, seq))
        return caps

    def _reconcile(
        self, name: str, mod: nn.Module, hidden: Tensor, logits: Tensor, idx: Tensor, w_model: Tensor,
        k: int, batch: int, seq: int,
    ) -> RouterCapture:
        n_tok = batch * seq
        if logits.ndim != 2 or logits.shape[0] != n_tok:
            raise RouterReconciliationError(f"{name}: scores shape {tuple(logits.shape)} != ({n_tok}, E)")
        if idx.shape != (n_tok, k) or w_model.shape != (n_tok, k):
            raise RouterReconciliationError(
                f"{name}: topk_idx {tuple(idx.shape)} / weights {tuple(w_model.shape)} != ({n_tok}, {k})"
            )
        logits = logits.float()
        # (a) the scores really came from this input through this gate.
        relinear = torch.nn.functional.linear(hidden.reshape(n_tok, -1).float(), self.router_weight(mod).float())
        if not torch.allclose(relinear, logits, rtol=1e-4, atol=1e-5):
            raise RouterReconciliationError(
                f"{name}: captured scores differ from input @ gate weight by up to "
                f"{(relinear - logits).abs().max().item():.3e}; wrong tensor captured?"
            )
        # (b) the model's own rule, re-applied to the captured scores, must give
        #     the captured decision -- as a SET per token, plus weights.
        r_idx, r_w = self.recompute(logits, k)
        set_ok = (idx.sort(dim=-1).values == r_idx.sort(dim=-1).values).all(dim=-1)
        topk1 = logits.topk(k + 1, dim=-1).values if logits.shape[-1] > k else None
        tie = (topk1[:, k - 1] == topk1[:, k]) if topk1 is not None else torch.zeros(n_tok, dtype=torch.bool, device=logits.device)
        bad = (~set_ok) & (~tie)
        if bad.any():
            t = int(bad.nonzero()[0].item())
            raise RouterReconciliationError(
                f"{name}: {int(bad.sum())} tokens' top-{k} set differs from recomputation and is not a tie; "
                f"first token {t}: model={idx[t].tolist()} recomputed={r_idx[t].tolist()}"
            )
        w_sorted = w_model.float().sort(dim=-1).values
        rw_sorted = r_w.float().sort(dim=-1).values
        ok_rows = ~tie
        if not torch.allclose(w_sorted[ok_rows], rw_sorted[ok_rows], rtol=1e-4, atol=1e-6):
            raise RouterReconciliationError(
                f"{name}: combine weights differ from recomputation by up to "
                f"{(w_sorted[ok_rows] - rw_sorted[ok_rows]).abs().max().item():.3e}; "
                "normalized/unnormalized mix-up or softmax order wrong?"
            )
        w_model_f = w_model.float()
        renorm = w_model_f / w_model_f.sum(dim=-1, keepdim=True)
        E = logits.shape[-1]
        if E <= k:
            raise RouterReconciliationError(f"{name}: n_experts={E} <= top_k={k}; the (k+1)-th score does not exist")
        scores = logits.reshape(batch, seq, E)
        kplus1 = logits.topk(k + 1, dim=-1).values[:, k].reshape(batch, seq)
        return RouterCapture(
            layer_idx=int(name.split(".layers.")[1].split(".")[0]) if ".layers." in name else -1,
            module_path=name,
            scores_pre_bias=scores,   # same object: no expert-bias mechanism in this family
            scores_post_bias=scores,
            topk_idx=idx.reshape(batch, seq, k).long(),
            topk_weights_model=w_model_f.reshape(batch, seq, k),
            topk_weights_renorm=renorm.reshape(batch, seq, k),
            score_kplus1=kplus1,
            tie_token_count=int(tie.sum()),
        )

    # ---- rows ---------------------------------------------------------------
    def rows_for_capture(
        self, cap: RouterCapture, *, step: int, spec: RoutingSpec, extra: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """`router` + `zloss_state` rows via collector (contract weights =
        renormalized), plus one `adapter_router_extra` row with the mass
        computed from the weights the model actually multiplied by and the
        per-token "selection separation" (k-th minus (k+1)-th score) summary.

        All reductions run on CPU copies: `expert_mass_distribution`'s
        `scatter_add_` is nondeterministic on CUDA (measured: ~3e-6 relative
        drift between two identical runs), and products must be
        byte-identical across runs.

        `scores_are_probs=True`: the margin-family fields are
        OMITTED (not zeroed, not nulled) and a reason is recorded -- those
        quantities are not scale-invariant under softmax."""
        tag = {**row_tags(spec), **(extra or {})}
        E, k = spec.n_experts, spec.top_k
        scores = cap.scores_post_bias.detach().cpu()
        pre = cap.scores_pre_bias.detach().cpu()
        idx = cap.topk_idx.detach().cpu()
        w_renorm = cap.topk_weights_renorm.detach().cpu()
        w_model = cap.topk_weights_model.detach().cpu()
        router = _collector.router_layer_loop_record(
            step=step, layer_idx=cap.layer_idx, loop_step=0, router_logits=scores,
            topk_idx=idx, topk_weights=w_renorm, n_experts=E, top_k=k,
        )
        zrow = _collector.zloss_state_record(step=step, layer_idx=cap.layer_idx, loop_step=0, router_logits=pre)
        load, _ = _moe.expert_load_distribution(idx, E)
        mass_model = _moe.expert_mass_distribution(idx, w_model, E)
        mv_count = float(_moe.maxvio(load).item())
        mv_mass_model = float(_moe.maxvio(mass_model).item())
        kth = scores.topk(k, dim=-1).values[..., -1]
        sep = (kth - cap.score_kplus1.detach().cpu()).reshape(-1)
        extra_row = {
            "kind": "adapter_router_extra",
            "step": step, "layer_idx": cap.layer_idx, "loop_step": 0,
            "n_experts": E, "top_k": k, "n_tokens": int(idx.shape[0] * idx.shape[1]),
            "module_path": cap.module_path,
            "aggregated_quantity": "mass_model_weights",
            "mass_reduction_device": "cpu",  # CUDA scatter_add_ is nondeterministic
            "combine_weights_normalized_by_model": spec.combine_weights_normalized_by_model,
            "maxvio_mass_model_weights": mv_mass_model,
            "maxvio_mass_to_count_ratio_model_weights": (mv_mass_model / mv_count) if mv_count > 1e-6 else None,
            "tie_token_count": cap.tie_token_count,
        }
        if spec.scores_are_probs:
            reason = ("scores_are_probs=true: margin-family quantities (top1-topk, kth-(k+1)th) are not "
                      "scale-invariant under softmax; omitted rather than computed on probabilities")
            router = {kk: v for kk, v in router.items() if not kk.startswith("margin")}
            router["margin_fields_omitted_reason"] = reason
            extra_row["separation_fields_omitted_reason"] = reason
        else:
            extra_row["separation_mean"] = float(sep.mean().item())
            extra_row["separation_min"] = float(sep.min().item())
            extra_row["separation_tie_ratio"] = float((sep == 0).float().mean().item())
        return [{**router, **tag, "module_path": cap.module_path, "tie_token_count": cap.tie_token_count, "mass_reduction_device": "cpu"},
                {**zrow, **tag, "module_path": cap.module_path}, {**extra_row, **tag}]


def row_tags(spec: RoutingSpec) -> dict[str, Any]:
    """Keys every adapter-written row carries so it is self-describing once
    it leaves the file: identity is
    `model_id` + `revision_commit` (a resolved commit hash, never a branch
    name -- branches move), and the two facts that decide whether a
    `maxvio_mass` / margin number is the same construction as another
    model's ride along with the number, not only in `adapter_meta`."""
    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "revision_commit": spec.revision_commit,
        "variant": spec.variant,
        "scoring_order": spec.scoring_order,
        "scores_are_probs": spec.scores_are_probs,
        "combine_weights_normalized_by_model": spec.combine_weights_normalized_by_model,
    }


def pooled_scores(captures_by_seed: Iterable[list[RouterCapture]]) -> dict[int, Tensor]:
    """Concatenate each layer's post-bias scores across probe seeds into the
    2-D `(n_tokens, n_experts)` form the collector's records require. Grids
    must match exactly across seeds."""
    pooled: dict[int, list[Tensor]] = {}
    layers_ref: list[int] | None = None
    for caps in captures_by_seed:
        layers = [c.layer_idx for c in caps]
        if layers_ref is None:
            layers_ref = layers
        elif layers != layers_ref:
            raise RouterCoverageError(f"layer grid differs across seeds: {layers} vs {layers_ref}")
        for c in caps:
            pooled.setdefault(c.layer_idx, []).append(c.scores_post_bias.reshape(-1, c.scores_post_bias.shape[-1]))
    return {li: torch.cat(v, dim=0) for li, v in pooled.items()}


def pooled_kplus1(captures_by_seed: Iterable[list[RouterCapture]]) -> dict[int, Tensor]:
    """Concatenate each layer's (k+1)-th scores across seeds to `(n_tokens,)`,
    in the same token order as `pooled_scores`."""
    pooled: dict[int, list[Tensor]] = {}
    for caps in captures_by_seed:
        for c in caps:
            pooled.setdefault(c.layer_idx, []).append(c.score_kplus1.reshape(-1))
    return {li: torch.cat(v, dim=0) for li, v in pooled.items()}


def self_check_margin_known_answers() -> dict[str, Any]:
    """Two known-answer assertions on the SAME code path the
    products use (`_moe.topk_margin`): all-equal scores -> margin exactly 0;
    top-1 raised by delta above an otherwise tied field -> exactly delta."""
    E, k, delta = 16, 4, 0.75
    tied = torch.zeros(3, E)
    m0 = _moe.topk_margin(tied, k)
    boosted = tied.clone(); boosted[:, 5] += delta
    m1 = _moe.topk_margin(boosted, k)
    ok0 = bool((m0 == 0).all()); ok1 = bool((m1 == delta).all())
    if not (ok0 and ok1):
        raise RuntimeError(f"margin known-answer self-check failed: all-equal -> {m0.tolist()}, +delta -> {m1.tolist()}")
    return {"all_equal_gives_zero": ok0, "top1_plus_delta_gives_delta": ok1, "delta": delta, "E": E, "k": k}

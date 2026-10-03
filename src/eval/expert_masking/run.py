"""Expert-masking evaluation: shares on corpus A, losses on corpus B.

Procedure version 1 (see README, Expert masking).

Per model: one forward on corpus A to get the shares, then 1 + 3 + 3*5 = 19
forwards on corpus B (unmasked, each target tier, five random controls per
tier). Inference only; no weights change.
"""
from __future__ import annotations

import argparse, contextlib, json, socket, subprocess, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.probe_corpus import (
    LARGE_CORPUS_B_SMOLLM2_ID, LARGE_CORPUS_B_SMOLLM2_PATH, PROBE_SEQ, build_probe_batch,
)
from src.eval.expert_masking.masking import loop_layers, masked_routing
from src.eval.expert_masking.selection import (
    layer_sizes, leaves_enough, per_layer_min_remaining, random_controls,
    shares_by_physical_expert, target_set,
)
from src.eval.probes.offline_cross_arch_probe import (
    CORPORA, exclusion_rule_for_tokenizer, resolve_tokenizer,
)
from src.eval.probes.probe_dtype import load_kwargs
from src.metrics.provenance import code_revision, machine_provenance

#: Corpora this evaluation may name. The probe driver's registry plus corpus B
#: (`probe_corpus.LARGE_CORPUS_B_*`, 485 documents disjoint from corpus A by
#: sha256). Path and id are the corpus module's own constants, so
#: this adds a key, not a second copy of the identity.
MASKING_CORPORA = {
    **CORPORA,
    "large500B_smollm2": (LARGE_CORPUS_B_SMOLLM2_PATH, LARGE_CORPUS_B_SMOLLM2_ID, "smollm2"),
}

#: Recorded next to every rho so the number cannot travel without it.
FEASIBILITY_RULE = (
    "Every masking set (the target set T and each random control R) must leave at least top_k usable experts in every layer. "
    "A random-control seed that violates this is skipped in deterministic order and the next seed is used (recorded in skipped_seeds); if T violates it, that tier is undefined for this model. "
    "Reason: one control once masked all 4 experts of a layer and the loss was NaN."
)

RHO_INTERPRETATION_NOTE = (
    "Read ρ only within one model (the ratio of that model's rarely used expert set T to its own random controls R). "
    "Do not compare across architectures: models differ in the scale of ΔL, the number of experts and the share distribution, so a difference between two models' ρ is undefined."
)


@contextlib.contextmanager
def recording_selections(model):
    """Collect each loop layer's chosen experts, summed over the loop passes.

    A layer is entered once per pass, so the hook fires `num_stacks` times per
    layer per forward; adding them up is exactly the merge section 2 asks for.
    """
    layers = loop_layers(model)
    counts = {i: None for i in range(len(layers))}
    picks = {i: [] for i in range(len(layers))}
    restore = []
    for i, layer in enumerate(layers):
        router = layer.router
        width = router.gate.weight.shape[0]
        counts[i] = np.zeros(width, dtype=np.int64)

        def make(router=router, i=i):
            # Wrap whatever this router currently resolves to, NOT
            # `type(router).forward`. When a mask is already installed it lives
            # as an INSTANCE attribute, and going to the class would call the
            # unmasked implementation -- the recording would silently undo the
            # masking, every masked pass would equal the baseline, and the only
            # symptom is that every delta comes out exactly 0.000000.
            original = router.forward

            def forward(x):
                logits, probs, top, idx = original(x)
                flat = idx.reshape(-1).detach().cpu().numpy()
                np.add.at(counts[i], flat, 1)
                picks[i].append(idx.detach().cpu().numpy().copy())
                return logits, probs, top, idx
            return forward

        restore.append((router, "forward" in router.__dict__))
        router.forward = make()
    try:
        yield counts, picks
    finally:
        for router, had_own in restore:
            if not had_own:
                del router.forward


def _batch(tokenizer, corpus_key, n_documents):
    """Build one corpus's probe batch.

    `n_documents` is required and has no default. Corpus A and corpus B need
    not be the same size -- corpus B is "the next eligible documents after A's
    picks" and came out at 485, not 500 -- and a hardcoded 500 would make the
    whole run die at the first forward on any corpus that has fewer. The
    document count also sets N, which every loss here is averaged over, so it
    is a semantic parameter and not something to infer silently.
    """
    corpus_path, corpus_id, tokenizer_name = MASKING_CORPORA[corpus_key]
    ids, source = build_probe_batch(
        tokenizer, corpus_path, batch=n_documents, seq=PROBE_SEQ,
        exclusion_rule=exclusion_rule_for_tokenizer(tokenizer_name), seed=0,
    )
    return ids, corpus_id, source


def forward_pass(model, ids, device, doc_chunk, record=False):
    """Per-token losses, and optionally the routing choices, over one corpus."""
    per_token, all_picks = [], []
    with torch.no_grad():
        for start in range(0, ids.shape[0], doc_chunk):
            chunk = ids[start:start + doc_chunk].to(device)
            ctx = recording_selections(model) if record else contextlib.nullcontext((None, None))
            with ctx as (counts, picks):
                out = model(chunk)
                logits = out.logits if hasattr(out, "logits") else out[0]
            losses = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                chunk[:, 1:].reshape(-1), reduction="none",
            )
            per_token.append(losses.view(chunk.shape[0], chunk.shape[1] - 1).cpu().numpy().astype("float32"))
            if record:
                # The hook fires once per loop pass, in pass order, so the list
                # for a layer is [pass0, pass1, ...], each (chunk_docs, S, k).
                # Kept as a pass axis: `(num_stacks, docs, S, k)`.
                all_picks.append({k: np.stack(v, axis=0) for k, v in picks.items()})
                if start == 0:
                    merged = {k: c.copy() for k, c in counts.items()}
                else:
                    for k, c in counts.items():
                        merged[k] += c
    losses = np.concatenate(per_token, axis=0)
    if not record:
        return losses, None, None
    picks = {k: np.concatenate([p[k] for p in all_picks], axis=1) for k in all_picks[0]}
    return losses, merged, picks


def reassignments_per_token(baseline_picks: dict, experts) -> np.ndarray:
    """Per token `(docs, S)`: over every loop position (layer x pass), how many of
    the UNMASKED top-k experts belong to the masked set.

    Procedure section 3: a token is "reassigned" at a position when one of the
    experts it would have used is masked there, and the count adds over
    positions. It is a property of the baseline routing and the set, so it
    needs no masked forward. `baseline_picks[layer]` is `(num_stacks, docs, S, k)`.
    """
    by_layer: dict[int, set[int]] = {}
    for layer, expert in experts:
        by_layer.setdefault(int(layer), set()).add(int(expert))
    first = next(iter(baseline_picks.values()))
    if first.ndim != 4:
        raise SystemExit(f"REFUSING: picks must be (passes, docs, S, k); got {first.shape}")
    total = np.zeros(first.shape[1:3], dtype=np.int32)
    for layer, picks in baseline_picks.items():
        masked = by_layer.get(int(layer))
        if not masked:
            continue
        hit = np.isin(picks, sorted(masked))            # (passes, docs, S, k)
        total += hit.sum(axis=(0, 3)).astype(np.int32)  # over passes and the k slots
    return total


def extras(delta: np.ndarray) -> dict:
    """The appendix statistics the procedure asks for, over per-token deltas."""
    flat = delta.reshape(-1)
    order = np.sort(flat)[::-1]
    top10 = order[: max(1, len(order) // 10)].sum()
    increase = flat[flat > 0].sum()
    return {
        "delta_median": float(np.median(flat)),
        "delta_p90": float(np.quantile(flat, 0.90)),
        "delta_p99": float(np.quantile(flat, 0.99)),
        "top10pct_share_of_total_increase": float(top10 / increase) if increase > 0 else None,
        "fraction_tokens_improved": float((flat < 0).mean()),
    }


def mask_tag(name: str) -> str:
    return name.replace("(", "_").replace(")", "").replace("%", "pct").replace("/", "_")


def evaluate_mask(model, mask, ids_b, base_loss, base_picks, out: Path, device, doc_chunk,
                  save_per_token: bool) -> dict:
    """One masked forward over corpus B; the per-set row plus its per-token file.

    Shared with `extend.py`, which adds random controls to a finished product:
    a second copy of this loop would be a second place for the row to differ.
    """
    base_mean = float(base_loss.mean())
    with masked_routing(model, mask.experts):
        loss, _, _ = forward_pass(model, ids_b, device, doc_chunk, record=False)
    if not np.isfinite(loss).all():
        # Masking every expert of a layer zeroes that layer's routing and the
        # renormalisation divides by zero. A NaN written into the product would
        # poison the mean over controls silently (seen once on S11).
        raise SystemExit(
            f"REFUSING: non-finite loss under {mask.name} (per-layer masked counts "
            f"{mask.per_layer_counts}); a layer was probably left with no usable expert")
    if save_per_token:
        # `reassign_per_token` is aligned with `loss`: routing at input position
        # t is what produced the prediction scored at loss[:, t], so the last
        # input position (no target) is dropped.
        np.savez_compressed(
            out / f"per_token_{mask_tag(mask.name)}.npz",
            loss=loss, delta=loss - base_loss,
            reassign_per_token=reassignments_per_token(base_picks, mask.experts)[:, :-1],
        )
    return {**mask.as_dict(), "mean_loss": float(loss.mean()),
            "delta_L": float(loss.mean()) - base_mean, **extras(loss - base_loss)}


def top_k_of(model) -> int:
    """The loop routers' k, read from the model (every loop layer shares it)."""
    ks = {int(layer.router.num_active) for layer in loop_layers(model)}
    if len(ks) != 1:
        raise SystemExit(f"REFUSING: loop layers disagree on top-k: {sorted(ks)}")
    return ks.pop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--run-label", required=True, help="output directory name under --out-root")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--corpus-a", required=True, choices=sorted(MASKING_CORPORA))
    ap.add_argument("--corpus-b", required=True, choices=sorted(MASKING_CORPORA))
    ap.add_argument("--tiers", required=True, help="comma-separated percentages, e.g. 5,10,20")
    ap.add_argument("--n-random", type=int, required=True)
    ap.add_argument("--n-documents-a", type=int, required=True,
                    help="documents of corpus A to take for the share statistics")
    ap.add_argument("--n-documents-b", type=int, required=True,
                    help="documents of corpus B to evaluate on; need not equal A")
    ap.add_argument("--doc-chunk", type=int, required=True)
    ap.add_argument("--probe-dtype", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-per-token", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out_root) / args.run_label
    out.mkdir(parents=True, exist_ok=True)
    tiers = [float(t) for t in args.tiers.split(",")]

    _, tokenizer_repo = resolve_tokenizer(args.corpus_a, None)
    if MASKING_CORPORA[args.corpus_b][2] != MASKING_CORPORA[args.corpus_a][2]:
        raise SystemExit(
            f"REFUSING: corpus A ({args.corpus_a}) and corpus B ({args.corpus_b}) are "
            f"tokenized differently; one tokenizer is loaded for both and the model "
            f"has one vocabulary.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, trust_remote_code=True, **load_kwargs(args.probe_dtype)
    ).to(args.device)
    model.eval()

    ids_a, corpus_a_id, _ = _batch(tokenizer, args.corpus_a, args.n_documents_a)
    ids_b, corpus_b_id, _ = _batch(tokenizer, args.corpus_b, args.n_documents_b)
    same_corpus = args.corpus_a == args.corpus_b

    t0 = time.time()
    _, counts, _ = forward_pass(model, ids_a, args.device, args.doc_chunk, record=True)
    shares = shares_by_physical_expert(counts)
    print(f"[masking] corpus A done in {time.time()-t0:.0f}s; {len(shares)} physical experts", flush=True)

    base_loss, _, base_picks = forward_pass(model, ids_b, args.device, args.doc_chunk, record=True)
    base_mean = float(base_loss.mean())
    np.save(out / "per_token_unmasked.npy", base_loss)
    print(f"[masking] unmasked loss {base_mean:.6f}", flush=True)

    top_k = top_k_of(model)
    sizes = layer_sizes(shares)
    results = {}
    for x in tiers:
        target = target_set(shares, x)
        tier = {"target": target.as_dict(), "controls": [], "delta_T": None,
                "delta_R_mean": None, "delta_R_min": None, "delta_R_max": None, "rho": None,
                "per_layer_min_remaining": None, "skipped_seeds": [], "undefined": False}
        if not leaves_enough(target, sizes, top_k):
            tier.update(undefined=True, reason=(
                f"T({x:g}%) leaves a layer with only {per_layer_min_remaining(target, sizes)} "
                f"usable experts (< top_k={top_k}); this tier is undefined on this model"))
            print(f"[masking] x={x:g}% UNDEFINED: {tier['reason']}", flush=True)
            results[f"{x:g}"] = tier
            continue
        controls, skipped = random_controls(shares, target, args.n_random, top_k)
        tier["skipped_seeds"] = skipped
        sets = [("T", target)] + [("R", c) for c in controls]
        deltas_r = []
        for kind, mask in sets:
            row = evaluate_mask(model, mask, ids_b, base_loss, base_picks, out,
                                args.device, args.doc_chunk, args.save_per_token)
            row["per_layer_min_remaining"] = per_layer_min_remaining(mask, sizes)
            delta = row["delta_L"]
            if kind == "T":
                tier["delta_T"] = delta
                tier["target_stats"] = row
            else:
                deltas_r.append(delta)
                tier["controls"].append(row)
            print(f"[masking] x={x:g}% {mask.name}: n={len(mask.experts)} "
                  f"share={mask.actual_share*100:.3f}% dL={delta:+.6f}", flush=True)
        tier["delta_R_mean"] = float(np.mean(deltas_r))
        tier["delta_R_min"] = float(np.min(deltas_r))
        tier["delta_R_max"] = float(np.max(deltas_r))
        tier["delta_R_each"] = deltas_r
        tier["rho"] = (tier["delta_T"] / tier["delta_R_mean"]) if tier["delta_R_mean"] != 0 else None
        tier["per_layer_min_remaining"] = min(
            [tier["target_stats"]["per_layer_min_remaining"]]
            + [c["per_layer_min_remaining"] for c in tier["controls"]])
        results[f"{x:g}"] = tier
        print(f"[masking] x={x:g}% rho={tier['rho']}", flush=True)

    summary = {
        "procedure": "expert masking v1 (see README, Expert masking)",
        "checkpoint_dir": args.checkpoint_dir,
        "corpus_a": corpus_a_id, "corpus_b": corpus_b_id,
        "n_documents_a": args.n_documents_a, "n_documents_b": args.n_documents_b,
        "n_tokens_b": int(ids_b.shape[0] * (ids_b.shape[1] - 1)),
        "corpus_b_is_corpus_a": same_corpus,
        "corpus_caveat": (
            "Corpus B is the same as corpus A: the sets were chosen and the losses measured on the same tokens, a pilot defect that the final procedure corrects. "
            "This product only exercises the code path and is not a valid reading."
            if same_corpus else None),
        "unmasked_mean_loss": base_mean,
        # Sits next to the number it qualifies, not in a report someone may not
        # have. Corpus A and corpus B are different documents (500 vs 485), so
        # this absolute loss is not comparable across them; only the deltas are,
        # because within one corpus N cancels. When a run measured on A is
        # redone on B this number moves, and without this line the move reads
        # as a regression in the model.
        "unmasked_mean_loss_note": (
            "The absolute value is meaningful only within one corpus: corpora A and B are different document sets (500 / 485 documents), "
            "so the same model's unmasked loss differs between them. ΔL and ρ are unaffected (differences within one corpus, N cancels). "
            "Comparing this number on A with the one on B, or reading a rerun on B as a model regression, is wrong."
        ),
        "rho_interpretation": RHO_INTERPRETATION_NOTE,
        "top_k": top_k,
        "feasibility_rule": FEASIBILITY_RULE,
        "tiers": results,
        # Random controls are seeded 0..n_random-1 from the SAME target set, so
        # raising the count later (e.g. to 20) adds seeds n_random..19 and
        # leaves T and the first five controls exactly as they are here.
        "n_random": args.n_random,
        "shares": {f"{l},{e}": v for (l, e), v in sorted(shares.items())},
        "code_revision": code_revision(),
        "machine": machine_provenance(),
        "host": socket.gethostname(),
        "command": " ".join(["python", "-m", "src.eval.expert_masking.run"] + __import__("sys").argv[1:]),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[masking] wrote {out/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

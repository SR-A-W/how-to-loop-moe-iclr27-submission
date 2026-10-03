"""Add random controls to a finished expert-masking product without redoing T.

Raises the random-control count (e.g. 5 -> 20). The target sets, the five
existing controls and the unmasked baseline are already measured; what is
missing is seeds `n_old .. n_new-1`. Controls are drawn from the SAME shares
and target with `random_control(shares, target, seed)`, and the seed alone
fixes a control, so seed 7 drawn today is what seed 7 would have been in the
original run. This module:

1. reads the product's `summary.json`, restores shares and each tier's target;
2. regenerates the existing seeds and refuses unless they match the recorded
   controls member for member (proves the restore is exact before spending GPU);
3. re-runs the unmasked forward and refuses unless its mean loss matches the
   recorded one (same weights, same corpus, same code path);
4. evaluates only the new seeds, merges them in, recomputes mean/min/max/rho
   over ALL controls, and writes `summary.json` after copying the old one to
   `summary.n_random_<n_old>.json`. Existing rows and per-token files are not
   touched.

The pure parts (restore, plan, merge) have no torch dependency and are tested
on hand-made summaries.
"""
from __future__ import annotations

import argparse, json, shutil, socket, sys, time
from pathlib import Path

import numpy as np

from src.eval.expert_masking.selection import (
    Expert, MaskSet, layer_sizes, leaves_enough, per_layer_min_remaining, random_control,
    random_controls,
)

#: Recorded and recomputed unmasked mean loss may differ by chunk-order
#: rounding (observed 3e-7 between two runs of the same product); anything
#: larger means a different model, corpus or code path.
UNMASKED_LOSS_TOLERANCE = 1e-5


def restore_shares(summary: dict) -> dict[Expert, float]:
    shares = {}
    for key, value in summary["shares"].items():
        layer, expert = key.split(",")
        shares[(int(layer), int(expert))] = float(value)
    if not shares:
        raise ValueError("summary has no shares")
    return shares


def restore_target(tier: dict) -> MaskSet:
    t = tier["target"]
    experts = tuple(sorted((int(l), int(e)) for l, e in t["experts"]))
    return MaskSet(t["name"], experts, float(t["actual_share"]),
                   {int(k): v for k, v in t["per_layer_counts"].items()})


def control_name(seed: int, target: MaskSet) -> str:
    """The name the original run gives control `seed` of `target`."""
    return f"R{seed}({target.name})"


def check_existing_controls(summary: dict) -> None:
    """Regenerate seeds 0..n_random-1 and require them to equal the recorded rows.

    If this fails the product was made with different shares, a different
    target, or different selection code, and adding seeds to it would mix two
    definitions of "random control" under one rho.
    """
    shares = restore_shares(summary)
    n_old = int(summary["n_random"])
    for key, tier in summary["tiers"].items():
        target = restore_target(tier)
        controls = tier["controls"]
        if len(controls) != n_old:
            raise ValueError(f"tier {key}: {len(controls)} controls recorded, n_random={n_old}")
        for seed, row in enumerate(controls):
            fresh = random_control(shares, target, seed, name=control_name(seed, target))
            recorded = tuple(sorted((int(l), int(e)) for l, e in row["experts"]))
            if fresh.experts != recorded or fresh.name != row["name"]:
                raise ValueError(
                    f"tier {key} seed {seed}: regenerated control {fresh.name} {fresh.experts} "
                    f"!= recorded {row['name']} {recorded}")


def plan_extension(summary: dict, n_to: int, top_k: int) -> list[dict]:
    """Per tier: which recorded controls to keep, which to drop, which seeds to run.

    The wanted list is the first `n_to` FEASIBLE seeds (`random_controls`), so a
    recorded control whose seed is infeasible under the per-layer rule is
    dropped, and seeds missing from the record are run. A target that is
    itself infeasible makes the tier undefined. Refuses when there is nothing
    to do (not an extension and nothing to replace).
    """
    n_old = int(summary["n_random"])
    if n_to < n_old:
        raise ValueError(f"--extend-random-to {n_to} is less than the recorded n_random {n_old}; "
                         f"controls are never removed to shrink a product")
    shares = restore_shares(summary)
    sizes = layer_sizes(shares)
    plan, work = [], 0
    for key, tier in summary["tiers"].items():
        target = restore_target(tier)
        entry = {"tier": key, "target": target, "undefined": False, "keep": [], "drop": [],
                 "run": [], "skipped_seeds": [], "reason": None}
        if tier.get("undefined") or not leaves_enough(target, sizes, top_k):
            entry.update(undefined=True, reason=(
                f"{target.name} leaves a layer with only {per_layer_min_remaining(target, sizes)} "
                f"usable experts (< top_k={top_k}); this tier is undefined on this model"))
            entry["drop"] = [r["name"] for r in tier.get("controls", [])]
            work += len(entry["drop"])
            plan.append(entry)
            continue
        wanted, skipped = random_controls(shares, target, n_to, top_k)
        wanted_names = [c.name for c in wanted]
        recorded = {r["name"]: r for r in tier.get("controls", [])}
        entry["keep"] = [n for n in wanted_names if n in recorded]
        entry["drop"] = [n for n in recorded if n not in wanted_names]
        entry["run"] = [c for c in wanted if c.name not in recorded]
        entry["skipped_seeds"] = skipped
        work += len(entry["drop"]) + len(entry["run"])
        plan.append(entry)
    if work == 0:
        raise ValueError(
            f"nothing to do: every tier already holds the first {n_to} feasible controls "
            f"(recorded n_random {summary['n_random']})")
    return plan


def _seed_of(name: str) -> int:
    return int(name[1:name.index("(")])


def merge_controls(tier: dict, new_rows: list[dict], keep: list[str] | None = None,
                   sizes: dict[int, int] | None = None, skipped=None, dropped=None) -> dict:
    """The tier with kept rows + `new_rows`, ordered by seed, aggregates recomputed.

    Kept rows are the recorded objects unchanged; the aggregates are recomputed
    from the full list, not updated incrementally, so the result is what a
    single run with the larger count would have written. With `keep=None`
    every recorded row is kept (plain extension).
    """
    merged = dict(tier)
    old_rows = list(tier["controls"])
    if keep is not None:
        old_rows = [r for r in old_rows if r["name"] in set(keep)]
    rows = sorted(old_rows + list(new_rows), key=lambda r: _seed_of(r["name"]))
    if sizes is not None:
        for r in rows:
            if "per_layer_min_remaining" not in r:
                per_layer = {int(k): v for k, v in r["per_layer_counts"].items()}
                r["per_layer_min_remaining"] = min(sizes[l] - per_layer.get(l, 0) for l in sizes)
    merged["controls"] = rows
    if skipped is not None:
        merged["skipped_seeds"] = skipped
    if dropped:
        merged["dropped_controls"] = list(merged.get("dropped_controls", [])) + list(dropped)
    if sizes is not None and "per_layer_min_remaining" in tier.get("target_stats", {}):
        merged["per_layer_min_remaining"] = min([tier["target_stats"]["per_layer_min_remaining"]]
                                                + [r["per_layer_min_remaining"] for r in rows])
    elif sizes is not None:
        merged["per_layer_min_remaining"] = min(r["per_layer_min_remaining"] for r in rows)
    deltas = [float(r["delta_L"]) for r in merged["controls"]]
    merged["delta_R_each"] = deltas
    merged["delta_R_mean"] = float(np.mean(deltas))
    merged["delta_R_min"] = float(np.min(deltas))
    merged["delta_R_max"] = float(np.max(deltas))
    merged["rho"] = (merged["delta_T"] / merged["delta_R_mean"]) if merged["delta_R_mean"] != 0 else None
    return merged


def extended_summary(summary: dict, new_rows_by_tier: dict[str, list[dict]], n_to: int,
                     provenance: dict, plan: list[dict] | None = None) -> dict:
    out = dict(summary)
    sizes = layer_sizes(restore_shares(summary)) if plan is not None else None
    by_tier = {e["tier"]: e for e in (plan or [])}
    tiers = {}
    for key, tier in summary["tiers"].items():
        e = by_tier.get(key)
        if e is None:
            tiers[key] = merge_controls(tier, new_rows_by_tier.get(key, []))
        elif e["undefined"]:
            tiers[key] = {**tier, "controls": [], "delta_R_mean": None, "delta_R_min": None,
                          "delta_R_max": None, "delta_R_each": [], "rho": None,
                          "undefined": True, "reason": e["reason"],
                          "dropped_controls": [{"name": n, "reason": e["reason"]} for n in e["drop"]]}
        else:
            dropped = [{"name": n, "reason": "seed infeasible under the per-layer rule"} for n in e["drop"]]
            tiers[key] = merge_controls(tier, new_rows_by_tier.get(key, []), keep=e["keep"],
                                        sizes=sizes, skipped=e["skipped_seeds"], dropped=dropped)
    out["tiers"] = tiers
    for key, tier in out["tiers"].items():
        if not tier.get("undefined") and len(tier["controls"]) != n_to:
            raise ValueError(f"tier {key}: {len(tier['controls'])} controls after merge, expected {n_to}")
    out["n_random"] = n_to
    history = list(summary.get("n_random_history", [int(summary["n_random"])]))
    out["n_random_history"] = history + [n_to]
    out["extensions"] = list(summary.get("extensions", [])) + [provenance]
    return out


def write_extended(summary_path: Path, merged: dict, n_old: int) -> Path:
    """Copy the current summary aside, then replace it. Never overwrites a backup."""
    backup = summary_path.with_name(f"summary.n_random_{n_old}.json")
    i = 2
    while backup.exists():  # never overwrite: a rework of a rework gets its own copy
        backup = summary_path.with_name(f"summary.n_random_{n_old}.v{i}.json")
        i += 1
    shutil.copy2(summary_path, backup)
    summary_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    return backup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product-dir", required=True, help="directory holding summary.json")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--extend-random-to", type=int, required=True)
    ap.add_argument("--doc-chunk", type=int, required=True)
    ap.add_argument("--probe-dtype", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Heavy imports after argument parsing so the pure functions stay cheap.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.eval.expert_masking.run import (
        FEASIBILITY_RULE, MASKING_CORPORA, _batch, evaluate_mask, forward_pass, top_k_of,
    )
    from src.eval.probes.offline_cross_arch_probe import resolve_tokenizer
    from src.eval.probes.probe_dtype import load_kwargs
    from src.metrics.provenance import code_revision, machine_provenance

    out = Path(args.product_dir)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["checkpoint_dir"] != args.checkpoint_dir:
        # Not fatal by itself (staging paths move), but it is recorded.
        print(f"[extend] note: recorded checkpoint_dir {summary['checkpoint_dir']} "
              f"!= {args.checkpoint_dir}; weights are checked through the unmasked loss", flush=True)
    check_existing_controls(summary)
    n_old = int(summary["n_random"])
    shares = restore_shares(summary)

    corpus_keys = {cid: key for key, (_, cid, _) in MASKING_CORPORA.items()}
    corpus_a, corpus_b = corpus_keys[summary["corpus_a"]], corpus_keys[summary["corpus_b"]]
    _, tokenizer_repo = resolve_tokenizer(corpus_a, None)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, trust_remote_code=True, **load_kwargs(args.probe_dtype)
    ).to(args.device)
    model.eval()
    top_k = top_k_of(model)
    if "top_k" in summary and int(summary["top_k"]) != top_k:
        raise SystemExit(f"REFUSING: product top_k {summary['top_k']} != model's {top_k}")
    plan = plan_extension(summary, args.extend_random_to, top_k)
    for e in plan:
        print(f"[extend] x={e['tier']}%: keep {len(e['keep'])}, drop {e['drop']}, run "
              f"{[c.name for c in e['run']]}, skipped seeds {[d['seed'] for d in e['skipped_seeds']]}"
              + (f"; UNDEFINED: {e['reason']}" if e['undefined'] else ""), flush=True)
    sizes = layer_sizes(shares)
    ids_b, corpus_b_id, _ = _batch(tokenizer, corpus_b, int(summary["n_documents_b"]))
    if corpus_b_id != summary["corpus_b"]:
        raise SystemExit(f"REFUSING: corpus B resolved to {corpus_b_id}, product says {summary['corpus_b']}")

    base_loss, _, base_picks = forward_pass(model, ids_b, args.device, args.doc_chunk, record=True)
    recorded = np.load(out / "per_token_unmasked.npy")
    if base_loss.shape != recorded.shape:
        raise SystemExit(f"REFUSING: unmasked per-token shape {base_loss.shape} != recorded {recorded.shape}")
    gap = abs(float(base_loss.mean()) - float(summary["unmasked_mean_loss"]))
    if gap > UNMASKED_LOSS_TOLERANCE:
        raise SystemExit(
            f"REFUSING: unmasked mean loss {float(base_loss.mean()):.6f} differs from recorded "
            f"{summary['unmasked_mean_loss']:.6f} by {gap:.2e} (> {UNMASKED_LOSS_TOLERANCE}); "
            f"not the same weights/corpus/code path")
    # Deltas are taken against the RECORDED baseline so every control, old and
    # new, is measured against one and the same unmasked loss.
    base_loss = recorded

    t0 = time.time()
    new_rows: dict[str, list[dict]] = {}
    for e in plan:
        rows = []
        for mask in e["run"]:
            row = evaluate_mask(model, mask, ids_b, base_loss, base_picks, out,
                                args.device, args.doc_chunk, True)
            row["per_layer_min_remaining"] = per_layer_min_remaining(mask, sizes)
            rows.append(row)
            print(f"[extend] x={e['tier']}% {mask.name}: n={len(mask.experts)} "
                  f"share={mask.actual_share*100:.3f}% dL={row['delta_L']:+.6f}", flush=True)
        new_rows[e["tier"]] = rows

    provenance = {
        "from_n_random": n_old, "to_n_random": args.extend_random_to,
        "seeds_added": {e["tier"]: [c.name for c in e["run"]] for e in plan},
        "controls_dropped": {e["tier"]: e["drop"] for e in plan},
        "feasibility_rule": FEASIBILITY_RULE, "top_k": top_k,
        "unmasked_recheck_gap": gap, "elapsed_s": int(time.time() - t0),
        "code_revision": code_revision(), "machine": machine_provenance(),
        "host": socket.gethostname(),
        "command": " ".join(["python", "-m", "src.eval.expert_masking.extend"] + sys.argv[1:]),
    }
    merged = extended_summary(summary, new_rows, args.extend_random_to, provenance, plan)
    merged["top_k"] = top_k
    merged["feasibility_rule"] = FEASIBILITY_RULE
    backup = write_extended(summary_path, merged, n_old)
    for key, tier in merged["tiers"].items():
        print(f"[extend] x={key}% rho={tier['rho']} over {len(tier['controls'])} controls"
              + (" (UNDEFINED)" if tier.get("undefined") else ""), flush=True)
    print(f"[extend] wrote {summary_path}; previous copy at {backup}", flush=True)


if __name__ == "__main__":
    main()

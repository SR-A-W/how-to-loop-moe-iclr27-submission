"""Backfill per-token reassignment counts for a finished expert-masking product.

Older products carry a per-sequence `reassignments` array in
`per_token_<set>.npz`, not the per-token count the procedure asks for. The
per-token count (`reassignments_per_token`) is a property of the UNMASKED
routing and the masked set, so one recorded forward over corpus B gives it for
every set of the product; no masked forward is repeated.

Writes `per_token_reassign_<set>.npz` (new files; nothing existing is touched)
for the target set of every tier and, at no extra cost, for every recorded
random control, plus `reassign_backfill.json` with provenance. The unmasked
mean loss is re-checked against the record first (same weights, corpus, code).
"""
from __future__ import annotations

import argparse, json, socket, sys, time
from pathlib import Path

import numpy as np

from src.eval.expert_masking.extend import UNMASKED_LOSS_TOLERANCE, restore_target
from src.eval.expert_masking.run import (
    MASKING_CORPORA, _batch, forward_pass, mask_tag, reassignments_per_token,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--doc-chunk", type=int, required=True)
    ap.add_argument("--probe-dtype", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sets", choices=["targets", "all"], required=True,
                    help="targets: the T of each tier; all: T and every recorded control")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.eval.probes.offline_cross_arch_probe import resolve_tokenizer
    from src.eval.probes.probe_dtype import load_kwargs
    from src.metrics.provenance import code_revision, machine_provenance

    out = Path(args.product_dir)
    summary = json.loads((out / "summary.json").read_text())
    corpus_keys = {cid: key for key, (_, cid, _) in MASKING_CORPORA.items()}
    corpus_a, corpus_b = corpus_keys[summary["corpus_a"]], corpus_keys[summary["corpus_b"]]
    _, tokenizer_repo = resolve_tokenizer(corpus_a, None)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, trust_remote_code=True, **load_kwargs(args.probe_dtype)
    ).to(args.device)
    model.eval()
    ids_b, corpus_b_id, _ = _batch(tokenizer, corpus_b, int(summary["n_documents_b"]))
    if corpus_b_id != summary["corpus_b"]:
        raise SystemExit(f"REFUSING: corpus B resolved to {corpus_b_id}, product says {summary['corpus_b']}")

    t0 = time.time()
    base_loss, _, base_picks = forward_pass(model, ids_b, args.device, args.doc_chunk, record=True)
    gap = abs(float(base_loss.mean()) - float(summary["unmasked_mean_loss"]))
    if gap > UNMASKED_LOSS_TOLERANCE:
        raise SystemExit(f"REFUSING: unmasked mean loss differs from the record by {gap:.2e}")

    written = []
    for key, tier in summary["tiers"].items():
        if tier.get("undefined"):
            continue
        sets = [restore_target(tier)]
        if args.sets == "all":
            from src.eval.expert_masking.selection import MaskSet
            for c in tier["controls"]:
                sets.append(MaskSet(c["name"], tuple((int(l), int(e)) for l, e in c["experts"]),
                                    float(c["actual_share"]), {}))
        for mask in sets:
            counts = reassignments_per_token(base_picks, mask.experts)[:, :-1]
            if counts.shape != base_loss.shape:
                raise SystemExit(f"shape {counts.shape} != loss {base_loss.shape}")
            path = out / f"per_token_reassign_{mask_tag(mask.name)}.npz"
            np.savez_compressed(path, reassign_per_token=counts.astype(np.int32))
            written.append({"set": mask.name, "file": path.name, "n_experts": len(mask.experts),
                            "tokens_reassigned_fraction": float((counts > 0).mean()),
                            "mean_reassignments": float(counts.mean())})
            print(f"[backfill] {key}% {mask.name}: reassigned {float((counts > 0).mean()):.4f} of tokens, "
                  f"mean {float(counts.mean()):.3f}", flush=True)
    prov = {
        "what": "per-token reassignment counts, backfilled from one recorded unmasked forward",
        "definition": "for each token, over every loop position (layer x pass), the number of the UNMASKED "
                      "top-k experts that belong to the masked set; aligned with per-token loss "
                      "(input position t -> loss[:, t]; last input position dropped)",
        "sets": args.sets, "files": written, "unmasked_recheck_gap": gap,
        "elapsed_s": int(time.time() - t0), "code_revision": code_revision(),
        "machine": machine_provenance(), "host": socket.gethostname(),
        "backfilled_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(["python", "-m", "src.eval.expert_masking.backfill_reassign"] + sys.argv[1:]),
    }
    (out / "reassign_backfill.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2))
    print(f"[backfill] wrote {len(written)} files + reassign_backfill.json in {int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()

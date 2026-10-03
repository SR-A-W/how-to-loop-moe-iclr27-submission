"""Build a real-text probe corpus from NeelNanda/pile-10k.

Adapted from an earlier diagnostics code base (original path
`scripts/prepare_corpus.py`), Apache License 2.0. Selection logic unchanged.

Why this is a separate step that must run first
-----------------------------------------------
Random token ids let a correct and an incorrect implementation both pass: with
6 random token ids, the `cumsum` and `survival` exit modes of Ouro both scored
6/6, the wrong one included; 81 tokens of real text separated them (survival
81/81, cumsum 67/81). The corpus is part of the evidence, not an implementation
detail. So this script:
  1. takes real natural text (not random ids, not templates);
  2. takes several domains (attention-sink and rank metrics are sensitive to
     text type; a single domain would make a domain property look like an
     architecture property);
  3. writes the provenance into the output (dataset id, split, original index,
     per-document sha256, character count), so the token source can be
     reproduced by others.

Why filter by characters rather than tokens
-------------------------------------------
Different models use different tokenizers, so the same text yields different
token counts. This script applies a conservative character floor; the exact
token-count check is left to each consumer, which raises if a document is
shorter than the sequence length. There is no padding: pad tokens would produce
residual/router decisions at meaningless positions and contaminate per-token
metrics.

Network
-------
Only this step needs network access (Hugging Face Hub). Run it on a machine with
internet access; later GPU jobs only read the output file. `datasets` is needed
only for this step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET_ID = "NeelNanda/pile-10k"
DATASET_SPLIT = "train"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output JSON path (e.g. src/data/artifacts/corpus/*.json)")
    ap.add_argument("--n-docs", type=int, required=True,
                    help="number of distinct documents to select (the probe batch size B)")
    ap.add_argument("--min-chars", type=int, required=True,
                    help="minimum characters per document. No default: it depends on the sequence "
                         "length S and the tokenizer, and a wrong guess only fails halfway through a "
                         "probe run. For S=512, 4096 works (a conservative ~8 characters per token).")
    ap.add_argument("--max-per-domain", type=int, default=1,
                    help="maximum documents per domain (default 1, maximising domain diversity)")
    ap.add_argument("--domains", default=None,
                    help="comma-separated domain whitelist (e.g. 'Github,Pile-CC'); omit for no restriction."
                         "\nWith a whitelist, every listed domain must reach --max-per-domain documents, "
                         "otherwise the script raises. Without that check, '--n-docs 32 --max-per-domain 4' "
                         "can return 11 domains with 1-4 documents each: the per-domain cap holds, but "
                         "domain effects can no longer be separated from single-document effects, and the "
                         "output does not show it.")
    args = ap.parse_args()

    whitelist: list[str] | None = None
    if args.domains is not None:
        whitelist = [d.strip() for d in args.domains.split(",") if d.strip()]
        if not whitelist:
            raise SystemExit("--domains parsed to an empty list; omit the option for no restriction.")
        if len(set(whitelist)) != len(whitelist):
            raise SystemExit(f"--domains contains duplicates: {whitelist}")

    try:
        from datasets import load_dataset
    except ImportError as e:                      # noqa: F841
        raise SystemExit(
            "this script needs `datasets`: pip install datasets"
        )

    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)

    # Deterministic selection: scan in original index order and skip a domain once it
    # has max_per_domain documents. No random sampling: the selection must be exactly
    # reproducible by others.
    picked: list[dict] = []
    per_domain: dict[str, int] = {}
    seen_sha: set[str] = set()
    for i, row in enumerate(ds):
        text = row["text"]
        if len(text) < args.min_chars:
            continue
        domain = (row.get("meta") or {}).get("pile_set_name", "unknown")
        if whitelist is not None and domain not in whitelist:
            continue
        if per_domain.get(domain, 0) >= args.max_per_domain:
            continue
        sha = hashlib.sha256(text.encode()).hexdigest()
        if sha in seen_sha:                       # same document seen earlier in the dataset
            continue
        seen_sha.add(sha)
        per_domain[domain] = per_domain.get(domain, 0) + 1
        picked.append({
            "name": f"{domain.replace(' ', '_')}#{i}",
            "domain": domain,
            "text": text,
            "source_index": i,
            "sha256": sha,
            "n_chars": len(text),
        })
        if len(picked) >= args.n_docs:
            break

    if len(picked) < args.n_docs:
        raise SystemExit(
            f"found only {len(picked)} documents with min_chars={args.min_chars} under the "
            f"per-domain cap; {args.n_docs} required. Raise --max-per-domain or lower --min-chars."
            f"\nDo not pad with repeated documents: the sample size {args.n_docs} would then be "
            f"misleading."
        )

    # Balance check: with a whitelist, every domain must be full, otherwise raise.
    # Without this check the output can satisfy "at most N per domain" while some
    # domains have a single document, and nothing in the output shows it.
    if whitelist is not None:
        short = {d: per_domain.get(d, 0) for d in whitelist
                 if per_domain.get(d, 0) < args.max_per_domain}
        if short:
            table = "\n".join(f"    {d:24s} {per_domain.get(d, 0)} / {args.max_per_domain}"
                              for d in whitelist)
            raise SystemExit(
                f"{len(short)} whitelisted domains have fewer than {args.max_per_domain} documents: {short}\n"
                f"  documents per domain:\n{table}\n"
                f"  The corpus would be unbalanced: in these domains, domain effects cannot be "
                f"separated from single-document effects.\n"
                f"  Options: lower --min-chars (changes the token mix), lower --max-per-domain, "
                f"or remove these domains from the whitelist."
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Provenance is stored with the corpus, so readers can see where every document came from.
    payload = {
        "provenance": {
            "dataset": DATASET_ID,
            "split": DATASET_SPLIT,
            "n_rows_scanned": len(ds),
            "selection": "deterministic: original index order, "
                         + (f"restricted to the {len(whitelist)}-domain whitelist "
                            f"{sorted(whitelist)}, " if whitelist is not None else "")
                         + f"max {args.max_per_domain} doc(s) per pile_set_name, "
                         f"min_chars={args.min_chars}, "
                         "exact-duplicate texts skipped by sha256; "
                         "no random sampling, no shuffling, no seed",
            "n_docs": len(picked),
            "domains": sorted(per_domain),
            # Documents per domain: balance is what lets domain effects be separated
            # from single-document effects, so it is recorded directly in the output.
            "docs_per_domain": {d: per_domain[d] for d in sorted(per_domain)},
            "domain_whitelist": sorted(whitelist) if whitelist is not None else None,
            "balanced": (whitelist is not None
                         and all(per_domain.get(d, 0) == args.max_per_domain for d in whitelist)),
        },
        "docs": picked,
    }
    out.write_text(json.dumps(payload, indent=1))

    print(f"dataset  : {DATASET_ID}/{DATASET_SPLIT}")
    print(f"selected : {len(picked)} docs, {len(per_domain)} distinct domains")
    print("per-domain counts:")
    for d in sorted(per_domain):
        print(f"  {d:24s} {per_domain[d]}")
    for d in picked:
        print(f"  {d['name']:<28} {d['n_chars']:>8} chars  sha={d['sha256'][:8]}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

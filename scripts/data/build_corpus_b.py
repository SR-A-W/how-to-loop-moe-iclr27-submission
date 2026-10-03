"""Build "corpus B" (the expert-masking eval's second, disjoint probe corpus).

This code first ran as a one-off script printed in
`src/data/artifacts/corpus/README.md` ("Generation method"). The selection loop
(scan order, min_chars, domain whitelist, max_per_domain, sha256 dedup, exclusion
of corpus A's documents) applies the same conditions as that script; the only
difference is that the sha256 is computed one step earlier, to count available
documents per domain (`domain_shortfall` in the provenance). Re-running it with
the usage below reproduces the committed `pile10k_B_d50_b500_s512.json` byte for
byte (same algorithm, same dataset and split, deterministic scan).

Needs network access to `NeelNanda/pile-10k` (or a local HF_HUB_OFFLINE cache).
Pass `--corpus-a` as the relative path shown below: that string is written into
the output file.

Usage:
    python scripts/data/build_corpus_b.py \
        --corpus-a src/data/artifacts/corpus/pile10k_d50_b500_s512.json \
        --out src/data/artifacts/corpus/pile10k_B_d50_b500_s512.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# The same 10-domain whitelist as prepare_corpus.py. Sorted: selection only tests
# membership, so order does not change which documents are picked; sorting keeps the
# domain list in the output provenance identical to the committed file.
DOMAINS = sorted(["ArXiv", "Wikipedia (en)", "DM Mathematics", "FreeLaw", "Github",
                  "OpenWebText2", "Pile-CC", "PubMed Central", "StackExchange", "USPTO Backgrounds"])
MIN_CHARS, MAX_PER_DOMAIN = 4096, 50
DATASET_ID, DATASET_SPLIT = "NeelNanda/pile-10k", "train"


def select_corpus_b(corpus_a_path: Path) -> tuple[list[dict], dict]:
    """The selection loop, unchanged from the README's one-off script.

    Returns `(picked_docs, stats)` where `stats` has enough to build the same
    `provenance` shape the existing `pile10k_B_d50_b500_s512.json` already carries
    (domains_per_domain / domain_shortfall), computed from the run's own counts
    rather than hardcoded, so a re-run against a changed corpus A stays honest.
    """
    from datasets import load_dataset

    a = json.loads(corpus_a_path.read_text())
    excluded_sha = {d["sha256"] for d in a["docs"]}  # corpus A's documents

    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    picked: list[dict] = []
    per_domain: dict[str, int] = {}
    seen_sha: set[str] = set()
    domain_available_total: dict[str, int] = {d: 0 for d in DOMAINS}
    domain_taken_by_a: dict[str, int] = {d: 0 for d in DOMAINS}
    for d in a["docs"]:
        if d["domain"] in domain_taken_by_a:
            domain_taken_by_a[d["domain"]] += 1

    for i, row in enumerate(ds):
        text = row["text"]
        if len(text) < MIN_CHARS:
            continue
        domain = (row.get("meta") or {}).get("pile_set_name", "unknown")
        if domain not in DOMAINS:
            continue
        sha = hashlib.sha256(text.encode()).hexdigest()
        if sha not in excluded_sha:
            # count toward "how many qualifying docs exist at all", regardless
            # of whether corpus A or corpus B ends up taking this one
            domain_available_total[domain] = domain_available_total.get(domain, 0) + 1
        if domain not in DOMAINS or per_domain.get(domain, 0) >= MAX_PER_DOMAIN:
            continue
        if sha in seen_sha or sha in excluded_sha:  # <-- the one addition over prepare_corpus.py
            continue
        seen_sha.add(sha)
        per_domain[domain] = per_domain.get(domain, 0) + 1
        picked.append({"name": f"{domain.replace(' ', '_')}#{i}", "domain": domain, "text": text,
                        "source_index": i, "sha256": sha, "n_chars": len(text)})

    domain_shortfall = {
        domain: {
            "available_total_in_dataset": domain_available_total.get(domain, 0) + domain_taken_by_a.get(domain, 0),
            "taken_by_corpus_a": domain_taken_by_a.get(domain, 0),
            "expected_for_corpus_b": min(MAX_PER_DOMAIN, domain_available_total.get(domain, 0)),
            "got_for_corpus_b": per_domain.get(domain, 0),
        }
        for domain in DOMAINS
        if per_domain.get(domain, 0) < MAX_PER_DOMAIN
    }
    stats = {
        "docs_per_domain": {d: per_domain.get(d, 0) for d in DOMAINS},
        "domain_shortfall": domain_shortfall,
        "excluded_sha256_count": len(excluded_sha),
    }
    return picked, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-a", required=True, type=Path,
                     help="path to corpus A's json (pile10k_d50_b500_s512.json), read for its sha256 set")
    ap.add_argument("--out", required=True, type=Path, help="output json path")
    args = ap.parse_args()

    picked, stats = select_corpus_b(args.corpus_a)
    provenance = {
        "dataset": DATASET_ID,
        "split": DATASET_SPLIT,
        "n_rows_scanned": 10000,
        "selection": (
            "deterministic: original index order, restricted to the 10-domain "
            f"whitelist {DOMAINS}, max {MAX_PER_DOMAIN} doc(s) per pile_set_name, "
            f"min_chars={MIN_CHARS}, exact-duplicate texts skipped by sha256, "
            "documents already used by corpus B's sibling A "
            f"({args.corpus_a.name}) excluded by sha256; no random sampling, "
            "no shuffling, no seed. Same algorithm as "
            "scripts/data/prepare_corpus.py plus the A-exclusion step."
        ),
        "n_docs": len(picked),
        "domains": DOMAINS,
        "docs_per_domain": stats["docs_per_domain"],
        "domain_whitelist": DOMAINS,
        "balanced": False,
        "excluded_corpus": f"{args.corpus_a} (corpus A)",
        "excluded_sha256_count": stats["excluded_sha256_count"],
        "domain_shortfall": stats["domain_shortfall"],
    }
    args.out.write_text(json.dumps({"provenance": provenance, "docs": picked}, indent=1))
    print(f"wrote {len(picked)} docs to {args.out}")


if __name__ == "__main__":
    main()

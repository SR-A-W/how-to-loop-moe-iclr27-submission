#!/usr/bin/env python3
"""Make every <ORG>/<prefix>* model repo on the Hugging Face Hub public.

Reads the token from HF_TOKEN or the local huggingface token file; never prints it.
Idempotent: repos already public are skipped. Pass --only PREFIX to restrict
(e.g. --only <prefix>), or --dry-run to list without changing. The default prefix
matches no repository, so pass --only.
"""
import argparse, os, sys
from pathlib import Path
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--org", required=True, help="Hugging Face user or organisation that owns the repos")
ap.add_argument("--only", default="<prefix>", help="repo-name prefix after '<ORG>/' (the default matches nothing)")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
tok = os.environ.get("HF_TOKEN") or Path.home().joinpath(".cache/huggingface/token").read_text().strip()
api = HfApi(token=tok)
repos = sorted(m.id for m in api.list_models(author=a.org) if m.id.startswith(a.org + "/" + a.only))
flipped, kept = [], []
for r in repos:
    if not api.model_info(r).private:
        kept.append(r); continue
    if not a.dry_run:
        api.update_repo_settings(repo_id=r, private=False)
        assert not api.model_info(r).private, r
    flipped.append(r)
print("%s %d repo(s):" % ("would flip" if a.dry_run else "flipped to public", len(flipped)))
for r in flipped: print("  " + r.split("/")[1])
print("already public: %d" % len(kept))

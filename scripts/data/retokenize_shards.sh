#!/bin/bash
# Re-tokenize an existing raw-parquet shard set (the 100BT pretrain corpus, OR the FineWeb-Edu
# validation split run_eval.py's `compute_fineweb_edu_val_loss` reads -- both are just
# `raw_manifest.json` + parquet directories as far as `tokenize_to_bin_idx.py` is concerned, no
# separate code path needed) with a DIFFERENT tokenizer than whatever originally produced the
# existing bin/idx directory: the same documents, re-cut with a new tokenizer into a new
# corpus directory, leaving the old one untouched.
#
# ALWAYS writes to a NEW --out-dir -- never overwrites/rewrites the directory an existing
# tokenizer's output lives in (enforced below: refuses to run if OUT_DIR already exists,
# published or staging).
#
# Required (no defaults -- which raw shards, which tokenizer, and how many parallel workers to
# use are all decisions that change behavior/resource usage and must be explicit):
#   RAW_DIR         directory with *.parquet + raw_manifest.json (download_shards.py's output)
#   OUT_DIR         NEW output directory (must not already exist)
#   TOKENIZER_NAME  llama3 | smollm2 (see src/data/tokenizer.py's TOKENIZER_CLASSES)
#   NUM_WORKERS     how many parallel tokenize processes (see tokenize_to_bin_idx.py --num-workers)
#
# Optional:
#   TOKENIZER_REPO       override TOKENIZER_NAME's own bundled directory (another local dir, or
#                        an HF Hub repo id) -- omit to use that name's repo-bundled default
#   TOKENIZER_REVISION   HF Hub revision (only meaningful for a Hub repo id, not a local dir)
#   PUBLISH_EVERY        default 8 (retokenizing shards already downloaded is CPU-bound and
#                        fast per shard compared to the original download+tokenize run, so a
#                        coarser publish cadence than the default-1 first-time-tokenize case is
#                        fine; override if incremental-consumption during a long re-tokenize
#                        run matters for your case)
#   TRAIN_ENV            conda env (or .venv-<name>) name of the training environment, default `looptrain`
#
# Usage:
#   RAW_DIR=pretrain/data/raw/sample-100BT \
#   OUT_DIR=pretrain/data/tokenized/sample-100BT-smollm2 \
#   TOKENIZER_NAME=smollm2 \
#   NUM_WORKERS=16 \
#     bash scripts/data/retokenize_shards.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

: "${RAW_DIR:?RAW_DIR is required (no default -- which shards to re-tokenize is a semantic choice)}"
: "${OUT_DIR:?OUT_DIR is required (no default -- must be a NEW directory, see script header)}"
: "${TOKENIZER_NAME:?TOKENIZER_NAME is required (no default -- llama3 or smollm2)}"
: "${NUM_WORKERS:?NUM_WORKERS is required (no default -- see tokenize_to_bin_idx.py --num-workers)}"
TOKENIZER_REPO="${TOKENIZER_REPO:-}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-}"
PUBLISH_EVERY="${PUBLISH_EVERY:-8}"
ENV_NAME="${TRAIN_ENV:-looptrain}"

if [ -e "$OUT_DIR" ] || [ -e "${OUT_DIR}.staging" ]; then
    echo "refusing to run: OUT_DIR '$OUT_DIR' (or its .staging dir) already exists -- " \
         "re-tokenizing with a different tokenizer must go to a NEW directory, never overwrite " \
         "an existing one (old-directory-unchanged is the whole point of this task)." >&2
    exit 1
fi

if command -v conda >/dev/null 2>&1 && conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    PY=(conda run -n "${ENV_NAME}" python)
elif [ -x ".venv-${ENV_NAME}/bin/python" ]; then
    PY=("${REPO_ROOT}/.venv-${ENV_NAME}/bin/python")
else
    echo "No conda env '${ENV_NAME}' and no .venv-${ENV_NAME}/ found." \
         "Create the training environment first (README, Installation), or set TRAIN_ENV to an existing env." >&2
    exit 1
fi

TOKENIZER_ARGS=(--tokenizer-name "$TOKENIZER_NAME")
if [ -n "$TOKENIZER_REPO" ]; then
    TOKENIZER_ARGS+=(--tokenizer-repo "$TOKENIZER_REPO")
fi
if [ -n "$TOKENIZER_REVISION" ]; then
    TOKENIZER_ARGS+=(--tokenizer-revision "$TOKENIZER_REVISION")
fi

echo "== re-tokenizing $RAW_DIR -> $OUT_DIR with tokenizer-name=$TOKENIZER_NAME" \
     "repo-override=${TOKENIZER_REPO:-<default>} revision=${TOKENIZER_REVISION:-<none>}," \
     "num-workers=$NUM_WORKERS =="
"${PY[@]}" -m src.data.tokenize_to_bin_idx \
    --raw-dir "$RAW_DIR" \
    --out-dir "$OUT_DIR" \
    "${TOKENIZER_ARGS[@]}" \
    --num-workers "$NUM_WORKERS" \
    --publish-every "$PUBLISH_EVERY"

echo "== verifying with build_manifest.py's independent guard =="
N_SHARDS=$("${PY[@]}" -c "import json; print(len(json.load(open('$RAW_DIR/raw_manifest.json'))['shards']))")
"${PY[@]}" -m src.data.build_manifest \
    --ledger-path "${OUT_DIR}.staging/tokenize_ledger.jsonl" \
    --num-shards "$N_SHARDS" \
    --raw-manifest "$RAW_DIR/raw_manifest.json" \
    --out "${OUT_DIR}.staging/verified_manifest.json"

echo "done -- new tokenized corpus at $OUT_DIR, verified manifest at ${OUT_DIR}.staging/verified_manifest.json"

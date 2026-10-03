#!/bin/bash
# Print (never execute) the torchrun launch command for one registered
# from-scratch run (continuation runs need a starting checkpoint: use
# `python -m src.pretrain.print_run_commands --continuation-source ...`).
# If recipe.py's torchrun_command signature or the RUN_LIST lookup changes, the
# single adaptation point is the inline python block near the bottom of this file.
#
# The package and submodule are detected from --repo-root's layout
# (<pkg>/pretrain/recipe.py or <pkg>/train/recipe.py).
#
# All four parameters are required, no defaults -- a plausible-looking
# guessed path here would launch against the wrong tree or the wrong data and
# say nothing.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: launch_ablation.sh --repo-root DIR --config NAME --data-manifest DIR --out-root DIR [--nproc N]

  --repo-root      Repository root (the directory containing src/). Required.
  --config         Run name as registered in that repo's RUN_LIST (e.g. an
                    ablation config once registered, such as S1 or L3).
                    Required.
  --data-manifest  Tokenized data directory, passed through as torchrun_command's
                    data_dir. Required.
  --out-root       Output directory for this run, passed through as
                    torchrun_command's out_dir. Required.
  --nproc          GPUs per node (default: 8, matching an 8xH200 node).

Prints the torchrun command to stdout. Does not execute it, does not touch
the network, does not write any file.
USAGE
}

REPO_ROOT=""
CONFIG=""
DATA_MANIFEST=""
OUT_ROOT=""
NPROC=8

while [ $# -gt 0 ]; do
  case "$1" in
    --repo-root) REPO_ROOT=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --data-manifest) DATA_MANIFEST=$2; shift 2 ;;
    --out-root) OUT_ROOT=$2; shift 2 ;;
    --nproc) NPROC=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

for name in REPO_ROOT CONFIG DATA_MANIFEST OUT_ROOT; do
  if [ -z "${!name}" ]; then
    echo "missing required argument (see --help): $name" >&2
    exit 2
  fi
done

if [ ! -d "$REPO_ROOT" ]; then
  echo "repo root does not exist: $REPO_ROOT" >&2
  exit 2
fi

# Package name at the top level of the repository.
if [ -d "$REPO_ROOT/src" ]; then
  PKG=src
else
  echo "cannot determine package name at $REPO_ROOT: no src/ directory" >&2
  exit 2
fi

# Submodule holding recipe.py (`pretrain` in this repository).
SUBMODULE=""
for candidate in pretrain train; do
  if [ -f "$REPO_ROOT/$PKG/$candidate/recipe.py" ]; then
    SUBMODULE=$candidate
    break
  fi
done
if [ -z "$SUBMODULE" ]; then
  echo "cannot find recipe.py under $REPO_ROOT/$PKG/{pretrain,train}/ -- checked both known submodule names" >&2
  exit 2
fi

MODULE="$PKG.$SUBMODULE.recipe"

# The one place this script depends on recipe.py's shape: importing RUN_LIST
# and calling .torchrun_command(data_dir=, out_dir=, nproc=). If that
# interface moves, this block is the only thing to edit.
# Which interpreter reads recipe.py. `python3` off PATH only works when the
# caller has activated the training env; `$LAUNCH_PY` lets a caller that already
# knows its interpreter (a job script) say so instead of relying on an
# ambient PATH -- the failure otherwise is "ModuleNotFoundError: torch" from a
# script that looks like it is broken.
LAUNCH_PY=${LAUNCH_PY:-${LOOPTRAIN_PY:-python3}}
PYTHONPATH="$REPO_ROOT" "$LAUNCH_PY" - "$MODULE" "$CONFIG" "$DATA_MANIFEST" "$OUT_ROOT" "$NPROC" <<'PY'
import importlib
import sys

module_name, config, data_manifest, out_root, nproc = sys.argv[1:6]
recipe = importlib.import_module(module_name)

matches = [r for r in recipe.RUN_LIST if r.name == config]
if not matches:
    available = ", ".join(sorted(r.name for r in recipe.RUN_LIST)) or "(RUN_LIST is empty)"
    sys.exit(
        f"unknown --config {config!r} in {module_name}.RUN_LIST; "
        f"registered run names: {available}"
    )
run = matches[0]
print(run.torchrun_command(data_dir=data_manifest, out_dir=out_root, nproc=int(nproc)))
PY

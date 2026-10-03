#!/bin/bash
# Generate (and, only if asked, submit) an sbatch script for one ablation
# run, single 8-GPU node. This does NOT reimplement the
# torchrun-command logic: it shells out to launch_ablation.sh (this
# directory) for that, so there is exactly one place that knows how to turn
# a config name into a torchrun command, however the ablation configs end
# up registered.
#
# Partition, account and GPU-resource syntax differ between clusters, so
# they are required arguments here, never guessed. Default action is --dry-run: print the
# generated sbatch script and stop. Pass --submit to actually call `sbatch`
# (only meaningful on a node that can reach the target cluster's scheduler).
#
# All arguments are required, no defaults -- a plausible-looking guessed
# partition/account/walltime would submit successfully and either burn a
# job slot on the wrong queue or (worse) run silently against the wrong
# cluster's node fraction.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAUNCH_ABLATION="$SCRIPT_DIR/launch_ablation.sh"

usage() {
  cat >&2 <<'USAGE'
Usage: launch_ablation_slurm.sh --repo-root DIR --config NAME \
         --data-manifest DIR --out-root DIR \
         --partition NAME --account NAME --walltime HH:MM:SS \
         --gpu-directive STRING --conda-env NAME \
         [--nproc N] [--job-name NAME] [--submit]

  --repo-root      Repository root (the directory containing src/).
  --config         Run name as registered in that repo's RUN_LIST.
  --data-manifest  Tokenized data directory (torchrun_command's data_dir).
  --out-root       Output directory for this run (torchrun_command's out_dir).
  --partition      SLURM partition name on the target cluster. Required --
                    no cluster-wide default exists across sites.
  --account        SLURM account name on the target cluster. Required.
  --walltime       SLURM --time value, e.g. 24:00:00. Required.
  --gpu-directive  The raw SBATCH line(s) requesting 8 GPUs on that cluster,
                    e.g. "#SBATCH --gres=gpu:h200:8" or
                    "#SBATCH --gpus-per-node=8" -- GPU resource syntax is
                    cluster-specific and NOT guessed here. Required.
  --conda-env      Name of the training conda environment (e.g. looptrain).
                    Required -- see this project's rule that semantic
                    parameters carry no silent defaults.
  --nproc          GPUs per node for torchrun (default: 8).
  --job-name       SLURM job name (default: ablation_<config>).
  --submit         Actually call `sbatch` on the generated script. Without
                    this flag, the script is only printed (dry run) -- the
                    default, since we cannot verify submission works from
                    here.

Never places --output under /tmp; the generated script always writes its
SLURM log under --out-root.
USAGE
}

REPO_ROOT=""; CONFIG=""; DATA_MANIFEST=""; OUT_ROOT=""
PARTITION=""; ACCOUNT=""; WALLTIME=""; GPU_DIRECTIVE=""; CONDA_ENV=""
NPROC=8; JOB_NAME=""; SUBMIT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --repo-root) REPO_ROOT=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --data-manifest) DATA_MANIFEST=$2; shift 2 ;;
    --out-root) OUT_ROOT=$2; shift 2 ;;
    --partition) PARTITION=$2; shift 2 ;;
    --account) ACCOUNT=$2; shift 2 ;;
    --walltime) WALLTIME=$2; shift 2 ;;
    --gpu-directive) GPU_DIRECTIVE=$2; shift 2 ;;
    --conda-env) CONDA_ENV=$2; shift 2 ;;
    --nproc) NPROC=$2; shift 2 ;;
    --job-name) JOB_NAME=$2; shift 2 ;;
    --submit) SUBMIT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

for name in REPO_ROOT CONFIG DATA_MANIFEST OUT_ROOT PARTITION ACCOUNT WALLTIME GPU_DIRECTIVE CONDA_ENV; do
  if [ -z "${!name}" ]; then
    echo "missing required argument (see --help): $name" >&2
    exit 2
  fi
done

[ -d "$REPO_ROOT" ] || { echo "repo root does not exist: $REPO_ROOT" >&2; exit 2; }
[ -x "$LAUNCH_ABLATION" ] || { echo "cannot find launch_ablation.sh next to this script" >&2; exit 2; }

JOB_NAME=${JOB_NAME:-"ablation_${CONFIG}"}

# The one place this script depends on another: resolving the torchrun
# command via launch_ablation.sh rather than reimplementing that lookup.
# If it fails (e.g. --config isn't registered yet), fail here rather than
# generating an sbatch script with a broken or empty launch line.
TORCHRUN_CMD=$("$LAUNCH_ABLATION" \
  --repo-root "$REPO_ROOT" --config "$CONFIG" \
  --data-manifest "$DATA_MANIFEST" --out-root "$OUT_ROOT" --nproc "$NPROC")

mkdir -p "$OUT_ROOT"
SBATCH_SCRIPT=$(mktemp "${TMPDIR:-/tmp}/launch_ablation_slurm.XXXXXX.sbatch")
trap 'rm -f "$SBATCH_SCRIPT"' EXIT

cat > "$SBATCH_SCRIPT" <<SBATCH
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
${GPU_DIRECTIVE}
#SBATCH --time=${WALLTIME}
# Never /tmp: a shared node's /tmp is not shared storage, so the log could
# be lost across a node change. Always the run's own out-root, which is
# shared storage.
#SBATCH --output=${OUT_ROOT}/slurm_%x_%j.out

set -uo pipefail
cd "${REPO_ROOT}" || exit 2

echo "=== [0] node info / GPU driver check \$(date) @ \$(hostname) ==="
nvidia-smi || true

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
# Ignore user-site packages (~/.local) so the run uses only this
# environment's packages.
export PYTHONNOUSERSITE=1

echo "=== [1] launch \$(date) ==="
${TORCHRUN_CMD}
RC=\$?
echo "=== done rc=\$RC \$(date) ==="
exit \$RC
SBATCH

if [ "$SUBMIT" = 1 ]; then
  FINAL="${OUT_ROOT}/${JOB_NAME}.sbatch"
  cp "$SBATCH_SCRIPT" "$FINAL"
  echo "submitting: sbatch $FINAL" >&2
  sbatch "$FINAL"
else
  echo "# --dry-run (default): not submitted. Generated script:" >&2
  cat "$SBATCH_SCRIPT"
fi

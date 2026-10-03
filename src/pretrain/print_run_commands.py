"""Print the launch command for every planned run.

The single source for launch commands: regenerate rather than copy, so a recipe
change cannot leave a stale command in a document someone then runs.

    python -m src.pretrain.print_run_commands --repo '$REPO' [--data DIR] [--markdown]

`--repo` is substituted into the data and output paths; leaving it as the literal
string `$REPO` produces commands that a PBS script can expand itself.
"""

from __future__ import annotations

import argparse

from src.model.config_registry import build
from src.pretrain.recipe import RUN_LIST, command_for

DEFAULT_DATA_SUBDIR = "pretrain/data/tokenized/sample-100BT"
"""Where the tokenize step publishes. Keep in sync with the --out-dir given to
`src/data/tokenize_to_bin_idx.py` (see README, Data preparation)."""


def _parameter_count(config) -> int:
    """Total parameters, counted from a model built on the meta device.

    Counted, not derived. The closed-form expression this replaced summed the
    looped stack and the embeddings, which is the whole model only while there
    are no layers outside the loop -- it reported 436M for S1 against the
    recipe's 520M, and 436M again for S15, whose head/tail layers are twice as
    thick. A formula stays right until the architecture grows a part it does not
    know about, and then it is wrong silently, in the one line a reader skims
    before launching a run.

    The meta device allocates no storage, so this costs a module tree and no
    memory. For every (architecture, tier) run the count is unchanged from the
    old formula to the digit.
    """
    import torch

    from src.model.modeling_loop_lm import LoopMoEForCausalLM

    with torch.device("meta"):
        model = LoopMoEForCausalLM(config)
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="$REPO", help="repo root, substituted into paths")
    parser.add_argument(
        "--data", default=None,
        help="tokenized data dir (default: <repo>/pretrain/data/tokenized/sample-100BT)",
    )
    parser.add_argument("--nproc", type=int, default=8, help="GPUs per node")
    parser.add_argument("--markdown", action="store_true", help="wrap in markdown")
    parser.add_argument(
        "--resume-checkpoint-interval",
        dest="resume_checkpoint_interval",
        type=int,
        default=None,
        help="steps between resumable checkpoints (default: the recipe's "
             "RESUME_CHECKPOINT_INTERVAL_STEPS). To retune it for a cluster's "
             "preemption rate, pass it here rather than sed-ing the "
             "generated command, so the plan and the job agree.",
    )
    parser.add_argument(
        "--continuation-source",
        default="{repo}/_runs/{run}/checkpoint/step-{step}",
        help="path template for a continuation's starting checkpoint, with "
             "{repo}, {run} and {step}. The default points at this repo's own "
             "_runs tree; on a cluster whose finished runs live elsewhere, pass "
             "the real location -- a continuation command that names the wrong "
             "checkpoint is the one failure this whole path is built to avoid.",
    )
    args = parser.parse_args()

    # Must match where tokenization wrote: `tokenize_to_bin_idx` writes to a
    # directory named after the HF dataset subset (`sample-100BT`). A mismatch
    # here is launch-blocking and silent until the job starts -- every run would
    # die on FileNotFoundError after queueing.
    data_dir = args.data or f"{args.repo}/{DEFAULT_DATA_SUBDIR}"

    for run in RUN_LIST:
        # Through the plan, which knows which of the two builders applies. Calling
        # `build(run.arch, run.tier)` here raised `unknown architecture 'S1'` the
        # moment the ablation family entered RUN_LIST.
        config = run.build_config()
        d, L, E = config.d_model, config.num_layers_in_stack, config.num_experts
        nparams = _parameter_count(config)
        header = (
            f"{run.name}  |  {run.arch} @ {run.tier} (d_model={d}, L={L}, "
            f"R={config.num_stacks}, E={E})  |  {nparams/1e6:.0f}M params  |  "
            f"seed {run.seed}  |  {run.total_tokens/1e9:.3f}B tokens  |  "
            f"{run.steps:,} steps  |  warmup {run.warmup_steps:,}  |  "
            f"{len(run.trajectory_checkpoints)} trajectory checkpoints"
        )
        command = command_for(
            run,
            data_dir=data_dir, out_dir=f"{args.repo}/_runs/{run.name}", nproc=args.nproc,
            continuation_source=args.continuation_source.replace("{repo}", args.repo),
            resume_checkpoint_interval=args.resume_checkpoint_interval,
        )
        if args.markdown:
            print(f"### {header}\n\n```bash\n{command}\n```\n")
        else:
            print(f"# {header}\n{command}\n")


if __name__ == "__main__":
    main()

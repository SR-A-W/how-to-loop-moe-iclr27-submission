"""Production launcher: torchtitan Trainer with our looped-MoE TrainSpec.

Run under torchrun, e.g. 4-GPU single node::

    torchrun --nproc_per_node=4 -m src.pretrain.train_titan \\
        --arch toy --data /abs/path/to/tokenized/smoke --seed 42 \\
        --job.dump_folder /abs/path/to/run1 \\
        --training.steps 20 --training.seq_len 512 --training.local_batch_size 2

Why a launcher instead of `python -m torchtitan.train`: `register_train_spec`
has to run before the Trainer looks the spec up, and torchtitan's entry point
imports nothing of ours. This module registers, then delegates to torchtitan's
own `main` -- so the actual training loop, FSDP2 wrapping, DCP checkpointing and
metrics all remain torchtitan's code, not a reimplementation.

Our arguments (`--arch`, `--data`, `--seed`, `--verify-sha256`) are consumed here
and stripped; everything else is passed through to torchtitan's config parser
untouched, so its full `--section.field` surface stays available.

Import order is load-bearing
----------------------------
torchtitan's config manager is declared as::

    def parse_args(self, args: list[str] = sys.argv[1:]) -> JobConfig:

Python evaluates that default **once, when the module is imported**, so
`sys.argv[1:]` is snapshotted at import time. Rewriting `sys.argv` afterwards has
no effect whatsoever -- torchtitan still sees the original command line and
rejects our `--arch`/`--data`/`--seed` flags as unrecognised.

Hence: parse our own flags and rewrite `sys.argv` **before importing anything
that pulls in torchtitan**, and only then import it. The torchtitan imports below
are deliberately inside `main()`; moving them to module scope silently breaks
argument handling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# NOTE: config_registry only reaches transformers, not torchtitan, so it is safe
# at module scope. Anything importing `train_spec` is not.
from src.model.config_registry import ABLATIONS, ARCHITECTURES, ATTENTION_VARIANTS, TIERS, build

SPEC_NAME = "loopmoe"
"""Value to pass as `--model.name`. Set automatically below."""


def build_parser() -> argparse.ArgumentParser:
    """Our flags, separate from torchtitan's.

    Exposed so tests can push every generated command through the real parser.
    Generated commands and this parser are two artifacts that must agree; when they
    drift, the mismatch stays invisible until a job dies in argparse after
    queueing.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--arch", required=True,
        choices=[*sorted(ARCHITECTURES), *sorted(ABLATIONS), *sorted(ATTENTION_VARIANTS), "toy"],
        help=(
            "which architecture to train: an (architecture, tier) name from "
            "ARCHITECTURES, an ablation configuration name (S1-S15, L1-L3), a "
            "per-pass-attention variant (U1-U4, UL1-UL2), or 'toy' (see "
            "src/model/config_registry.py). Ablation names and variant names take no --tier."
        ),
    )
    parser.add_argument(
        "--tier", default=None, choices=sorted(TIERS),
        help="size tier; required for every architecture except 'toy'",
    )
    parser.add_argument(
        "--run-name", dest="run_name", default=None,
        help=(
            "the run's own name from src.pretrain.recipe.RUN_LIST (e.g. "
            "noloop_moe_1p3b_lb0p02). Recorded in every checkpoint manifest so a "
            "checkpoint states which run produced it without relying on its "
            "directory name. Optional: without it, manifests record null rather "
            "than a guess"
        ),
    )
    parser.add_argument(
        "--lb-loss-factor", dest="lb_loss_factor", type=float, default=None,
        help=(
            "override the load-balancing coefficient from config_registry.SHARED; "
            "omit to use the registry value (0.01). Only a run that deliberately "
            "departs from it passes this -- the value reaches the checkpoint "
            "manifest via model_config, so a checkpoint states which coefficient "
            "trained it"
        ),
    )
    parser.add_argument(
        "--allow-steps-override", dest="allow_steps_override", action="store_true",
        help=(
            "run a registered plan with a --training.steps that differs from its "
            "planned length (smoke tests). Without this flag the mismatch is an "
            "error: a shortened run that keeps the full plan's schedule silently "
            "produces different artifacts from the ones its listing promises"
        ),
    )
    parser.add_argument(
        "--data", required=True, help="absolute path to a tokenized shard directory",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="data-order seed (the data order is determined by the seed)",
    )
    parser.add_argument(
        "--verify-sha256", dest="verify_sha256",
        action=argparse.BooleanOptionalAction, required=True,
        help="check shard hashes at load time",
    )
    parser.add_argument(
        "--trace-every", dest="trace_every", type=int, required=True,
        help="collect router/residual metrics every N steps (0 disables)",
    )
    parser.add_argument(
        "--metrics-out", dest="metrics_out", default=None,
        help="JSONL metrics path (default: <job.dump_folder>/train_metrics.jsonl)",
    )
    parser.add_argument(
        "--on-boundary", dest="on_boundary", required=True, choices=["raise", "wait"],
        help=(
            "behaviour when training catches up to the tokenizer: 'raise' fails fast "
            "(production -- a PBS job should not block silently), 'wait' polls "
            "(small streaming runs only; gives up reproducible data order)"
        ),
    )
    return parser


def flag_value(passthrough: list[str], name: str, default: str) -> str:
    """The value torchtitan will actually use for `name` -- the LAST occurrence.

    Why last, and why this function is not an implementation detail: torchtitan
    parses its arguments with argparse, where a repeated flag keeps the last
    value given (verified against v0.2.2: `--activation_checkpoint.mode selective
    ... --activation_checkpoint.mode none` yields `none`). This function used to
    return the FIRST occurrence, so the two disagreed whenever a flag appeared
    twice -- which is exactly what happens when an operator appends an override
    to a generated command instead of editing it in place.

    The consequence was silent and bad. Appending `--training.steps 20` to a
    command that already said `--training.steps 50000` left the guard in
    `resolve_plan` looking at 50,000: it matched the registered plan, so nothing
    was refused and `--allow-steps-override` was not required, while torchtitan
    ran 20 steps. The manifest then recorded `training_steps: 50000` and
    `steps_overridden: false` -- a checkpoint that states a run length it never
    had, which is the one thing a manifest exists to prevent.

    Both spellings are scanned (`--flag value` and `--flag=value`), because a
    command may mix them and only the position matters.
    """
    value = default
    for i, token in enumerate(passthrough):
        if token == name and i + 1 < len(passthrough):
            value = passthrough[i + 1]
        elif token.startswith(name + "="):
            value = token.split("=", 1)[1]
    return value


def build_config(args):
    """Build the model configuration this run asked for.

    Two families, built two different ways. The (architecture, tier) family takes
    its width from `--tier`; an ablation configuration carries its width inside its
    registry entry, so `--tier` has no meaning there and is REFUSED rather than
    ignored: accepting and discarding it would let a command say `S1 --tier 1p3b`
    and train something that is not what the command says.
    """
    from src.model.config_registry import ABLATIONS, ATTENTION_VARIANTS, build_ablation

    # A variant name (U1-U4) is ablation-built too -- `build_ablation` resolves
    # it to its source shape internally -- so it is refused --tier/--lb-loss-factor
    # for the same reason as any other ablation configuration.
    if args.arch in ABLATIONS or args.arch in ATTENTION_VARIANTS:
        if args.tier is not None:
            raise SystemExit(
                f"--tier is not applicable to ablation configuration {args.arch!r}: its width "
                "tier comes from the registry entry. Drop --tier."
            )
        if args.lb_loss_factor is not None:
            raise SystemExit(
                f"--lb-loss-factor is not applicable to ablation configuration {args.arch!r}: "
                "the family shares one coefficient, and a per-run override would make the "
                "eighteen runs incomparable in a way no artifact records."
            )
        return build_ablation(args.arch)
    return build(args.arch, args.tier, lb_loss_factor=args.lb_loss_factor)


def resolve_plan(args, steps: int):
    """The run's plan, taken from `RUN_LIST` rather than rebuilt here.

    This function exists because rebuilding it here was wrong in a way nothing
    caught. The old code constructed a fresh `RunPlan` from `--arch` and the step
    count, which silently dropped every field the registry entry carries but the
    command line does not -- `trajectory_uniform_every` above all. The result:
    the launch listing promised the recipe's 15 trajectory checkpoints while the
    trainer would have written 13 at a different stride, and the contract tests
    agreed with the listing because they tested `RUN_LIST`, not the plan the
    trainer actually builds. Two sources for one fact, diverging silently.

    `--training.steps` shorter than the plan's is how a smoke test is run, so it
    is allowed -- but only when said out loud (`--allow-steps-override`), and the
    manifest records that it happened. Silently honouring it would let a 200-step
    job claim the schedule of a 50,000-step one.
    """
    from dataclasses import replace

    from src.pretrain.recipe import LT2_RECIPE, RUN_LIST, RunPlan

    name = args.run_name or args.arch
    listed = next((p for p in RUN_LIST if p.name == name), None)

    if listed is None:
        if args.arch != "toy":
            raise SystemExit(
                f"no run named {name!r} in recipe.RUN_LIST. Every production run is "
                "launched from a registered plan, because the plan carries settings the "
                "command line does not (the trajectory schedule among them). Register it "
                "in RUN_LIST, or pass --run-name naming an entry that exists."
            )
        return RunPlan(
            name=name, arch=args.arch, tier=args.tier or "toy",
            seed=args.seed, total_tokens=steps * LT2_RECIPE.tokens_per_step,
        )

    if steps == listed.steps:
        return listed
    if not args.allow_steps_override:
        raise SystemExit(
            f"--training.steps {steps} disagrees with run {name!r}'s planned {listed.steps}. "
            "Pass --allow-steps-override to run a shortened version deliberately; the "
            "manifest will record that the step count was not the recipe's."
        )
    # A shortened run gets a schedule proportional to its length: the family's
    # 4,000-step stride on a 200-step smoke would put every checkpoint in the
    # first few steps and one at the end.
    return replace(
        listed,
        total_tokens=steps * listed.recipe.tokens_per_step,
        trajectory_uniform_every=None,
    )


def _registered_steps(args) -> int | None:
    """The planned step count for this run name, or None if it is not registered."""
    from src.pretrain.recipe import RUN_LIST

    listed = next((p for p in RUN_LIST if p.name == (args.run_name or args.arch)), None)
    return listed.steps if listed else None


def main() -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    config = build_config(args)

    # Rewrite argv BEFORE importing torchtitan -- see "Import order is
    # load-bearing" above. Pin model name/flavor to what we register, so a caller
    # cannot ask for an architecture that was never built.
    sys.argv = [
        sys.argv[0],
        *passthrough,
        "--model.name", SPEC_NAME,
        "--model.flavor", "default",
        # seq_len must agree with the architecture; `update_from_config` raises
        # on a mismatch rather than reconciling silently.
        "--training.seq_len", str(config.context_length),
    ]

    from src.pretrain.train_spec import register_loopmoe

    register_loopmoe(
        SPEC_NAME,
        config,
        tokenized_dir=args.data,
        seed=args.seed,
        verify_sha256=args.verify_sha256,
        on_boundary=args.on_boundary,
    )

    # Trajectory checkpoints + metrics. torchtitan's own DCP checkpointing keeps
    # running for resume; this adds the HF-format science output it cannot
    # produce, on the same schedule the toy harness uses.
    import re

    from src.pretrain.trainer import TRAJECTORY_STATE, LoopMoETrainer
    from src.pretrain.recipe import LT2_RECIPE, RunPlan

    steps = int(flag_value(passthrough, "--training.steps", str(LT2_RECIPE.steps_for())))
    dump_folder = flag_value(passthrough, "--job.dump_folder", "./outputs")
    plan = resolve_plan(args, steps)

    collector = None
    if args.trace_every >= 0:
        from src.pretrain.jsonl_collector import JsonlMetricsCollector

        metrics_path = args.metrics_out or f"{dump_folder}/train_metrics.jsonl"
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        collector = JsonlMetricsCollector(train_metrics_path=metrics_path)

    TRAJECTORY_STATE.configure(
        steps=plan.trajectory_checkpoints,
        out_dir=dump_folder,
        trace_every=args.trace_every,
        collector=collector,
        config=config,
        # The run's name, falling back to the architecture only for commands
        # that predate `--run-name`. `manifest_run_name` is what reaches the
        # manifest, and it stays None in that case rather than recording an
        # architecture name as if it were a run name.
        run_name=args.run_name or args.arch,
        manifest_run_name=args.run_name,
        # Always known -- `--arch` is required -- so unlike `run_name` this needs
        # no fallback story.
        arch=args.arch,
        # What this job actually runs, and whether that is the registered plan's
        # length. A 200-step smoke and a 50,000-step production run otherwise
        # produce manifests that differ only in fields nobody compares.
        training_steps=plan.steps,
        steps_overridden=plan.steps != _registered_steps(args),
    )

    from torchtitan.train import main as titan_main

    try:
        titan_main(LoopMoETrainer)
    finally:
        if collector is not None:
            collector.close()


if __name__ == "__main__":
    main()

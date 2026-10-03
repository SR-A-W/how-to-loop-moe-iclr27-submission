"""Integration harness: train a small looped-MoE for a few hundred steps.

**This is not the production trainer.** Production runs go through torchtitan
(`src/pretrain/train_spec.py`), which supplies FSDP2, DCP and the 8-GPU
plumbing. This script exists to prove, on CPU and without a scheduler queue, that
the pieces actually fit together:

    the bin/idx loader -> the model -> aux losses -> metrics JSONL
    -> trajectory checkpoints -> resume

The acceptance criterion: a ~1k-step toy run whose checkpoints the
diagnostics pipeline can ingest, and whose resume is indistinguishable from an
uninterrupted run.

Deliberately kept to one file and one process. It shares every component with the
production path except the trainer loop itself, so the two cannot drift on
anything that matters (model, data, losses, checkpoint format, metrics schema).

Usage (all shape parameters are required -- no defaults):

    python -m src.pretrain.train_toy --steps 200 --seq-len 512 --batch-size 2 \
        --d-model 128 --layers 2 --loops 2 --experts 4 \
        --data pretrain/data/tokenized/smoke --out /tmp/toy_run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.pretrain.checkpointing.schedule import estimate_disk_bytes, trajectory_steps
from src.pretrain.checkpointing.store import load_resume, save_resume, save_trajectory
from src.model.loop_trace import LossComponents, assert_trace_released
from src.model.modeling_loop_lm import LoopMoEConfig, LoopMoEForCausalLM
from src.metrics.provenance import resolve_code_revision


def _common_manifest_fields(config, model, step: int, tokens_consumed: int, dataset) -> dict:
    """Manifest fields shared by both checkpoint kinds.

    `data_position` comes from the dataset's own `state_dict()` rather than being
    recomputed as `step * batch_size`: the dataset knows which shard and byte
    offset it is actually at, and a derived guess would drift the moment sharding
    or packing changed -- while still looking plausible in the manifest.
    """
    # Resolved together so the recorded source cannot disagree with the
    # recorded commit -- see `resolve_code_revision`.
    rev, rev_source = resolve_code_revision()
    return dict(
        step=step,
        tokens_consumed=tokens_consumed,
        git_commit=rev["commit"],
        code_dirty=rev["dirty"],
        code_diff_sha256=rev["diff_sha256"],
        code_revision_source=rev_source,
        compute_dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
        autocast={"enabled": False},
        torch_version=torch.__version__,
        transformers_version=__import__("transformers").__version__,
        model_config=config.to_dict(),
        data_position=dataset.state_dict(),
        perturbation_probe={"runs_offline": True},
    )


def build_config(args) -> LoopMoEConfig:
    """Every semantic field stated explicitly; none inherited from a default."""
    return LoopMoEConfig(
        vocab_size=args.vocab_size,
        context_length=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        d_ff=args.d_ff,
        rope_theta=10000.0,
        width_ratio=args.d_model / 128,  # muP: d_model / d_base
        num_layers_in_stack=args.layers,
        num_stacks=args.loops,
        num_experts=args.experts,
        num_active=args.active,
        lb_loss_factor=0.01,
        lz_loss_factor=0.001,
    )


def make_loader(args, seed: int, state: dict | None = None):
    """Build the dataset, optionally restored to a saved position.

    The position must be reinstalled through the dataset's own `load_state_dict`
    rather than by skipping N batches: skipping would re-consume the RNG and give
    a resumed run a different data order, which is exactly the divergence a
    replay-identical resume forbids.
    """
    from src.data.torchtitan_adapter import PackedSeqIterableDataset

    # Fail legibly if the dataset directory is incomplete. Tokenized sets are
    # rebuilt in place, so a training run can arrive mid-rewrite and see shards
    # without their manifest. Without this check that surfaces as a bare
    # FileNotFoundError from inside the reader, and the reader is several layers
    # down from anything that hints at the real cause.
    data_dir = Path(args.data)
    manifest = data_dir / "manifest.json"
    if not manifest.is_file():
        present = sorted(p.name for p in data_dir.glob("*")) if data_dir.is_dir() else []
        raise RuntimeError(
            f"{data_dir} has no manifest.json, so the dataset is either absent or "
            f"still being written. Present files: {present or '(none)'}. "
            "Wait for the tokenization job to finish before training on it."
        )

    dataset = PackedSeqIterableDataset(
        tokenized_dir=args.data,
        seq_len=args.seq_len,
        seed=seed,
        dp_rank=0,
        dp_world_size=1,
        # Not hardcoded: the point of the hash check is to catch corruption of the
        # data being trained on *now*, which is a different guarantee from "the
        # data tests exercise the flag".
        verify_sha256=args.verify_sha256,
        # Required, no default, deliberately (see the adapter's docstring): the
        # two behaviours -- block until more corpus is written, or fail -- must
        # each be chosen consciously. For toy runs the answer is always "raise":
        # a toy corpus is a small fixed set that is fully written before the run
        # starts, so "wait" could only ever hang forever on a corpus that is
        # never going to grow. Production reads this from its data config
        # instead (`train_spec.py`), where "wait" is a real option.
        on_boundary="raise",
    )
    if state is not None:
        dataset.load_state_dict(state)
    return dataset


def batches(dataset, batch_size: int):
    """Group the loader's (input_dict, labels) pairs into batches."""
    inputs, labels = [], []
    for input_dict, label in iter(dataset):
        inputs.append(input_dict["input"])
        labels.append(label)
        if len(inputs) == batch_size:
            yield torch.stack(inputs), torch.stack(labels)
            inputs, labels = [], []


def train(args) -> dict:
    torch.manual_seed(args.seed)
    config = build_config(args)
    model = LoopMoEForCausalLM(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Resume before anything consumes randomness, so the restored RNG streams are
    # the ones the continued run actually uses.
    start_step = 0
    loader_state = None
    if args.resume_from:
        info = load_resume(args.resume_from, model=model, optimizer=optimizer)
        start_step = info["step"]
        loader_state = info["dataloader"]
        print(f"[toy] resumed from {args.resume_from} at step {start_step}")

    nparams = sum(p.numel() for p in model.parameters())
    ckpt_steps = set(
        trajectory_steps(args.steps, first_step=args.first_ckpt, uniform_every=args.ckpt_every)
    )
    budget = estimate_disk_bytes(
        nparams=nparams, n_trajectory_points=len(ckpt_steps), n_resume_kept=1
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[toy] params={nparams:,} checkpoints={sorted(ckpt_steps)} "
        f"disk~{budget['total_bytes']/1e6:.1f} MB"
    )

    # The real collector, not a hand-rolled loss dump. The point of this harness
    # is to prove the metrics chain works end to end; writing our own JSONL here
    # would have verified everything except the component that produces it.
    from src.pretrain.jsonl_collector import JsonlMetricsCollector

    collector = JsonlMetricsCollector(train_metrics_path=out / "train_metrics.jsonl")

    losses: list[float] = []
    tokens_consumed = 0
    started = time.time()

    dataset = make_loader(args, args.seed, state=loader_state)
    with collector:
        # Pull exactly the batches we will train on. Iterating the generator and
        # breaking on `step > steps` would draw one batch too many: the extra
        # batch is discarded, but the dataset cursor has already advanced past
        # it, so the saved position points one batch beyond the last trained
        # step and a resumed run silently skips a batch.
        #
        # That bug is invisible in the loss curve of any single run -- it only
        # shows up when a resumed run is compared step-by-step against an
        # uninterrupted one, which is why exactly that comparison is the test.
        data_iter = batches(dataset, args.batch_size)
        for step in range(start_step + 1, args.steps + 1):
            inputs, labels = next(data_iter)

            # Trace on a schedule: the tensors are free (detached views of live
            # activations), but the metric reductions over them are not.
            tracing = args.trace_every > 0 and step % args.trace_every == 0
            trace_sink: dict = {} if tracing else None

            out_ = model(inputs, labels=labels, trace_sink=trace_sink)
            optimizer.zero_grad()
            out_.loss.backward()
            # Keep the return value: it is the gradient norm *before* clipping,
            # and it is the only cheap evidence that the model is learning at
            # all. A run whose weights were zero-initialised produces a constant
            # loss of ln(vocab_size) and a gradient norm of exactly zero while
            # exiting 0 and writing every expected file -- the failure that cost
            # us a multi-card smoke run in August. The toy smoke acceptance
            # criteria ask for this number, so it has to reach the log.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tokens_consumed += labels.numel()
            # detach before float(): these tensors are still attached to the graph
            # at this point, and torch warns about the implicit detach.
            components = LossComponents(
                task_loss=out_.task_loss.detach().item(),
                lb_loss=out_.lb_loss.detach().item(),
                lb_loss_factor=config.lb_loss_factor,
                z_loss=out_.z_loss.detach().item(),
                z_loss_factor=config.lz_loss_factor,
                total_loss=out_.loss.detach().item(),
            )
            losses.append(components.total_loss)
            collector.on_metrics_step(step, components)
            if trace_sink is not None:
                collector.on_trace_step(step, trace_sink, config.trace_meta)
                # Enforce the lifetime contract on the real path, not just in
                # unit tests: a collector that retained these would leak whole
                # activation graphs and only surface as an OOM much later.
                assert_trace_released(trace_sink)

            if step % args.log_every == 0:
                print(
                    f"[toy] step {step:4d}  loss {components.total_loss:.4f}  "
                    f"task {components.task_loss:.4f}  lb {components.lb_loss:.4f}  "
                    f"gnorm {grad_norm.item():.4f}"
                )

            if step in ckpt_steps:
                save_trajectory(
                    model,
                    out / f"trajectory/step_{step:06d}",
                    manifest_fields=dict(
                        **_common_manifest_fields(
                            config, model, step, tokens_consumed, dataset
                        ),
                        storage_dtype="bfloat16",
                        rng_state_saved=False,  # trajectory checkpoints carry weights only
                    ),
                )

            # Periodic resume saves. Saving only at the end would mean a run
            # killed mid-flight has nothing to resume from -- which is precisely
            # the scenario the resume checkpoint exists for, and the one the
            # README's acceptance recipe ("kill the process, resume, compare")
            # actually describes.
            if args.resume_every > 0 and step % args.resume_every == 0:
                save_resume(
                    out / "resume/latest",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    dataloader_state=dataset.state_dict(),
                    manifest_fields=dict(
                        **_common_manifest_fields(
                            config, model, step, tokens_consumed, dataset
                        ),
                        storage_dtype="float32",  # full precision + optimizer state
                        rng_state_saved=True,
                    ),
                )

    elapsed = time.time() - started
    save_resume(
        out / "resume/latest",
        model=model,
        optimizer=optimizer,
        step=args.steps,
        # The dataset's own state, not a reconstructed guess: it records the
        # shard, offset and sample index that a restart has to land on exactly.
        dataloader_state=dataset.state_dict(),
        manifest_fields=dict(
            **_common_manifest_fields(config, model, args.steps, tokens_consumed, dataset),
            storage_dtype="float32",
            rng_state_saved=True,
        ),
    )

    return {
        "nparams": nparams,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_first_10": sum(losses[:10]) / len(losses[:10]),
        "mean_last_10": sum(losses[-10:]) / len(losses[-10:]),
        "tokens_consumed": tokens_consumed,
        "elapsed_s": elapsed,
        "tokens_per_s": tokens_consumed / elapsed,
        "checkpoints": sorted(ckpt_steps),
        "out": str(out),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seq-len", dest="seq_len", type=int, required=True)
    p.add_argument("--batch-size", dest="batch_size", type=int, required=True)
    p.add_argument("--d-model", dest="d_model", type=int, required=True)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--loops", type=int, required=True)
    p.add_argument("--experts", type=int, required=True)
    # These four map straight onto LoopMoEConfig fields that the config class
    # itself makes mandatory. Giving them argparse defaults would reintroduce, at
    # the CLI layer, exactly the silent defaulting the config forbids -- and this file's
    # own docstring promises they are required.
    p.add_argument("--active", type=int, required=True)
    p.add_argument("--heads", type=int, required=True)
    p.add_argument("--d-ff", dest="d_ff", type=int, required=True)
    p.add_argument("--vocab-size", dest="vocab_size", type=int, required=True)
    p.add_argument(
        "--verify-sha256",
        dest="verify_sha256",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="check shard hashes at load time; production runs must enable this",
    )
    p.add_argument(
        "--trace-every",
        dest="trace_every",
        type=int,
        required=True,
        help="collect router/residual metrics every N steps (0 disables)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", dest="log_every", type=int, default=25)
    p.add_argument("--first-ckpt", dest="first_ckpt", type=int, default=50)
    p.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=200)
    p.add_argument(
        "--resume-every",
        dest="resume_every",
        type=int,
        required=True,
        help="save a resume checkpoint every N steps (0 = only at the end)",
    )
    p.add_argument(
        "--resume-from",
        dest="resume_from",
        default=None,
        help="resume checkpoint directory; training continues to --steps",
    )
    args = p.parse_args()

    summary = train(args)
    print("\n[toy] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

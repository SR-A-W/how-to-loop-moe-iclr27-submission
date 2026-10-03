"""Evaluate one HF-format trajectory checkpoint: LM-Eval-Harness zero-shot
tasks + FineWeb-Edu validation loss.

## Why a standalone script, not a training-loop hook

The LT2 training recipe (arXiv 2605.20670) sets the in-training eval interval far beyond the total
number of steps and disables probing ("OOM with looped architecture"). LT2
themselves run eval OUT of the training loop for the same cost reason
-- an eval pass is an expensive-tier operation (full LM-Eval-Harness corpus
downloads, thousands of forward passes) and does not belong sharing GPU
memory/time with the training step. This script is invoked once per
checkpoint, as a separate process/job -- never imported by the training loop.

## Why results land IN the checkpoint directory

Every evaluation number must point back to the weights that produced it. A results file living anywhere
else (a central eval-results database, a differently-named sibling directory)
can be copied, renamed, or separated from the weights that produced it with
no way to tell after the fact. Writing `eval_results.json` next to
`pytorch_model.bin` and `manifest.json` in the SAME directory makes that
question unaskable -- there is only one place results for a checkpoint can
live.

## Env

This script's dependencies (`lm_eval` and its many transitive deps: datasets,
accelerate, sqlitedict, rouge_score, ...) are NOT in `requirements-train.txt`
-- eval runs as a separate process, so it gets its own environment
(`requirements-eval.txt`) rather than widening the training environment's
dependency surface for something the training loop never imports.
`transformers`/`torch` versions in the two envs are not
required to match exactly -- this script loads the checkpoint purely through
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`, and the
checkpoint bundles its own modeling code (see
`src.pretrain.checkpointing.store._bundle_modeling_code`), so it does not import
anything from the `pretrain` training package except the two read-only
utilities below.

## Reuse, not reinvention

- `src.pretrain.checkpointing.store.is_complete` -- refuses to evaluate a
  checkpoint that never got a manifest (a half-written directory produces
  numbers for weights that do not durably exist).
- `src.data.packed_dataset.PackedSeqDataset` -- the FineWeb-Edu
  validation loss reads validation-split bin/idx shards through the SAME
  reader the training loop uses for training shards (imported, not
  reimplemented), just pointed at a validation-only
  `tokenized_dir` instead of a training one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.eval.tasks import LT2_ZERO_SHOT_TASKS, NUM_FEWSHOT
from src.metrics.provenance import code_revision

RESULTS_NAME = "eval_results.json"

WEIGHTS_FILENAME = "pytorch_model.bin"


def _read_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    """Validate the checkpoint is safe to evaluate, return its manifest dict.

    Two checks, deliberately distinct: `is_complete()`
    only guarantees the manifest itself is present and parses -- it is the
    completeness MARKER, not a content check on the payload it vouches for
    (`src.pretrain.checkpointing.store`'s own docs: "It does NOT validate the payload
    itself"). A manifest that parses fine can still sit next to a weights file
    that was truncated by something other than the save path this project
    controls (a bad `cp`, a disk-full mid-copy on a THIRD machine, ...). Checking
    `WEIGHTS_FILENAME` exists and is non-empty catches that class of failure
    at eval time with a clear message, instead of `from_pretrained` failing deep
    inside torch's own deserializer with a much less legible error.
    """
    from src.pretrain.checkpointing.store import MANIFEST_NAME, is_complete

    if not is_complete(checkpoint_dir):
        raise ValueError(
            f"{checkpoint_dir} has no valid manifest.json -- refusing to evaluate an "
            "incomplete/unverified checkpoint. is_complete() is the only sanctioned "
            "check (see src/pretrain/checkpointing/store.py); a directory with plausible-"
            "looking weight files but no manifest may be a half-written save."
        )
    weights_path = checkpoint_dir / WEIGHTS_FILENAME
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise ValueError(
            f"{checkpoint_dir} has a valid manifest but {WEIGHTS_FILENAME} is missing or "
            "empty -- is_complete() only checks the manifest marker, not the weights "
            "payload it vouches for (see src.pretrain.checkpointing.store's own module "
            "docstring). Refusing to hand a truncated/absent weights file to "
            "from_pretrained, which would otherwise fail with a much less legible error "
            "deep inside torch's deserializer."
        )
    with open(checkpoint_dir / MANIFEST_NAME) as fh:
        return json.load(fh)


def _generation_tasks(tasks: tuple[str, ...], task_manager: Any) -> set[str]:
    """Which of `tasks` lm_eval will run with `generate_until`.

    Asked of lm_eval rather than kept as a list here: a hand-maintained list of
    "the generation tasks" is a second source of truth that goes stale the first
    time a task is added, and the failure it produces -- a generation task
    silently run through the checkpoint's own model code -- looks like a normal
    result. If the task cannot be loaded (offline, missing dataset), it is
    reported as NOT a generation task and the caller keeps the old path: that is
    the behaviour every existing product was produced with.
    """
    import lm_eval.tasks as lm_tasks

    found: set[str] = set()

    def walk(obj: Any, name: str | None = None) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, key if isinstance(key, str) else name)
            return
        output_type = getattr(obj, "OUTPUT_TYPE", None) or getattr(
            getattr(obj, "config", None), "output_type", None
        )
        if output_type == "generate_until" and name:
            found.add(name)

    for task in tasks:
        try:
            walk(lm_tasks.get_task_dict([task], task_manager), task)
        except Exception as exc:  # noqa: BLE001 - see docstring
            print(f"[run_eval] could not inspect task {task!r} ({exc}); assuming log-likelihood", flush=True)
    return found


def _refuse_mixed_task_batch(tasks: tuple[str, ...], generation_tasks: set[str]) -> None:
    """Refuse a batch that mixes generation and log-likelihood tasks.

    The model handed to `lm_eval.simple_evaluate` is one object for the whole
    call, so the choice between the checkpoint's bundled model code and this
    repository's is per CALL, not per task. A mixed batch would therefore run
    the log-likelihood tasks through the repo code as well -- silently changing
    the code path that every delivered log-likelihood product was produced with,
    while the output still looked normal.

    Refusing is the cheap half of the fix: splitting the batch in two and
    merging results would also work, but nothing asks for a single call to carry
    both kinds, and a wrong-but-plausible number is worse than an error message.
    """
    if not generation_tasks:
        return
    loglikelihood = [t for t in tasks if t not in generation_tasks]
    if loglikelihood:
        raise ValueError(
            "refusing to evaluate generation and log-likelihood tasks in one call: "
            f"generation {sorted(generation_tasks)} vs log-likelihood {sorted(loglikelihood)}. "
            "Generation tasks need this repository's model code (transformers 5.15 "
            "requires config.num_hidden_layers), log-likelihood tasks keep the "
            "checkpoint's bundled code so existing products stay comparable, and "
            "lm_eval takes one model per call. Run them as two invocations."
        )


def run_lm_eval_harness(
    checkpoint_dir: Path,
    *,
    tasks: tuple[str, ...],
    num_fewshot: int,
    limit: int | float | None,
    device: str,
    batch_size: str | int,
    tokenizer: str,
    max_length: int,
    fewshot_seed: int | None = None,
    log_samples: bool = False,
    include_path: str | None = None,
    gen_kwargs: str | None = None,
) -> dict[str, Any]:
    """Run LM-Eval-Harness against the checkpoint, return its raw results dict.

    `max_length`: context window HFLM uses for truncation and for rolling
    log-likelihood (wikitext). Passed explicitly because HFLM only auto-detects
    `n_positions` / `max_position_embeddings` / `n_ctx` from the model config;
    our checkpoints record the window as `context_length`, which it does not
    read, so it silently took the tokenizer's `model_max_length` instead (8192
    for smollm2) -- longer than the model's 4096-position RoPE table, which
    made wikitext's rolling log-likelihood crash with an IndexError. Hence:
    required, no default, recorded in `eval_config`.

    `fewshot_seed`: forwarded as `fewshot_random_seed` (the seed lm_eval's
    per-task few-shot sampler draws demonstrations with; irrelevant at
    num_fewshot=0). None = lm_eval's own default (1234). `log_samples`: keep
    per-document requests/log-likelihoods in the returned dict (`samples`),
    which `main` writes next to the results. Both were added for the
    0/5-shot x 3-seed evaluation batch; defaults reproduce the
    pre-existing behaviour exactly.

    `include_path`: extra task-definition directory handed to lm_eval's
    `TaskManager` (our overrides, e.g. `scripts/eval/tasks` with
    `wsc273_nsmirror`, whose stock `wsc273` dataset id has no namespace and
    fails to load under current huggingface_hub). None = stock tasks only.

    `gen_kwargs`: lm_eval's own comma-separated override string for generative
    (`generate_until`) tasks, e.g. "max_gen_toks=64". Forwarded verbatim; None
    leaves each task's yaml settings untouched. Added for the
    generative task group, where a 256-token cap on a model that generates
    without a KV cache can cost more than the eval is worth.

    `limit`: forwarded verbatim to `lm_eval.simple_evaluate` (None = full eval;
    an int/float = only that many examples per task, e.g. `limit=8` for a fast
    toy/CI check). No default -- callers must decide, since a script that
    silently ran full-corpus eval when a smoke test was intended would burn
    hours of the wrong thing, and a script that silently capped a "real" run at
    a small limit would ship an underpowered number as if it were the real one.

    `tokenizer`: a repo id or local directory `AutoTokenizer.from_pretrained`
    accepts. Passed explicitly rather than left for `HFLM` to guess from
    `checkpoint_dir` -- `save_trajectory` does NOT bundle tokenizer files (only
    the modeling source + weights + config, see
    `src.pretrain.checkpointing.store._bundle_modeling_code`), so an unset
    tokenizer here would hard-fail on every real checkpoint, not just this
    toy test. `main()` resolves this from its required `--tokenizer-name`
    argument via `src.data.tokenizer.load_named_tokenizer` -- no default here
    or at the CLI (this docstring used to justify a silent default on "the project has
    exactly one tokenizer", which stopped being true once SmolLM2 was added
    alongside Llama-3; a silent default now risks evaluating a
    SmolLM2-tokenized checkpoint through the wrong vocabulary). The value
    actually used is recorded in the output JSON's `eval_config` (both the
    resolved path and the caller-facing name), so it is never a silent choice.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    task_manager = (
        __import__("lm_eval.tasks", fromlist=["TaskManager"]).TaskManager(include_path=include_path)
        if include_path is not None
        else None
    )

    # Generation tasks (`generate_until`) are run through the model code in THIS
    # repository, not the `modeling_loop_lm.py` bundled inside the checkpoint.
    # Why: `generate()` in transformers 5.15 builds a `DynamicCache(config)` and
    # reads `config.num_hidden_layers`, which the bundled copies predate -- every
    # delivered checkpoint would fail to generate, and re-uploading model code
    # into checkpoints that are already published is not an option (their sha256
    # is the identity the paper cites). The config still comes from the
    # checkpoint, so the architecture evaluated is the checkpoint's own; only the
    # Python implementing it is newer, and `modeling_source` in `eval_config`
    # records which. Log-likelihood tasks keep the previous path byte for byte,
    # so existing products stay comparable.
    generation_tasks = _generation_tasks(tasks, task_manager)
    _refuse_mixed_task_batch(tasks, generation_tasks)
    modeling_source = "checkpoint (bundled modeling_loop_lm.py)"
    pretrained: Any = str(checkpoint_dir)
    if generation_tasks:
        from src.metrics.provenance import code_revision
        from src.model.modeling_loop_lm import LoopMoEConfig, LoopMoEForCausalLM

        repo_root = Path(__file__).resolve().parents[2]
        revision = code_revision(str(repo_root))
        cfg = LoopMoEConfig.from_pretrained(str(checkpoint_dir))
        model = LoopMoEForCausalLM.from_pretrained(
            str(checkpoint_dir), config=cfg, dtype=torch.bfloat16
        )
        # Moved here, not left to lm_eval. `HFLM.__init__` handles a
        # pre-instantiated model with `self._device = self._model.device` and
        # never calls `.to()`: the `--device` argument is silently ignored on
        # this branch (verified in lm_eval 0.4.12's source). Without this line a
        # `--device cuda` run would quietly evaluate on CPU -- correct numbers,
        # hours instead of minutes, and nothing in the output saying why.
        model = model.to(device).eval()
        pretrained = model
        modeling_source = f"repo@{revision['commit']}" + ("+dirty" if revision["dirty"] else "")
        print(
            f"[run_eval] generation tasks {sorted(generation_tasks)} -> repo model code "
            f"({modeling_source}) on {device}; checkpoint config unchanged",
            flush=True,
        )

    lm = HFLM(
        pretrained=pretrained,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device=device,
        batch_size=batch_size,
        dtype="bfloat16",  # matches save_trajectory's storage dtype -- no silent upcast/downcast
        max_length=max_length,
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=list(tasks),
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=log_samples,
        **({"gen_kwargs": gen_kwargs} if gen_kwargs is not None else {}),
        **({"fewshot_random_seed": fewshot_seed} if fewshot_seed is not None else {}),
        **({"task_manager": task_manager} if task_manager is not None else {}),
    )
    results["_modeling_source"] = modeling_source
    # `results["results"]` is what actually has the per-task metrics; the rest
    # of the top-level dict (`configs`, `versions`, ...) is large and mostly
    # useful for exact-reproduction, not for reading -- kept, not trimmed,
    # since "smaller JSON" is not a goal this project has ever stated and a
    # trimmed field is a field someone eventually needed and didn't have.
    return results


def compute_fineweb_edu_val_loss(
    tokenized_dir: str,
    model: Any,
    *,
    seq_len: int,
    num_sequences: int | None,
    device: str,
) -> dict[str, Any]:
    """Mean cross-entropy loss + perplexity over the FineWeb-Edu validation
    split, reusing the training data's `PackedSeqDataset` reader
    (same bin/idx format, sha256-verified shards) rather than a second,
    eval-specific data-reading implementation.

    `num_sequences`: how many packed sequences to average over. `None` means
    every sequence in the validation corpus (the "real" number); an int caps
    it (smoke/toy runs). Sequences are taken in canonical (seed-independent)
    corpus order starting at index 0 -- deterministic, not sampled, so two
    runs against the same `tokenized_dir` and `num_sequences` always average
    over the identical set.
    """
    from src.data.packed_dataset import PackedSeqDataset

    # seed is irrelevant here: PackedSeqDataset's `seed` only controls the
    # per-epoch SHUFFLE order used for training; get_sequence() addresses
    # packed sequences directly in canonical corpus order regardless of seed,
    # which is exactly the deterministic-iteration-order guarantee eval needs.
    ds = PackedSeqDataset(tokenized_dir, seq_len=seq_len, seed=0)
    n = ds.num_sequences if num_sequences is None else min(num_sequences, ds.num_sequences)
    if n == 0:
        raise ValueError(f"{tokenized_dir} has 0 usable sequences at seq_len={seq_len}")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(n):
            ids = torch.from_numpy(ds.get_sequence(i)).long().unsqueeze(0).to(device)
            # `labels=ids` (unshifted) was wrong -- LoopMoEForCausalLM.forward()
            # does not shift internally (src/model/loop_trace.py:320's
            # documented contract: the caller must supply pre-shifted labels).
            # Unshifted, this computed CE(logits[t], token[t]) -- a trivial
            # copy target visible under causal attention -- instead of the
            # intended next-token validation loss (the same root cause as an
            # earlier probe collapse; this eval path was never exercised then
            # since it was skipped for lack of a val dir, but the same bug was
            # present here).
            labels = torch.full_like(ids, -100)
            labels[:, :-1] = ids[:, 1:]
            out = model(input_ids=ids, labels=labels)
            # sum, not mean, so sequences of different effective length (the
            # last shard's tail could differ) are weighted by token count
            # rather than by sequence count -- matches the LM-loss convention
            # the rest of this project already reports (collector.py's
            # repeat_stratified_loss_records has the identical rationale).
            n_targets = ids.numel() - 1  # next-token targets, one fewer than input
            # `out.task_loss`, not `out.loss` -- `out.loss` is task_loss plus the
            # weighted lb_loss/z_loss aux terms. A pure LM validation loss/ppl
            # must not include those -- they'd contaminate the reported
            # perplexity (exp() amplifies the distortion) the next time this
            # path actually runs with a real --val-tokenized-dir given. No
            # historical numbers were affected: every earlier eval_results.json
            # had `fineweb_edu_val: {"skipped": true}` (all 92 checked).
            total_loss += out.task_loss.item() * n_targets
            total_tokens += n_targets

    mean_loss = total_loss / total_tokens
    return {
        "tokenized_dir": tokenized_dir,
        "seq_len": seq_len,
        "n_sequences": n,
        "n_sequences_available": ds.num_sequences,
        "n_tokens": total_tokens,
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True, help="Trajectory checkpoint directory (HF format)")
    ap.add_argument(
        "--out", default=None,
        help=f"Output JSON path (default: <checkpoint-dir>/{RESULTS_NAME})",
    )
    ap.add_argument(
        "--tasks", default=None,
        help="Comma-separated lm-eval task names (default: the 11 tasks of the LT2 paper, see tasks.py)",
    )
    ap.add_argument(
        "--limit", type=float, default=None,
        help="lm-eval per-task example limit (unset = full eval; use e.g. 8 for a smoke test)",
    )
    ap.add_argument(
        "--val-tokenized-dir", default=None,
        help="FineWeb-Edu validation bin/idx shard directory (unset = skip val-loss section, "
             "recorded explicitly as skipped, not silently omitted)",
    )
    ap.add_argument("--val-seq-len", type=int, default=None, help="Required if --val-tokenized-dir is given")
    ap.add_argument(
        "--val-num-sequences", type=int, default=None,
        help="Cap on validation sequences (unset = every sequence in the corpus)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--num-fewshot", type=int, default=NUM_FEWSHOT,
                    help=f"Few-shot demonstrations per task (default {NUM_FEWSHOT} = the zero-shot setting all products so far used)")
    ap.add_argument("--fewshot-seed", type=int, default=None,
                    help="Seed for lm_eval's few-shot sampler (fewshot_random_seed); default None = lm_eval's own 1234. Ignored at 0-shot.")
    ap.add_argument("--include-path", default=None,
                    help="Extra lm_eval task-definition directory (task overrides such as scripts/eval/tasks)")
    ap.add_argument("--gen-kwargs", default=None,
                    help="lm_eval gen_kwargs override for generative tasks, comma-separated (e.g. max_gen_toks=64); default: each task's yaml")
    ap.add_argument("--no-lm-eval", action="store_true",
                    help="Skip the lm-eval harness entirely (held-out validation loss only; requires --val-tokenized-dir)")
    ap.add_argument("--log-samples", action="store_true",
                    help="Also write per-document samples (requests + log-likelihoods) to <out stem>_samples.json")
    ap.add_argument(
        "--max-length", type=int, required=True,
        help="Context window for lm_eval's HFLM (truncation + rolling loglikelihood). Required, no default: "
             "the model config's `context_length` is not read by HFLM, which would otherwise silently use the "
             "tokenizer's model_max_length (8192 for smollm2, beyond the model's 4096-position RoPE table).",
    )
    from src.data.tokenizer import TOKENIZER_CLASSES

    ap.add_argument(
        "--tokenizer-name", dest="tokenizer_name", choices=sorted(TOKENIZER_CLASSES), required=True,
        help="Which tokenizer this checkpoint expects -- see src.data.tokenizer.TOKENIZER_CLASSES "
             "(trajectory checkpoints do not bundle tokenizer files, only modeling code + weights, "
             "so this cannot be inferred from the checkpoint itself). No default: "
             "this project has more than one "
             "tokenizer, so a silent default here would risk evaluating a SmolLM2-tokenized "
             "checkpoint through the wrong vocabulary.",
    )
    args = ap.parse_args(argv)
    if args.out:
        # Create it now: the results are written only after every task has run, and a
        # missing directory at that point would throw away the whole evaluation.
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    manifest = _read_manifest(checkpoint_dir)

    # Read BEFORE the (expensive, possibly network-bound) eval run, not after:
    # `code_revision()` raises if git is unusable, e.g. a transient lock from a
    # concurrent `git add`/`commit` in a shared working tree. Calling it only after
    # `run_lm_eval_harness` returns would mean that failure -- entirely
    # unrelated to whether eval itself succeeded -- discards a completed
    # eval run's results instead of failing fast for near-zero cost first.
    eval_script_revision = code_revision()

    tasks = tuple(t.strip() for t in args.tasks.split(",")) if args.tasks else LT2_ZERO_SHOT_TASKS

    from src.data.tokenizer import load_named_tokenizer

    tokenizer = load_named_tokenizer(args.tokenizer_name).repo_id

    print(f"[run_eval] checkpoint: {checkpoint_dir} (step={manifest.get('step')}, kind={manifest.get('kind')})")
    print(f"[run_eval] tokenizer: {args.tokenizer_name} ({tokenizer})")
    print(f"[run_eval] lm-eval tasks: {tasks} (num_fewshot={args.num_fewshot}, fewshot_seed={args.fewshot_seed}, limit={args.limit}, log_samples={args.log_samples})")

    if args.no_lm_eval:
        if args.val_tokenized_dir is None:
            raise ValueError("--no-lm-eval without --val-tokenized-dir would evaluate nothing")
        lm_eval_results = {"skipped": True, "reason": "--no-lm-eval"}
    else:
        lm_eval_results = run_lm_eval_harness(
            checkpoint_dir,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            device=args.device,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            max_length=args.max_length,
            fewshot_seed=args.fewshot_seed,
            log_samples=args.log_samples,
            include_path=args.include_path,
            gen_kwargs=args.gen_kwargs,
        )

    if args.val_tokenized_dir is not None:
        if args.val_seq_len is None:
            raise ValueError("--val-seq-len is required when --val-tokenized-dir is given")
        from transformers import AutoModelForCausalLM

        print(f"[run_eval] FineWeb-Edu val loss: {args.val_tokenized_dir} (seq_len={args.val_seq_len})")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(args.device)
        fineweb_val = compute_fineweb_edu_val_loss(
            args.val_tokenized_dir,
            model,
            seq_len=args.val_seq_len,
            num_sequences=args.val_num_sequences,
            device=args.device,
        )
    else:
        fineweb_val = {"skipped": True, "reason": "--val-tokenized-dir not given"}

    # Identity of the weights actually evaluated, in the product itself: a
    # results.json must be matchable against the published weights' index without trusting the directory name.
    h = hashlib.sha256()
    with open(checkpoint_dir / WEIGHTS_FILENAME, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    checkpoint_bin_sha256 = h.hexdigest()

    output = {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_bin_sha256": checkpoint_bin_sha256,
        "checkpoint_manifest": manifest,
        "eval_config": {
            "tasks": [] if args.no_lm_eval else list(tasks),
            "num_fewshot": args.num_fewshot,
            "fewshot_seed": args.fewshot_seed,
            "log_samples": args.log_samples,
            "include_path": args.include_path,
            "gen_kwargs": args.gen_kwargs,
            # Which Python implemented the architecture during this run: the
            # checkpoint's bundled copy (log-likelihood tasks, unchanged) or this
            # repository at a stated commit (generation tasks; see
            # run_lm_eval_harness). Recorded so a result is never ambiguous about
            # the code that produced it.
            # `pop`, not `get`: `output["lm_eval"]` is the same object, so a
            # leftover private key would ship inside every results.json as a
            # second, duplicate copy of this fact.
            "modeling_source": (lm_eval_results or {}).pop("_modeling_source", "checkpoint (bundled modeling_loop_lm.py)"),
            "no_lm_eval": args.no_lm_eval,
            "limit": args.limit,
            "tokenizer": tokenizer,
            "tokenizer_name": args.tokenizer_name,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        "lm_eval": lm_eval_results,
        "fineweb_edu_val": fineweb_val,
        "eval_provenance": {
            "eval_script_git_commit": eval_script_revision["commit"],
            "eval_script_code_dirty": eval_script_revision["dirty"],
            "eval_script_code_diff_sha256": eval_script_revision["diff_sha256"],
            "lm_eval_version": __import__("lm_eval").__version__,
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    out_path = Path(args.out) if args.out else checkpoint_dir / RESULTS_NAME
    samples = lm_eval_results.pop("samples", None)  # per-document records: separate file, never inside results
    if args.log_samples:
        if samples is None:
            raise ValueError("--log-samples given but lm_eval returned no 'samples' -- refusing to write a results file that claims samples were logged")
        samples_path = out_path.with_name(out_path.stem + "_samples.json")
        tmp_samples = samples_path.with_name(samples_path.name + ".tmp")
        tmp_samples.write_text(json.dumps(samples, default=str))
        tmp_samples.replace(samples_path)
        output["eval_config"]["samples_file"] = str(samples_path)
        print(f"[run_eval] wrote {samples_path}")
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    tmp_path.replace(out_path)  # atomic -- a reader never sees a truncated results file
    print(f"[run_eval] wrote {out_path}")


if __name__ == "__main__":
    main()

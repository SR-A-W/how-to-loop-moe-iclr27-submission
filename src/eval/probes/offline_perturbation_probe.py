"""Offline perturbation-growth driver: error-amplification / contraction rate for
a saved looped-MoE checkpoint.

## Provenance

The core measurement below -- inject a scaled random perturbation into the
loop-carried state at loop step `k0`, diff against a clean run at every
later step, report `r_k`/`g_k` and a geometric-mean growth `rate` -- is the
SAME definition as an earlier, separate diagnostics code base's
`verify/probe_perturbation_growth.py` (read-only reference; that script's own module
docstring is the fuller account of WHY this measurement answers the "does
the loop amplify or wash out error" question, and its "DETERMINISM
PRECONDITIONS"/"clean-vs-clean must give r==0" discipline, both reproduced
here). Per `_vendored_loopdiag`'s provenance-preserving
convention (`moe.py`/`looped.py` each carry a header naming their
source path), the formula/sanity-check logic here is a direct re-typing of that
script's math against this project's own tensors and checkpoint-loading
(NOT copy-pasted -- the reference script's CLI is built around
`--family looplm/raven/ouro` and an external `corpus_batch` helper this
project does not have, and supports a `--granularity layer` mode this
project has no use for, see below). The ONE genuinely new piece of code is
`_DVFLoopTap`, adapting the reference's "FAMILY HOOK POINTS" section to
`LoopMoEForCausalLM` -- a family the reference script never had, since it
predates this project's own architecture.

`_DVFLoopTap`'s hook point is `model.model.stack` (a `src.model.
modeling_loop_lm.LoopedStack` instance), read off `LoopedMoETransformer.
forward`'s `for loop_step in range(self.num_stacks): x, lb, lz =
self.stack(x, loop_step=loop_step, trace_sink=trace_sink)` -- `x` is the
loop-carried state, passed positionally, so a plain (non-kwargs)
`register_forward_pre_hook` sees it as `args[0]` and can replace it without
touching the `loop_step=`/`trace_sink=` keyword arguments at all (PyTorch's
old-style forward-pre-hook only ever sees the positional-args tuple). This
is structurally IDENTICAL to the reference script's `looplm` family
(`model.model.stack` called once per loop step, state passed positionally)
-- the naming happening to match is a coincidence of a shared code
lineage (`modeling_loop_lm.py`'s own docstring: ported from the same
seedvar `loop-lm` architecture the reference repo's `looplm` family
diagnoses), not something this module assumes without having read both
`forward` methods side by side.

## Why only "step" granularity, not also "layer"

The reference script's `--granularity layer` mode exists to compare looped
vs non-looped quadrants of seedvar's 2x2 factorial (see that script's
`LayerTap` docstring) -- this project has no non-looped quadrant to compare
against here, so that mode is not ported: this measurement does not cover
seedvar's non-looped quadrant comparison, so the granularity is fixed to step.

## Determinism precondition, verified by reading (not assumed)

This model has no fresh-noise initial state (unlike Raven/Huginn, whose `h_0` is
resampled every forward) and no router jitter (`Router`'s own docstring:
"There is no jitter noise and no temperature; routing is deterministic
given the input") -- so, unlike the reference script's Raven branch, there
is no known non-determinism source to guard against beyond generic
kernel-level nondeterminism. The mandatory clean-vs-clean check below is
still NOT skipped on the strength of that reading -- it is the empirical
enforcement of exactly this claim, and it may not be skipped.

## Parameters are fixed constants, not CLI flags

Same rationale as `offline_trajectory_probe.py`'s `PROBE_BATCH`/`PROBE_SEQ`:
the three-point comparison (base checkpoint and two later training stages) requires every
measurement to use identical parameters, so `eps`/`k0`/`repeats`/`seed`
below are fixed module constants, not `argparse` options a caller could vary between
invocations. The reference script requires all of these as CLI flags with
NO default (its own module docstring: "Every parameter below decides WHAT
is measured, so none has a default") -- this module keeps that "no silent
default" discipline by fixing them once, here, with each value's rationale
attached, rather than by exposing a flag whose default could silently
drift between the three invocations that must agree.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.data.probe_corpus import PROBE_BATCH, PROBE_SEQ, WEIGHTS_FILENAME
from src.data.probe_corpus import DEFAULT_CORPUS_PATH, build_probe_batch, RULE_POSITION_0
from src.eval.probes.probe_dtype import add_probe_dtype_argument, dtype_fields, load_kwargs
from src.eval.probes.artifact_paths import assert_no_account_paths, corpus_block, named_checkpoint

PERTURBATION_EPS = 0.01
"""1% relative perturbation, the value fixed in the measurement plan before
any run (not tuned here)."""

PERTURBATION_K0 = 0
"""Inject at the first loop step: the model's num_stacks (R) is a
single-digit count that varies a lot between dvf_a (R=4) and dvf_c (R=16),
so k0=0 gives the longest observable trajectory in every case."""

PERTURBATION_REPEATS = 8
"""Independent random perturbation directions -- at least 8 are required; a
single direction is one draw, not a property of the model."""

PERTURBATION_SEED = 0
"""Fixed seed for the perturbation-direction generator (a dedicated
`torch.Generator`, never the global RNG stream -- see `_measure_perturbation_
growth` below). The exact value does not matter for validity (any fixed
seed gives 8 independent-enough directions); what matters is that all three
comparison points (the base checkpoint and two later training stages) use the
SAME seed, which a hardcoded
constant guarantees and a CLI default could silently fail to."""


class _DVFLoopTap:
    """Reads (and optionally perturbs) `LoopedStack`'s loop-carried state at
    the start of every loop-step call. See module docstring for why
    `model.model.stack` is the correct, read-off-the-forward-method hook
    point rather than an inferred/guessed one."""

    def __init__(self, model: Any) -> None:
        self.module = model.model.stack
        self.calls = 0
        self.states: list[Tensor] = []
        self.inject_step: int | None = None
        self.delta: Tensor | None = None
        self._handle: Any = None

    def pre_hook(self, mod: Any, args: tuple) -> tuple:
        x = args[0]
        if self.inject_step is not None and self.calls == self.inject_step:
            x = x + self.delta
            args = (x,) + args[1:]
        # Detached clone, same lifetime discipline as src.model.loop_trace's
        # TracePoint contract -- these lists are read back within this same
        # function call and never retained past it.
        self.states.append(x.detach().clone())
        self.calls += 1
        return args

    def __enter__(self) -> "_DVFLoopTap":
        self._handle = self.module.register_forward_pre_hook(self.pre_hook)
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.remove()


def _rel_diff(a: Tensor, b: Tensor) -> float:
    """`|| a - b ||_F / || b ||_F` -- the `r_k` definition."""
    return (a - b).norm().item() / b.norm().item()


def _run_dvf_forward(
    model: Any, input_ids: Tensor, *, inject_step: int | None = None, delta: Tensor | None = None
) -> list[Tensor]:
    """One `no_grad` forward, tapped at every loop step. Returns the R
    loop-carried states observed (post-injection at `inject_step`, if given).
    Caller is responsible for `model.eval()`/mode restoration (see
    `measure_perturbation_growth`, which owns that lifecycle since it calls
    this function many times per measurement)."""
    tap = _DVFLoopTap(model)
    tap.inject_step, tap.delta = inject_step, delta
    with tap, torch.no_grad():
        model(input_ids)
    return tap.states


def measure_perturbation_growth(
    model: Any,
    input_ids: Tensor,
    *,
    eps: float,
    k0: int,
    last_step: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """The full perturbation-growth measurement, gated on the
    mandatory clean-vs-clean sanity check. Raises (does not silently report
    numbers) if that check fails, or if no repeat produced a usable rate.
    """
    was_training = model.training
    try:
        model.eval()

        clean = _run_dvf_forward(model, input_ids)
        clean2 = _run_dvf_forward(model, input_ids)
        control = [_rel_diff(a, b) for a, b in zip(clean2, clean)]
        if max(control) != 0.0:
            raise RuntimeError(
                f"clean-vs-clean control is not bitwise identical (max relative "
                f"diff {max(control):.3e} across {len(control)} loop steps) -- the "
                "forward is not reproducible under these settings, so a "
                "perturbation trajectory cannot be separated from the model's own "
                "nondeterminism. Refusing to report numbers (the clean-vs-clean "
                "sanity check is mandatory)."
            )

        R = len(clean)
        if not (0 <= k0 < R):
            raise ValueError(f"k0={k0} out of range for R={R} loop steps")
        if not (k0 <= last_step < R):
            raise ValueError(f"last_step={last_step} out of range for R={R} loop steps (k0={k0})")

        base_norm = clean[k0].norm().item()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        trajectories: list[list[float]] = []
        for _ in range(repeats):
            # Direction drawn on CPU with an explicit generator -- never the
            # global RNG stream -- then moved to the clean state's own
            # device/dtype (same rationale as the reference script: a
            # device-side generator would silently make cpu/cuda runs
            # different experiments while both looked correct).
            d = torch.randn(clean[k0].shape, generator=gen).to(dtype=clean[k0].dtype, device=clean[k0].device)
            d = d / d.norm() * (eps * base_norm)
            pert = _run_dvf_forward(model, input_ids, inject_step=k0, delta=d)
            traj = [_rel_diff(p, c) for p, c in zip(pert[k0 : last_step + 1], clean[k0 : last_step + 1])]
            trajectories.append(traj)

        n_points = last_step - k0 + 1
        r_by_step: dict[str, dict[str, float]] = {}
        for i in range(n_points):
            col = [t[i] for t in trajectories]
            r_by_step[str(k0 + i)] = {"min": min(col), "median": statistics.median(col), "max": max(col)}

        rates = [
            (t[-1] / t[0]) ** (1.0 / (len(t) - 1))
            for t in trajectories
            if len(t) > 1 and t[0] > 0
        ]
        if not rates:
            raise RuntimeError(
                "no repeat produced a usable rate (either every trajectory has a "
                "single point, i.e. k0 == last_step, or every t[0] was 0) -- cannot "
                "report a geometric-mean rate. Not expected with k0=0/last_step=R-1 "
                "and eps>0; if this fires, the injection is not reaching the traced "
                "state (check the hook point)."
            )
        rate = {"min": min(rates), "median": statistics.median(rates), "max": max(rates)}

        return {
            "kind": "perturbation_probe",
            "eps": eps,
            "k0": k0,
            "last_step": last_step,
            "repeats": repeats,
            "clean_control_max_rel_diff": max(control),
            "r_by_step": r_by_step,
            "rate": rate,
        }
    finally:
        model.train(was_training)


def compute_perturbation_probe(
    model: Any,
    *,
    tokenizer: Any,
    device: str,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
) -> dict[str, Any]:
    """Ties the fixed probe corpus (same `PROBE_BATCH`/`PROBE_SEQ` as
    `offline_trajectory_probe.py`) to `measure_perturbation_growth`, using
    `model.config.num_stacks` for `last_step = R - 1` rather than
    a caller-supplied value -- `R` differs between `dvf_a` (4) and `dvf_c`
    (16), and reading it off the loaded checkpoint's own config is the only
    way to avoid a caller silently passing the wrong architecture's R."""
    input_ids, token_source = build_probe_batch(tokenizer, corpus_path, batch=PROBE_BATCH, seq=PROBE_SEQ, exclusion_rule=RULE_POSITION_0)
    input_ids = input_ids.to(device)
    last_step = model.config.num_stacks - 1

    result = measure_perturbation_growth(
        model, input_ids,
        eps=PERTURBATION_EPS, k0=PERTURBATION_K0, last_step=last_step,
        repeats=PERTURBATION_REPEATS, seed=PERTURBATION_SEED,
    )
    result["input_ids_sha8"] = token_source["input_ids_sha8"]
    return {"perturbation_probe": result, "token_source": token_source}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True, help="Trajectory checkpoint directory (HF format)")
    ap.add_argument("--out", required=True, help="Output JSONL path -- one line, kind=perturbation_probe")
    ap.add_argument(
        "--tokenizer", required=True,
        help="Tokenizer repo id or local directory (required: trajectory checkpoints do "
             "not bundle tokenizer files; the bundled SmolLM2 tokenizer is "
             "src/data/artifacts/tokenizer/smollm2).",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_probe_dtype_argument(ap)
    args = ap.parse_args(argv)

    from src.pretrain.checkpointing.store import MANIFEST_NAME, is_complete
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not is_complete(checkpoint_dir):
        raise ValueError(
            f"{checkpoint_dir} has no valid manifest.json -- refusing to probe an "
            "incomplete/unverified checkpoint (same check as src/eval/run_eval.py's "
            "_read_manifest)."
        )
    weights_path = checkpoint_dir / WEIGHTS_FILENAME
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise ValueError(f"{checkpoint_dir} has a manifest but {WEIGHTS_FILENAME} is missing/empty.")
    with open(checkpoint_dir / MANIFEST_NAME) as fh:
        manifest = json.load(fh)

    tokenizer_repo = args.tokenizer

    print(f"[offline_perturbation_probe] checkpoint: {checkpoint_dir} "
          f"(step={manifest.get('step')}, kind={manifest.get('kind')})")
    print(f"[offline_perturbation_probe] eps={PERTURBATION_EPS} k0={PERTURBATION_K0} "
          f"repeats={PERTURBATION_REPEATS} seed={PERTURBATION_SEED} (fixed, not flags)")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, trust_remote_code=True, **load_kwargs(args.probe_dtype)
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

    result = compute_perturbation_probe(model, tokenizer=tokenizer, device=args.device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # This driver recorded NO precision field at all until now, and loaded
    # bfloat16 unconditionally -- so every artifact it produced before this
    # change is bf16 and does not say so. New rows carry all three names.
    row = {
        "corpus": corpus_block(result["token_source"]),
        **named_checkpoint(checkpoint_dir),
        **dtype_fields(model, args.probe_dtype),
        **result["perturbation_probe"],
    }
    # Checked BEFORE the file is opened: a guard that fires after `open("w")`
    # has already truncated the previous artifact, turning a refusal to write
    # a bad row into the loss of a good one.
    assert_no_account_paths(row, what=f"{out_path}")
    with open(out_path, "w") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[offline_perturbation_probe] rate median={row['rate']['median']:.4f} "
          f"[{row['rate']['min']:.4f}, {row['rate']['max']:.4f}] -- wrote {out_path}")


if __name__ == "__main__":
    main()

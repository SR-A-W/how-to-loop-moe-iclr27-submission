"""The probe forward precision: one required flag, shared by every probe driver.

## Why it is required and has no default

Precision is a SEMANTIC parameter here, not a performance knob. The criteria
run two precision lines that answer different questions -- float32 for every
judgement, each model's native precision for deployment-characteristic
reporting only -- and a reading is meaningless until you know which line it
came from. A default would let two invocations of the same command produce
non-comparable numbers with nothing in the artifact saying so.

This replaces a hardcoded `FORWARD_DTYPE = torch.float32`, which was itself a
fix for an inherited `torch_dtype=bfloat16`. Hardcoding was right while there
was exactly one correct answer; now there are two, and the caller has to say
which.

## Why the resolved value is checked against the request

`torch_dtype` is a REQUEST. A checkpoint can load in a different dtype than
asked -- and the failure is silent: the artifact would record what was asked
while the forward ran in something else. So the drivers record what the loaded
model actually reports, and `check_matches_request` raises when the two differ,
naming both. `native` is the one case where no request was made, so nothing is
checked -- the point of `native` is to accept whatever the weights are.
"""

from __future__ import annotations

from typing import Any

import torch

# `native` maps to None: pass no `torch_dtype` at all and take the weights as
# they are. It is NOT `torch.float32` under another name -- most checkpoints
# here are bfloat16 on disk.
PROBE_DTYPES: dict[str, torch.dtype | None] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "native": None,
}


def add_probe_dtype_argument(parser: Any) -> None:
    """Add `--probe-dtype` as a REQUIRED argument. No default, deliberately."""
    parser.add_argument(
        "--probe-dtype",
        required=True,
        choices=sorted(PROBE_DTYPES),
        help=(
            "Forward precision for this probe run. Required: 'float32' is the "
            "judgement line, 'native' the reporting line, 'bfloat16' forces the "
            "old behaviour. Recorded in the artifact as `probe_dtype`."
        ),
    )


def load_kwargs(probe_dtype: str) -> dict[str, Any]:
    """Model-loading keyword arguments for this precision (empty for `native`)."""
    dtype = PROBE_DTYPES[probe_dtype]
    return {} if dtype is None else {"torch_dtype": dtype}


def check_matches_request(model: Any, probe_dtype: str) -> str:
    """Return the model's ACTUAL parameter dtype, raising if it ignored the request.

    Returns the string that goes into the artifact -- always the observed value,
    never the requested one, so the artifact describes what ran.
    """
    actual = next(model.parameters()).dtype
    requested = PROBE_DTYPES[probe_dtype]
    if requested is not None and actual != requested:
        raise RuntimeError(
            f"--probe-dtype {probe_dtype} asked for {requested}, but the loaded "
            f"model's parameters are {actual}. The forward would run at a "
            "precision the artifact does not name; refusing rather than "
            "recording a precision that was not used."
        )
    return str(actual)


def dtype_fields(model: Any, probe_dtype: str) -> dict[str, str]:
    """The precision fields every probe artifact carries.

    `probe_dtype` is the current name; `forward_dtype` is the older
    name kept for consumers already reading it, and carries the SAME value.
    `probe_dtype_requested` keeps the flag as typed, so `native` runs can still
    be told apart from an explicit `float32` run on float32 weights -- the two
    produce identical readings but answer different questions.
    """
    actual = check_matches_request(model, probe_dtype)
    return {
        "probe_dtype": actual,
        "forward_dtype": actual,  # legacy name, same value, to be dropped later
        "probe_dtype_requested": probe_dtype,
    }

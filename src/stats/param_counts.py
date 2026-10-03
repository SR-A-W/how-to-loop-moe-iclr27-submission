"""Exact parameter counts for the 37 ablation-family configurations.

One table covering: the 28 entries in
`config_registry.ABLATIONS` (26 original + L4/L5, the large-tier
counterparts of S3/S4), U1-U4 and UL1-UL4 (the
per-pass-attention variants of S1-S4 and, at the large tier, L1-L2 and L4-L5
-- registration only), and S1_seed43 (S1's shape
trained under a different seed -- not a distinct config, so its row is S1's
numbers under a different label).

Counting method
----------------
Each config is built into a `LoopMoEForCausalLM` on `torch.device("meta")` --
no weights are materialised, only shapes, so this is cheap and exact (unlike
deriving totals from the (E, L, R, head, tail) tuple by hand, which would have
to re-derive every architectural formula -- expert FFN width is `d_ff //
num_active`, per-pass attention multiplies the loop block's attention sets by
R, etc. -- and re-litigate it here).

`named_parameters()` is then bucketed by name prefix into four columns that
add up to the total (asserted per configuration in `count_params`):

  * ``attn_loop``    -- attention inside the shared stack (`model.stack.layers.*.attn*`;
                        for U1-U4 this already sums all R independent sets, since
                        `named_parameters` walks the whole `ModuleList`).
  * ``experts_loop`` -- routers + expert FFNs inside the shared stack
                        (`model.stack.layers.*.router.*` / `*.experts_w{1,2,3}`).
  * ``headtail``      -- everything in `model.head_layers.*` and `model.tail_layers.*`
                        (attention and experts together -- these layers run once,
                        so splitting them the way the loop block is split would
                        add columns nobody asked for).
  * ``other``          -- token embeddings + final norm (no params -- RMSNorm here
                        has no gain) + LM head.

Run: ``python -m src.stats.param_counts`` (writes the CSV; also importable via
`count_params`/`all_rows` for tests).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from src.model.config_registry import ABLATIONS, ATTENTION_VARIANTS, build_ablation
from src.model.modeling_loop_lm import LoopMoEForCausalLM

#: Rows that reuse an existing shape under a different label. S1_seed43 is S1's
#: (E, L, R, head, tail) shape and S1's `per_pass_attention=False` -- only the
#: training seed differs, which has no effect on parameter count. Not folded
#: into `ABLATIONS`/`ATTENTION_VARIANTS`: those two dicts are what
#: `build_ablation` and the 26/4-count tests key off, and adding a seed variant
#: there would inflate counts that are pinned elsewhere.
SEED_VARIANT_ROWS: dict[str, str] = {"S1_seed43": "S1"}

#: All 37 row names, in a fixed, deterministic order: the 28 ablations (dict
#: insertion order, i.e. the recipe's own grouping), then U1-U4/UL1-UL4, then
#: S1_seed43.
ROW_NAMES: list[str] = [*ABLATIONS, *ATTENTION_VARIANTS, *SEED_VARIANT_ROWS]

CSV_COLUMNS = [
    "config", "E", "L", "R", "head", "tail", "per_pass_attention",
    "attn_loop", "experts_loop", "headtail", "other", "total",
]


def _shape_name_and_variant(row_name: str) -> tuple[str, bool]:
    """Resolve a row name to the (E, L, R, head, tail) shape it builds from,
    and whether it carries per-pass attention."""
    if row_name in SEED_VARIANT_ROWS:
        return SEED_VARIANT_ROWS[row_name], False
    if row_name in ATTENTION_VARIANTS:
        return ATTENTION_VARIANTS[row_name], True
    return row_name, False


def count_params(config_name: str) -> dict[str, int]:
    """Exact parameter breakdown for one row, built via `build_ablation`.

    `config_name` is any of `ABLATIONS`, `ATTENTION_VARIANTS` (U1-U4, UL1-UL4),
    or `SEED_VARIANT_ROWS` (S1_seed43) -- the same 37 names `ROW_NAMES` lists.
    Returns attn_loop/experts_loop/headtail/other/total; the first four sum to
    the fifth by construction (asserted here, not left to the caller).
    """
    shape_name, _ = _shape_name_and_variant(config_name)
    # Build via the shape name for S1_seed43 (build_ablation has no such key),
    # via the row name itself for everything else (ABLATIONS/ATTENTION_VARIANTS
    # keys are exactly what build_ablation accepts).
    build_name = shape_name if config_name in SEED_VARIANT_ROWS else config_name
    config = build_ablation(build_name)

    with torch.device("meta"):
        model = LoopMoEForCausalLM(config)

    totals = {"attn_loop": 0, "experts_loop": 0, "headtail": 0, "other": 0}
    stack_prefix = "model.stack.layers."
    for name, param in model.named_parameters():
        n = param.numel()
        if name.startswith("model.head_layers.") or name.startswith("model.tail_layers."):
            totals["headtail"] += n
        elif name.startswith(stack_prefix):
            # "<layer_idx>.<field>...", where field is "attn" (shared or, under
            # per-pass attention, a ModuleList so the next segment is the pass
            # index -- either way "attn" is parts[1]), "router", or
            # "experts_w{1,2,3}".
            field = name[len(stack_prefix):].split(".", 2)[1]
            if field == "attn":
                totals["attn_loop"] += n
            elif field == "router" or field.startswith("experts_w"):
                totals["experts_loop"] += n
            else:
                raise ValueError(f"unrecognized loop-block parameter {name!r}")
        else:
            totals["other"] += n

    total = sum(param.numel() for param in model.parameters())
    parts_sum = sum(totals.values())
    assert parts_sum == total, (
        f"{config_name}: parts sum {parts_sum} != total {total} "
        "(attn_loop + experts_loop + headtail + other must equal total)"
    )
    totals["total"] = total
    return totals


def row_for(config_name: str) -> dict[str, Any]:
    """One CSV row: shape fields + the parameter breakdown."""
    shape_name, per_pass_attention = _shape_name_and_variant(config_name)
    experts, depth, loops, head, tail = ABLATIONS[shape_name]
    breakdown = count_params(config_name)
    return {
        "config": config_name,
        "E": experts,
        "L": depth,
        "R": loops,
        "head": head,
        "tail": tail,
        "per_pass_attention": per_pass_attention,
        **breakdown,
    }


def all_rows() -> list[dict[str, Any]]:
    return [row_for(name) for name in ROW_NAMES]


def write_csv(path: Path) -> list[dict[str, Any]]:
    rows = all_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=Path("stats/param_counts/looped_ablation_param_counts.csv"),
        help="output CSV path (default: stats/param_counts/looped_ablation_param_counts.csv)",
    )
    args = parser.parse_args()
    rows = write_csv(args.out)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

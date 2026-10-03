"""The 28 ablation config names -- zero non-stdlib imports, deliberately.

WHY THIS FILE EXISTS: `src.model.config_registry.ABLATIONS` is the single
authoritative definition of the ablation family (name -> (E, D, L, head,
tail)), but importing `config_registry` pulls in `modeling_loop_lm.py`,
which does `import torch` -- several seconds, and simply absent on a plain
status-reporting node. Anything that only needs to know WHICH configs
exist, never their shapes, imports this module instead of the real registry.
It is what a torch-free environment imports. `config_registry.py` imports FROM here (see its own `ABLATIONS`
definition and the assertion right after it) so there is exactly ONE place
these 28 names are typed -- never a second list that can silently drift
from the first.

This file must import under ANY Python, including a system Python 3.6, where
`from __future__ import annotations` -- itself only valid from 3.7 -- is a
SyntaxError. No
`from __future__` import, no PEP 585 builtin-generic subscription
(`tuple[str, ...]`, valid only from 3.9), no `typing` import either --
just a plain tuple literal.
"""

#: Kept in the recipe's own grouping order (small tier, large tier, pilot
#: tier) -- see config_registry.ABLATIONS for what each one actually is.
ABLATION_NAMES = (
    "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
    "S11", "S12", "S13", "S14", "S15",
    "L1", "L2", "L3", "L4", "L5",
    "P6", "P1", "P2", "P3",
    "N1", "N2", "N3", "N4",
)

#: The 100B continuation plans: each continues the like-named
#: ablation from its step-45,000 checkpoint. NOT ablation configurations --
#: they share the source's architecture, so they must never be added to
#: ABLATION_NAMES (config_registry and recipe count that tuple). Listed here so
#: torch-free tooling can recognise `<CONFIG>_100B` names. recipe.py asserts this matches its RUN_LIST.
CONTINUATION_NAMES = (
    "S1_100B", "S2_100B", "S3_100B", "S6_100B", "S14_100B",
    "L1_100B", "L2_100B", "L3_100B",
    # U1-U4 launched; S4 registered but held:
    "S4_100B", "U1_100B", "U2_100B", "U3_100B", "U4_100B",
    # UL1-UL2 continue straight after their 20B runs:
    "UL1_100B", "UL2_100B",
    # UL3-UL4 continue the same way:
    "UL3_100B", "UL4_100B",
)

#: Seed-variant plans: the same configuration and the same data
#: order, trained from a DIFFERENT initialisation. `S1_seed43` is `S1` with
#: `--debug.seed 43`; the data-order seed stays 42, so initialisation is the
#: only thing that differs and any difference in the result is attributable to
#: it. NOT ablation configurations (they share the source's architecture), and
#: NOT continuations (they train from step 0) -- a third list rather than a
#: flag on one of the other two, because torch-free tooling has to be able to
#: recognise the name without importing anything that knows what a seed is.
#: Extend the tuple to add further seeds.
SEED_NAMES = (
    "S1_seed43",
)

#: Attention variants: S1-S4's shapes again, with per-pass
#: attention instead of one shared set of attention weights across the R
#: passes. NOT ablation configurations (they share the source's (E,D,L,head,
#: tail) shape, so they must never be added to ABLATION_NAMES) -- a fourth
#: list rather than a flag on ABLATION_NAMES, for the same torch-free reason
#: CONTINUATION_NAMES and SEED_NAMES are separate lists. recipe.py asserts
#: this matches its RUN_LIST.
#:
#: UL1/UL2: the same idea at the large tier (L1/L2's shapes,
#: registration only -- see config_registry.ATTENTION_VARIANTS for why L3 has
#: no per-pass counterpart).
#:
#: UL3/UL4: L4's/L5's shapes (S3's/S4's large-tier counterparts,
#: registered alongside these two variants -- see config_registry.ABLATIONS).
VARIANT_NAMES = ("U1", "U2", "U3", "U4", "UL1", "UL2", "UL3", "UL4")

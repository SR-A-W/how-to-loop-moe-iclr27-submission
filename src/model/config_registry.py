"""The four architectures this project trains, as code.

Baseline plus dvf_a/b/c (the fixed-budget series) differ **only** in the (L, R, E) triple.
Everything else -- d_model, top-k, vocabulary, aux-loss coefficients -- is held
fixed, and that is the entire basis for attributing any observed difference to
expert placement.

Until now those four configurations existed only in prose. The series' invariant
tests exercised abstract (L, R, E) triples, which means `d_model == 704` and
`num_active == 2` were never asserted anywhere: not because they were proven
stable, but because no test ever gave them the chance to drift. This module is
the single place they are written down.

`toy` is a deliberately tiny fifth entry used for smoke runs. It is *not* an
architecture under study; it exists so that a launch path can be exercised
without a GPU-hours bill, and it is marked as such.
"""

from __future__ import annotations

from src.model.modeling_loop_lm import LoopMoEConfig
from src.pretrain.ablation_names import ABLATION_NAMES

__all__ = ["ABLATIONS", "ABLATION_TIERS", "ARCHITECTURES", "ATTENTION_VARIANTS",
           "SMOLLM2_VOCAB_SIZE", "TIERS", "build", "build_ablation", "SHARED", "TOY"]

# Fixed across every architecture *and* every size tier. Changing anything here
# changes what the experiment is comparing.
SHARED = dict(
    vocab_size=128256,     # Llama-3 tokenizer, per the LT2-aligned data pipeline (LT2: arXiv 2605.20670)
    context_length=4096,   # LT2 sequence length
    rope_theta=10000.0,
    num_active=2,          # top-k, "following Mixtral" (seedvar §2.2)
    lb_loss_factor=0.01,   # seedvar config.json; NOT stated in the paper.
    lz_loss_factor=0.001,
)

#: Size tiers: (d_model, d_ff, num_heads).
#:
#: `d_model` is **shared by every architecture within a tier** -- that is a hard
#: requirement, not a convenience. The fixed-budget comparison only attributes differences
#: to expert placement if width is held fixed; letting dvf_c use a different
#: d_model to hit an exact parameter target would destroy the comparison to save
#: a label.
#:
#: Consequence: architectures within a tier do
#: NOT have identical total parameters (attention params scale with L, which the
#: E*L=64 invariant does not constrain). The tier name is the tier, not a promise
#: of an exact parameter count.
#:
#: `d_ff/d_model` is held at seedvar's 1920/704 ratio. Head dim is 64 throughout,
#: which makes the head *count* odd at some widths -- seedvar itself uses 11 heads
#: at d=704, so that is consistent with the reference architecture, not a defect.
TIERS: dict[str, tuple[int, int, int]] = {
    # seedvar's published width, reproduced EXACTLY. This tier is a replication,
    # so fidelity outranks throughput: 1920/2 = 960 is not 128-aligned, and we
    # keep it anyway. NOTE: with the Llama-3 vocabulary this totals ~326M
    # parameters, not the 216M of the original -- see `build`'s docstring.
    "seedvar_d704": (704, 1920, 11),
    # New tiers are ours to choose, so they are chosen aligned:
    #   d_model % 128 == 0        -> head_dim 64, even head count
    #   (d_ff // num_active) % 256 == 0  -> the per-expert FFN width, which is the
    #                                       actual K/N dimension of the grouped_mm
    # 0.6B lands on d=1024 / 16 heads, which is *exactly* LT2's published 0.6B
    # shape -- solved for independently, which is a reassuring cross-check.
    "0p6b": (1024, 3072, 16),
    "1p3b": (1792, 4608, 28),
    # For the convergence pilot: a width below 0p6b, so a
    # 50,000-step run finishes in pilot time. Same alignment rules as above --
    # 512 % 128 == 0, and 1536 / 2 = 768 with 768 % 256 == 0 -- so it needs no
    # exception made for it.
    "d512": (512, 1536, 8),
}

BASE_D_MODEL = 128
"""muP proxy width; `width_ratio = d_model / BASE_D_MODEL`."""

#: (L, R, E) per architecture. Invariants: L*R = 16, E*L = 64.
#:
#: NOTE the aux-loss coefficients above are the published values for E=8 only.
#: seedvar never varied E, so applying them unchanged at E=16/32/64 is an
#: extrapolation with no literature support, which is why an aux-loss ablation
#: is required before the fixed-budget runs are believed.
ARCHITECTURES: dict[str, tuple[int, int, int]] = {
    "baseline": (8, 2, 8),
    "dvf_a": (4, 4, 16),
    "dvf_b": (2, 8, 32),
    "dvf_c": (1, 16, 64),
    # ------------------------------------------------------------------------
    # Round 2. NOT a member of the fixed-budget family, and
    # it deliberately BREAKS the E*L=64 invariant: E*L = 128 here, twice every
    # other entry, so this architecture stores twice the expert parameters.
    #
    # Why that is intentional rather than an oversight. This entry exists to pair
    # with `baseline` as "looped vs not looped": both apply 16 layers, both give
    # each layer a pool of 8 experts, both activate k=2 -- so per-token active
    # parameters and per-token FLOPs are *identical*, and the only difference is
    # whether those 16 layer applications reuse 8 sets of weights twice or use 16
    # sets once. Holding effective depth, pool size and per-token compute equal
    # is only possible by letting total expert parameters differ; the two cannot
    # both be held fixed. Total expert parameters are the quantity left unequal,
    # to be reported rather than compensated for.
    #
    # Consequence for reading results: this run carries +66.1% total parameters
    # against `baseline`, so "it scored better" has a parameter-count explanation
    # that cannot be excluded. The informative direction is the other one -- the
    # looped model matching it with 40% fewer parameters.
    "noloop_moe": (16, 1, 8),
}

SMOLLM2_VOCAB_SIZE = 49152
"""SmolLM2 tokenizer vocabulary.

The ablation family uses this; `SHARED` keeps the Llama-3 vocabulary so that every
pre-existing run stays reproducible. The value lives here once rather than being
written into each of the eighteen entries.
"""

#: The ablation family, keyed by the recipe's names. Value: (E, D, L, head, tail).
#:
#: (E, D, L) is the recipe's own notation: E experts per loop-block layer, D layers in the
#: block, L calls of the block. `head`/`tail` are MoE layers run once outside the loop.
#: Effective depth is head + D*L + tail, which is 2 + D*L everywhere except S15.
#:
#: These are NOT members of the fixed-budget (L, R, E) family above and deliberately break its
#: E*L=64 invariant -- the ablation varies E, D and L independently, which is the point.
ABLATIONS: dict[str, tuple[int, int, int, int, int]] = {
    # --- small tier, d_model 1024 -------------------------------------------------
    "S1":  (8, 8, 2, 1, 1),    # the reference point
    "S2":  (16, 4, 4, 1, 1),   # S2-S4: same expert total and same compute as S1,
    "S3":  (32, 2, 8, 1, 1),   #   fewer layers, more experts each, more loops
    "S4":  (64, 1, 16, 1, 1),
    "S5":  (16, 4, 2, 1, 1),   # S5-S7: half the compute of S1, three different ways
    "S6":  (8, 8, 1, 1, 1),
    "S7":  (8, 4, 2, 1, 1),
    "S8":  (8, 8, 3, 1, 1),    # S8-S9: only the loop count changes
    "S9":  (8, 8, 4, 1, 1),
    "S10": (16, 8, 2, 1, 1),   # S10-S11: only experts per layer changes
    "S11": (4, 8, 2, 1, 1),
    "S12": (8, 4, 4, 1, 1),    # S12-S13: fewer experts in total
    "S13": (4, 8, 4, 1, 1),
    "S14": (8, 16, 1, 1, 1),   # same compute, NOT looped: 16 independent layers
    "S15": (8, 8, 2, 2, 2),    # does head/tail thickness change the conclusion?
    # --- large tier, d_model 1792 -------------------------------------------------
    "L1":  (8, 8, 2, 1, 1),
    "L2":  (16, 4, 4, 1, 1),
    "L3":  (8, 16, 1, 1, 1),
    # L4/L5: the large-tier
    # counterparts of S3/S4 -- same (E, D, L) triples, same E*D=64 invariant,
    # registered so UL3/UL4 (per-pass attention) below have a base shape to
    # vary. Registration only, like L3; not yet launched.
    "L4":  (32, 2, 8, 1, 1),
    "L5":  (64, 1, 16, 1, 1),
    # --- pilot tier, d_model 512 --------------------------------------------------
    # Four shapes repeated from the small tier so the pilot answers "does the
    # ordering survive at this width", not "what happens to four new shapes".
    # P1/P2/P3 mirror S1/S2/S3; P6 mirrors S6.
    "P6":  (8, 8, 1, 1, 1),
    "P1":  (8, 8, 2, 1, 1),
    "P2":  (16, 4, 4, 1, 1),
    "P3":  (32, 2, 8, 1, 1),
    # --- loop-count series, d_model 1024 ------------------------------------------
    # These vary how many times the stack is CALLED (32, 32, 24,
    # 64) rather than how wide or deep one block is, which is why their
    # activation-checkpointing modes differ from the rest of the 0p6b tier -- see
    # ABLATION_AC_MODE.
    "N1":  (8, 4, 8, 1, 1),
    "N2":  (16, 4, 8, 1, 1),
    "N3":  (16, 4, 6, 1, 1),
    "N4":  (8, 8, 8, 1, 1),
}

# src.pretrain.ablation_names.ABLATION_NAMES is what torch-free callers (a
# plain status-reporting node with no GPU/torch) read instead of this dict,
# specifically so they never need to import torch just to know which config
# names exist. Asserted equal here, at import time, so the two lists can
# never silently drift apart -- a name added to one without the other fails
# loudly the next time anything imports this module, not months later when
# a torch-free caller quietly stops recognizing a real config.
assert set(ABLATIONS) == set(ABLATION_NAMES), (
    f"ABLATIONS and src.pretrain.ablation_names.ABLATION_NAMES have drifted apart: "
    f"only in ABLATIONS: {set(ABLATIONS) - set(ABLATION_NAMES)}; "
    f"only in ABLATION_NAMES: {set(ABLATION_NAMES) - set(ABLATIONS)}"
)

#: Which width tier each ablation entry runs at. Stated per entry rather than inferred
#: from the name's first letter: a name is a label, and deriving the compute bill from
#: a label's spelling is how a typo becomes a silently different experiment.
ABLATION_TIERS: dict[str, str] = {
    **{f"S{i}": "0p6b" for i in range(1, 16)},
    **{f"L{i}": "1p3b" for i in range(1, 6)},
    **{name: "d512" for name in ("P6", "P1", "P2", "P3")},
    **{name: "0p6b" for name in ("N1", "N2", "N3", "N4")},
}


#: U1-U4: S1-S4's shapes again, with per-pass
#: attention -- each of the R passes over the shared stack gets its own
#: attention parameters instead of sharing one set across passes. NOT a
#: fifth/sixth/... entry in `ABLATIONS`: the (E, D, L, head, tail) shape is
#: unchanged, so the shape table stays the single place that varies, and
#: `ABLATIONS`/`ABLATION_NAMES` keep their count unaffected. `build_ablation`
#: resolves a U-name to its S-name below, rather than this being a second
#: copy of the four shapes.
#:
#: UL1/UL2: the same per-pass-attention idea
#: applied to the large tier (d_model 1792) -- L1's and L2's shapes again,
#: with per_pass_attention=True. L3 is deliberately absent: L3's num_stacks is
#: 1, so a per-pass variant of it would hold exactly one attention set, which
#: is byte-for-byte what the non-per-pass L3 already builds -- there is no
#: second architecture to register.
#:
#: UL3/UL4: L4's and L5's shapes
#: (S3's/S4's large-tier counterparts) with per_pass_attention=True -- unlike
#: L3, L4 and L5 both have num_stacks > 1 (8 and 16 respectively), so their
#: per-pass variants genuinely add independent attention sets rather than
#: reproducing the base shape:
#:
#:   UL3 = L4 (E=32, D=2, num_stacks=8):  8 independent attention sets
#:         (one per loop pass) instead of 1 shared set.
#:         attn params:      25,690,112 (L4, shared) -> 205,520,896 (UL3, 8x)
#:         total params:  1,218,588,672 (L4)         -> 1,398,419,456 (UL3)
#:         (+179,830,784, +14.76%)
#:   UL4 = L5 (E=64, D=1, num_stacks=16): 16 independent attention sets
#:         instead of 1 shared set.
#:         attn params:      12,845,056 (L5, shared) -> 205,520,896 (UL4, 16x)
#:         total params:  1,205,743,616 (L5)         -> 1,398,419,456 (UL4)
#:         (+192,675,840, +15.98%)
#:
#: Both increments are exact -- built on `torch.device("meta")` via
#: `src.stats.param_counts.count_params`, not estimated -- and match
#: `(num_stacks - 1) * num_layers_in_stack * 4 * d_model**2` (the same formula
#: U1-U4/UL1-UL2 follow), i.e. `(R-1)*L*4*d^2`. The extra attention weights
#: also mean extra optimizer state (Adam's two moments) and gradient memory
#: for UL3/UL4 specifically -- not just an activation-checkpointing question
#: -- roughly +180M/+193M params each carrying their own momentum/variance/
#: gradient copies on top of what L4/L5 alone would need.
ATTENTION_VARIANTS: dict[str, str] = {
    "U1": "S1", "U2": "S2", "U3": "S3", "U4": "S4",
    "UL1": "L1", "UL2": "L2", "UL3": "L4", "UL4": "L5",
}


def build_ablation(name: str) -> LoopMoEConfig:
    """Build one of the eighteen ablation configurations, or a U1-U4 attention variant.

    Separate from `build` because the two families differ in more than a triple: the
    ablation uses the SmolLM2 vocabulary and carries head/tail layers, while `build`'s
    architectures must keep producing byte-identical configs so their finished runs stay
    reproducible.

    A name in `ATTENTION_VARIANTS` (U1-U4) resolves to its underlying S-name's shape and
    tier -- the same construction below, branched here rather than duplicated, plus
    `per_pass_attention=True`. Every other name is unaffected: `per_pass_attention` is
    passed only for a U-name, so the 26 existing configs still construct with exactly the
    kwargs they always have.
    """
    if name not in ABLATIONS and name not in ATTENTION_VARIANTS:
        raise ValueError(
            f"unknown ablation {name!r}; choose from "
            f"{sorted(set(ABLATIONS) | set(ATTENTION_VARIANTS))}"
        )
    per_pass_attention = name in ATTENTION_VARIANTS
    shape_name = ATTENTION_VARIANTS.get(name, name)
    experts, depth, loops, head, tail = ABLATIONS[shape_name]
    d_model, d_ff, num_heads = TIERS[ABLATION_TIERS[shape_name]]
    return LoopMoEConfig(
        **dict(SHARED, vocab_size=SMOLLM2_VOCAB_SIZE),
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        width_ratio=d_model / BASE_D_MODEL,
        num_layers_in_stack=depth,
        num_stacks=loops,
        num_experts=experts,
        num_head_layers=head,
        num_tail_layers=tail,
        **({"per_pass_attention": True} if per_pass_attention else {}),
    )


#: Smoke-test shape. Small enough to fit a short single-node job; keeps the
#: real k and a realistic R so the looped/MoE code paths are genuinely exercised.
TOY = dict(
    SHARED,
    context_length=512,
    d_model=256,
    num_heads=4,
    d_ff=512,
    width_ratio=2.0,
)


def build(
    name: str, tier: str | None = None, *, lb_loss_factor: float | None = None
) -> LoopMoEConfig:
    """Construct one of the named architectures at a given size tier.

    `tier` is required for every real architecture (only `"toy"` may omit it).
    It is not defaulted because the tier determines the compute bill and which
    comparison a checkpoint belongs to; silently getting the wrong one would be
    discovered only after spending node-days.

    ⚠️ **`seedvar_d704` is not a 216M model here.** seedvar's 216M figure is with
    the GPT-2 vocabulary (50,257). This project uses the Llama-3 tokenizer
    (128,256) to align with LT2's data, and the same architecture at the same
    width then totals ~326M, with embeddings rising from 33% to 55% of the model.
    The architecture is faithfully reproduced; the parameter count and the
    embedding/compute balance are not. Anyone comparing our `seedvar_d704`
    numbers against the published 216M checkpoints must account for that.

    `lb_loss_factor` overrides the load-balancing coefficient in `SHARED`. It is
    the one hyperparameter a run may vary, and only because a run exists to vary
    it (`noloop_moe_1p3b_lb0p02`). Passing `None`
    keeps the registry value, so every existing caller and every existing run is
    unaffected. Keep the override on the *run*, never edit `SHARED`: `SHARED` is
    what makes the other runs comparable, and a run's own value reaches the
    manifest through `model_config`, so a checkpoint states which coefficient
    trained it rather than requiring anyone to infer it from the run's name.
    """
    if name == "toy":
        if tier is not None:
            raise ValueError("'toy' has its own fixed shape; do not pass a tier")
        if lb_loss_factor is not None:
            raise ValueError("'toy' does not take an lb_loss_factor override")
        return LoopMoEConfig(**TOY, num_layers_in_stack=2, num_stacks=2, num_experts=4)

    if name not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {name!r}; choose from {sorted(ARCHITECTURES)} or 'toy'"
        )
    if tier is None:
        raise ValueError(
            f"architecture {name!r} requires an explicit tier; choose from {sorted(TIERS)}"
        )
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(TIERS)}")

    d_model, d_ff, num_heads = TIERS[tier]
    layers, loops, experts = ARCHITECTURES[name]
    shared = dict(SHARED)
    if lb_loss_factor is not None:
        if not lb_loss_factor > 0:
            raise ValueError(f"lb_loss_factor must be positive, got {lb_loss_factor!r}")
        shared["lb_loss_factor"] = lb_loss_factor
    return LoopMoEConfig(
        **shared,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        width_ratio=d_model / BASE_D_MODEL,
        num_layers_in_stack=layers,
        num_stacks=loops,
        num_experts=experts,
    )

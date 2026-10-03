"""SmolTalk `smol-smoltalk` loading + token-budget subsampling.

Dataset identity (verified against the HF Hub API, not assumed):
`HuggingFaceTB/smol-smoltalk` is its own dataset repo
(`load_dataset("HuggingFaceTB/smol-smoltalk", split="train")`), NOT a config
inside the (larger, different) `HuggingFaceTB/smoltalk` repo -- the two share a
near-identical name but are different Hub entities. `smol-smoltalk`'s `train`
split has 460,341 examples, each a `messages` list (`role`/`content` dicts) plus
a `source` field.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DATASET_ID = "HuggingFaceTB/smol-smoltalk"
DATASET_SPLIT = "train"
# Pinned to the dataset repo's current HEAD commit (read via the HF Hub API,
# `GET /api/datasets/HuggingFaceTB/smol-smoltalk` -> `sha`) --
# not left floating. Same bitwise-reproducibility rationale as everywhere else in this
# project: an unpinned `load_dataset(DATASET_ID, split=DATASET_SPLIT)` call
# would silently pick up whatever the Hub repo's default branch points to on
# the day a run happens to execute, which could differ between the probe run
# and the formal run (or between two formal runs) without either being
# reproducible from its own arguments. Also lets the training cluster pre-download this exact
# revision ahead of a maintenance window instead of only the unpinned latest.
DATASET_REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"


# In-repo cache for `load_full_dataset`'s full-corpus token count -- see
# `load_full_dataset`'s docstring for the full rationale. Committed to git
# (not gitignored, not a runtime-only artifact) so the training cluster and every formal run
# hit a warm cache from the first invocation, not just after some process
# happens to populate it locally.
TOKEN_STATS_CACHE_PATH = Path(__file__).resolve().parent / "token_stats_cache.json"


def _portable_tokenizer_identity(tokenizer: Any) -> str:
    """`tokenizer.name_or_path` is exactly what it was loaded from -- for
    this project's bundled tokenizer (a LOCAL directory, not a Hub repo id)
    that is an ABSOLUTE filesystem path under wherever THIS checkout happens to
    live (e.g. `<checkout root>/src/data/artifacts/tokenizer/smollm2` on one
    machine, a different prefix on another). Hashing that raw absolute path would make the cache key
    machine/checkout-dependent -- exactly the opposite of "commit one cache
    entry, every checkout hits it" this caching layer exists for. If the
    path is absolute and lives inside this repo, this returns it made
    relative to the repo root (stable across checkouts, since it's the
    same git repo everywhere); a genuine Hub repo id (not a local path at
    all) is returned unchanged -- it's already portable.
    """
    name_or_path = getattr(tokenizer, "name_or_path", None) or ""
    path = Path(name_or_path)
    if path.is_absolute() and path.exists():
        repo_root = Path(__file__).resolve().parent.parent
        try:
            return str(path.resolve().relative_to(repo_root))
        except ValueError:
            return name_or_path  # outside this repo -- no portable form, best-effort as-is
    return name_or_path


def _token_stats_cache_key(*, tokenizer: Any) -> str:
    """`num_tokens_selected` (the full-corpus token count `load_full_dataset`
    computes) is a pure function of five things: `DATASET_ID`, `DATASET_
    SPLIT`, `DATASET_REVISION` (all fixed module constants), the tokenizer
    repo, and the chat template applied -- NOT of `seed` (the shuffle order
    changes, but summing every row's token count is order-independent, so
    the sum is identical regardless of seed; verified by reading the code:
    `load_full_dataset` sums over ALL rows, no early break). Hashing exactly these five values (tokenizer identity via
    `_portable_tokenizer_identity`, NOT the raw `name_or_path` -- see that
    function's docstring for why a raw local path would break the whole
    point of a committed cache; `chat_template` is read directly so the key
    changes if a caller ever sets a different template, rather than
    trusting an out-of-band assumption that it's always `MINIMAL_CHAT_
    TEMPLATE`) means the cache key is exactly as specific as the actual
    dependency graph -- no less (which would risk a stale hit) and no more
    (which would force a needless recompute on something that doesn't
    actually change the answer, e.g. `seed`, or the checkout's absolute path).
    """
    material = "\x1f".join([
        DATASET_ID, DATASET_SPLIT, DATASET_REVISION,
        _portable_tokenizer_identity(tokenizer),
        tokenizer.chat_template or "",
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _load_token_stats_cache() -> dict[str, Any]:
    if not TOKEN_STATS_CACHE_PATH.exists():
        return {}
    return json.loads(TOKEN_STATS_CACHE_PATH.read_text())


def _write_token_stats_cache(cache: dict[str, Any]) -> None:
    # Sorted keys + trailing newline -- deterministic byte-for-byte output so
    # a re-run that recomputes the SAME entries produces a clean (or empty)
    # git diff, not spurious key-ordering churn.
    TOKEN_STATS_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def _rendered_token_count(messages: list[dict], tokenizer: Any) -> int:
    """Chat-template-rendered token count for one example's `messages` list.

    Shared by `load_probe_dataset` (budget accounting) and `load_full_dataset`
    (full-corpus token count) so the two never compute this differently --
    both need the SAME definition of "how many tokens does this example cost
    at train time" (template markers + eos included, see
    `load_probe_dataset`'s docstring for why untemplated counting would be
    wrong).
    """
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def load_full_dataset(
    *, seed: int, tokenizer: Any, recompute_token_stats: bool = False
) -> tuple[list[dict], dict[str, Any]]:
    """Load ALL of `smol-smoltalk`'s `train` split (no token-budget cutoff) --
    the formal instruction-tuning tier's data path (same dataset as the
    probe, but the full corpus, not a token-capped prefix).

    Unlike `load_probe_dataset`, this has no early-break over the EXAMPLES
    loop -- every row is loaded, always. `seed` is still required (same
    bitwise-reproducibility rationale as the probe) even though a full pass trains on
    every row regardless of shuffle order -- the shuffle determines epoch-1
    batch composition/order, which is part of what must be reproducible from
    the arguments alone.

    ## Token-count caching (superseding an earlier `compute_token_stats`
    "main-process-only" attempt at this same problem)

    The full-corpus token count exists to answer "how many tokens does the
    full corpus have", for the run's manifest -- it is NOT
    consumed by any scheduling/step-count math (`build_formal_sft_config`
    uses `num_examples`, i.e. `len(examples)`, which this function already
    returns for free without tokenizing anything). Computing it the naive
    way (render + tokenize every one of 460,341 rows) was measured on the
    training cluster at ~55 minutes wall-clock -- and this project's first attempt at
    fixing that (`compute_token_stats=False` on non-main ranks) turned out
    NOT to help: under `torchrun`, every non-main rank blocks on
    `main_process_first()` waiting for rank0's `SFTTrainer._prepare_dataset`
    call, which happens AFTER this function returns -- so rank0's ~55
    minutes here sits on the critical path regardless of who else skips it.

    Key insight (verified against the code, not assumed): this count
    is a PURE FUNCTION of `(DATASET_ID, DATASET_SPLIT, DATASET_REVISION,
    tokenizer repo, chat template)` -- it does NOT depend on `seed` (shuffle
    reorders rows; summing every row's count is order-independent) or on
    anything about the run (batch size, world size, dtype...). So it is safe
    to compute ONCE and cache -- see `_token_stats_cache_key`.

    `recompute_token_stats=False` (default): look up the cache
    (`TOKEN_STATS_CACHE_PATH`, committed to git). A cache MISS raises
    `RuntimeError` -- NOT a silent fall-back to the 55-minute pass (that
    would silently reintroduce the exact tax this caching layer exists to
    remove, the first time anyone forgets to pre-seed the cache for a new
    tokenizer/template/revision combination). `recompute_token_stats=True`
    (developer-invoked, e.g. `src/posttrain/sft_formal.py --recompute-token-
    stats`) runs the slow pass once and WRITES the result into the cache
    file, so every subsequent run (including every future preemption-
    triggered restart of a multi-day formal run) hits the cache instead.
    The intended real-world usage: a human runs this exactly once per
    (dataset revision, tokenizer, chat template) combination, on a dev
    machine, and commits the updated cache file -- the training cluster and
    the three formal runs then only ever hit the cache, never the slow path.

    A cache HIT is cross-checked against the freshly-loaded example count
    (`len(examples)`, always cheap) -- a mismatch means the cache entry is
    stale (e.g. the pinned revision's row count somehow changed) and raises
    rather than silently returning a token count that doesn't correspond to
    the data this call actually loaded.

    Returns `(examples, stats)` in the same shape as `load_probe_dataset` --
    `examples` is `trl.SFTTrainer`-ready `{"messages": [...]}` dicts,
    `stats` records the full-corpus token count for the run's manifest plus
    `token_count_cache_key` (which cache entry produced it, for auditability).
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT, revision=DATASET_REVISION).shuffle(seed=seed)

    # Cheap: no tokenizer call, just materializing the row list. This is
    # NOT the ~55-minute cost measured on the cluster -- that's `_rendered_token_
    # count` below, only reached on a cache miss + explicit recompute.
    examples: list[dict] = [{"messages": row["messages"]} for row in ds]

    cache_key = _token_stats_cache_key(tokenizer=tokenizer)
    cache = _load_token_stats_cache()
    cached_entry = cache.get(cache_key)

    if recompute_token_stats or cached_entry is None:
        if not recompute_token_stats:
            raise RuntimeError(
                f"No cached token count for key {cache_key} -- "
                f"({DATASET_ID}, {DATASET_SPLIT}, {DATASET_REVISION}, "
                f"tokenizer={_portable_tokenizer_identity(tokenizer)!r}) has never been standardized "
                "into token_stats_cache.json. Refusing to silently fall back to the ~55-minute "
                "per-row tokenize-and-count pass -- pass recompute_token_stats=True once "
                "(src/posttrain/sft_formal.py's "
                "--recompute-token-stats CLI flag) to compute and cache it, then commit the "
                "updated posttrain/token_stats_cache.json."
            )
        total_tokens = sum(_rendered_token_count(row["messages"], tokenizer) for row in ds)
        cache[cache_key] = {
            "num_tokens_selected": total_tokens,
            "num_examples_selected": len(examples),
            "dataset_id": DATASET_ID,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "tokenizer_repo": _portable_tokenizer_identity(tokenizer),
        }
        _write_token_stats_cache(cache)
        cached_entry = cache[cache_key]

    if cached_entry["num_examples_selected"] != len(examples):
        raise RuntimeError(
            f"cached token-stats entry for key {cache_key} claims "
            f"{cached_entry['num_examples_selected']} examples, but this run just loaded "
            f"{len(examples)} from {DATASET_ID}:{DATASET_SPLIT}@{DATASET_REVISION} -- the cache "
            "entry is stale (or the pinned revision's content changed unexpectedly). Re-run with "
            "recompute_token_stats=True to refresh it; do not trust the stale cached token count."
        )

    stats = {
        "dataset_id": DATASET_ID,
        "split": DATASET_SPLIT,
        "seed": seed,
        "num_examples_selected": len(examples),
        "num_tokens_selected": cached_entry["num_tokens_selected"],
        "token_count_cache_key": cache_key,
    }
    return examples, stats


def load_probe_dataset(
    *,
    seed: int,
    token_budget: int,
    tokenizer: Any,
) -> tuple[list[dict], dict[str, Any]]:
    """Shuffle `smol-smoltalk` deterministically by `seed`, then take examples
    (in that shuffled order) until their tokenized length (chat-template
    applied) sums to >= `token_budget`.

    Both `seed` and `token_budget` are required -- no default seed (bitwise
    reproducibility: a probe run's exact example set must be reproducible from
    the arguments alone) and no default budget (semantic parameters are
    required, never defaulted: how many tokens a probe SFT pass trains on is
    exactly the kind of choice that must not silently default -- see
    `src/posttrain/sft_probe.py`'s module docstring for the initial
    suggested-value table, which is NOT wired in here as a fallback).

    Iterates the shuffled dataset lazily (via `datasets.Dataset.shuffle` + a
    plain Python loop with an early break) rather than tokenizing the full
    460K-example corpus up front -- a probe's token budget only ever needs a
    small prefix of the shuffled order, and eagerly tokenizing the rest would
    be pure waste on the very budget (wall-clock) this function exists to
    respect.

    Returns `(examples, stats)` where `examples` is a list of raw
    `{"messages": [...]}` dicts (ready for `trl.SFTTrainer`'s expected input
    shape) and `stats` records exactly what was selected, for the run's
    manifest (`data_position` field of `save_trajectory`'s `manifest_fields`).
    """
    from datasets import load_dataset

    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")

    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT, revision=DATASET_REVISION).shuffle(seed=seed)

    selected: list[dict] = []
    total_tokens = 0
    for row in ds:
        messages = row["messages"]
        # Tokenize with the chat template applied -- an untemplated token count
        # would undercount (misses the `<|role|>` markers and eos separators
        # every example actually pays for at train time) and make the budget
        # math wrong in the direction that silently overshoots wall-clock.
        # (Shared with `load_full_dataset` via `_rendered_token_count` -- both
        # must use the same definition of "how many tokens does this cost".)
        selected.append({"messages": messages})
        total_tokens += _rendered_token_count(messages, tokenizer)
        if total_tokens >= token_budget:
            break
    else:
        raise RuntimeError(
            f"exhausted all {len(ds)} examples in {DATASET_ID}:{DATASET_SPLIT} "
            f"(only reached {total_tokens} tokens) without meeting "
            f"token_budget={token_budget}. The budget is larger than this "
            "dataset can supply -- lower it or add a second data source "
            "(not silently truncating the requested budget)."
        )

    stats = {
        "dataset_id": DATASET_ID,
        "split": DATASET_SPLIT,
        "seed": seed,
        "token_budget_requested": token_budget,
        "num_examples_selected": len(selected),
        "num_tokens_selected": total_tokens,
    }
    return selected, stats

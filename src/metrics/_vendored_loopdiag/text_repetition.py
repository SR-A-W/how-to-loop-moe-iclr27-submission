# ---------------------------------------------------------------------------
# VENDORED FILE — DO NOT EDIT BELOW THIS HEADER.
#
# Source: an earlier, separate diagnostics code base (read-only reference);
# original module path metrics/text_repetition.py.
#
# WHY vendored instead of imported: this is the repo that ships to the
# training cluster; that code base does not and is not a
# submodule. Training-time metrics must run standalone on the training
# cluster, so the metric DEFINITIONS this project already verified/tested are
# copied verbatim rather than reimplemented (reuse, don't reinvent).
#
# Everything from here to end-of-file is code-identical to the source path
# above. In the development repo a sync test (not shipped
# here) compares this file's code with the source file and fails on drift.
# ---------------------------------------------------------------------------

"""Token-level text repetition statistics (type-token ratio, bigram repeat
rate) for a document's model input window.

Reimplementation of the one-off script used in the Ouro exit-gate domain
analysis (that analysis was done directly in a throwaway script, never
landed in this package) -- promoted to a package function so that any tool
re-used across analyses lives here, not re-copy-pasted. **Verified (not
assumed) to exactly reproduce the original 8-document numbers** (8/8 exact
match in tests) before being trusted on the new 40-document corpus -- a
reimplementation is not the same as a verified reimplementation.

⚠️ Precise definitions (same-named metrics with incompatible definitions
have bitten this project before, e.g.
Huginn's `CosineExitEvaluator` vs `looped.consecutive_step_angle`):

- These statistics are computed on TOKEN IDS from the model's own
  tokenizer, truncated to the model's actual input window (e.g. 512
  tokens) -- NOT on whitespace-split words. Word-level and token-level
  values differ substantially (verified: Github's word-level TTR is 0.421,
  its token-level TTR is 0.156 -- a >2.5x difference, entirely an artifact
  of which unit is being counted, not a property of the text).
- `type_token_ratio(ids) = len(set(ids)) / len(ids)` -- fraction of the
  window's token positions that are the FIRST occurrence of their token id.
  Standard TTR, just on token ids instead of words.
- `bigram_repeat_rate(ids)`: let `bigrams[i] = (ids[i], ids[i+1])` for
  `i in 0..len(ids)-2` (overlapping, stride 1). This is **NOT**
  `1 - n_unique_bigrams / n_bigrams` (that alternative, equally
  defensible-sounding definition was tried first and gives a DIFFERENT
  number, 0.759 vs the correct 0.8395 for the Github reference case --
  verified by direct comparison, not assumed equivalent). The correct
  (original, reproduced) definition is: **the fraction of bigram POSITIONS
  whose bigram value occurs more than once anywhere in the window** --
  i.e. `sum(count - 1 for count in Counter(bigrams).values() if count > 1) / len(bigrams)`
  is mathematically NOT what this is; it is
  `sum(count for count in Counter(bigrams).values() if count > 1) / len(bigrams)`
  (every occurrence of a repeated bigram counts, including its first
  occurrence -- only bigrams that occur exactly once anywhere are excluded
  from the numerator). This weights by how often a repeated pattern
  recurs, not merely whether at least one repeat exists.
"""

from __future__ import annotations

from collections import Counter


def type_token_ratio(token_ids: list[int]) -> float:
    """`len(set(token_ids)) / len(token_ids)`. See module docstring for why
    this must be computed on token ids, not words.

    Args:
        token_ids: list of integer token ids (already truncated to whatever
            window is being analyzed, e.g. the model's actual input length).

    Returns:
        Float in (0, 1]. Raises ZeroDivisionError on an empty list (an
        empty document is a caller error, not silently handled here).
    """
    return len(set(token_ids)) / len(token_ids)


def bigram_repeat_rate(token_ids: list[int]) -> float:
    """Fraction of overlapping-bigram POSITIONS whose bigram value recurs
    elsewhere in `token_ids`. See module docstring for the exact formula
    and why the more obvious `1 - n_unique/n_total` alternative is a
    DIFFERENT, incorrect-for-this-project's-purposes quantity.

    Args:
        token_ids: list of integer token ids, length >= 2.

    Returns:
        Float in [0, 1). Raises ZeroDivisionError if `len(token_ids) < 2`
        (no bigrams to count).
    """
    bigrams = list(zip(token_ids[:-1], token_ids[1:]))
    counts = Counter(bigrams)
    n_repeated_positions = sum(count for count in counts.values() if count > 1)
    return n_repeated_positions / len(bigrams)


def text_repetition_stats(token_ids: list[int]) -> dict:
    """Both statistics in one call, for convenience.

    Returns:
        `{"type_token_ratio": ..., "bigram_repeat_rate": ..., "n_tokens": ...}`.
    """
    return {
        "type_token_ratio": type_token_ratio(token_ids),
        "bigram_repeat_rate": bigram_repeat_rate(token_ids),
        "n_tokens": len(token_ids),
    }

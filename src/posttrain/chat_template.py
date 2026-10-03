"""A minimal chat template + a minimal instruction-following probe.

## Why a chat template lives here

the Llama-3 tokenizer's `tokenizer_config.json` (not bundled with this repository;
see `src.data.tokenizer`) ships with no
`chat_template` field -- confirmed by reading the file directly. That is
the correct state for a *base*-tokenizer bundle (a real Llama-3-Instruct tokenizer
would carry one); SmolTalk's `smol-smoltalk` is `messages`-formatted chat data, so
something has to turn `[{"role": ..., "content": ...}, ...]` into a token sequence
before SFT can consume it. Rather than fetch a real Llama-3-Instruct chat template
(a new external dependency + a template designed for a different vocabulary of
special tokens, most of which the bundled tokenizer does not even have registered),
this module defines the simplest template that:

    1. Uses only plain text + the tokenizer's own `eos_token` -- no new special
       tokens, so no embedding-table resize is needed (a probe script has no
       business changing the model's vocabulary).
    2. Marks each turn with a literal `<|role|>` line, intended to let a
       prompt/completion-masking mechanism (if one is ever configured) tell
       prompt tokens apart from completion tokens.

Note: a `SFTTrainer.train()` run reporting `"Dropping fully masked examples"`
with 0 dropped is NOT evidence of correct prompt/completion masking. The
filter TRL reports this metric for is `any(label != -100)`; without
`completion_only_loss` or `assistant_only_loss` set, and without a
`{% generation %}` marker in the template for either to key off, every
example trivially has at least one non-masked label and the filter reports
0 dropped **unconditionally** -- a tautology that cannot distinguish
"masking worked" from "no masking is happening at all" (every token, prompt
included, contributing to the loss).

Masking policy: **assistant-only masking**, adopted before any of the three
formal runs started (earlier probe numbers produced under the old
full-sequence config are NOT comparable to anything trained after this
switch). `MINIMAL_CHAT_TEMPLATE` below wraps each assistant turn's content (+ its trailing
`eos_token`) in `{% generation %}...{% endgeneration %}`, and `posttrain/
sft_probe.py::build_sft_config()` (shared by the probe and, via its own
`build_formal_sft_config()`, the formal tier) sets `assistant_only_loss=True`
-- the `<|role|>` markers are no longer purely cosmetic, TRL's dataset
preparation reads the generation markers to build `assistant_masks`, which
become `-100` on every non-assistant (user/system) position.

This is deliberately NOT trying to look like a real product's chat format --
probe SFT only needs the model to reliably learn "assistant turns follow user
turns and end at eos", not to match any downstream serving stack's prompt format.
"""

from __future__ import annotations

import re
import string

# Jinja2 chat template, HF `tokenizer.apply_chat_template` convention.
# `{{ eos_token }}` after each turn gives the model a token to learn to stop on
# (both training turns and, at generation prompt time, no trailing eos so
# generation can continue).
#
# `{% generation %}...{% endgeneration %}` around the assistant turn's content
# (assistant-only masking policy): these are NOT
# literal output -- transformers' chat-template Jinja environment recognizes
# them as markers (they render to nothing themselves) that `tokenizer.
# apply_chat_template(..., return_assistant_tokens_mask=True)` uses to build
# an `assistant_masks` array alongside `input_ids`, which is what TRL's
# `assistant_only_loss=True` (`src/posttrain/sft_probe.py::build_sft_config()`)
# needs to mask every non-assistant token's label to -100. Rendered TEXT is
# therefore byte-identical to before this change -- only the `role == "assistant"`
# branch exists so the marker can wrap just the content + its trailing
# `eos_token` (not the `<|assistant|>\n` role header, which stays outside the
# mask: it is prompt context the model conditions on, not something to train
# it to reproduce). The `eos_token` is deliberately INSIDE the generation
# block, not after it -- TRL warns (`is_chat_template_stop_token_trained`) if
# a template's end-of-turn token falls outside the assistant mask, since that
# would leave the model never trained to predict where to stop.
MINIMAL_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<|' + message['role'] + '|>\n' }}"
    "{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}"
    "{{ '\n' }}"
    "{% else %}"
    "{{ '<|' + message['role'] + '|>\n' + message['content'] + eos_token + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)


# ---------------------------------------------------------------------------
# Minimal instruction-following probe
# ---------------------------------------------------------------------------
# Scope: a minimal version (fixed prompts + rule-based format checks), not
# IFEval. The rule criteria are hard-coded and tested.
#
# Three prompts, each with a single deterministic, regex-based format rule. This
# is NOT a correctness/quality eval (a probe SFT run with a token budget in the
# tens-of-millions range has no business being graded on *content* quality) --
# it only answers "did the model learn to follow a simple format instruction at
# all", which is the cheapest observable signal for "instruction-following
# capability did or did not emerge from this SFT pass". See Snell et al.
# (COLM 2024) and Du (arXiv 2601.13244) for why finetuning-as-probe is the
# project's chosen methodology.

INSTRUCTION_PROBES: tuple[dict[str, str], ...] = (
    {
        "id": "three_items",
        "prompt": "List exactly three colors, one per line, and nothing else.",
        "rule": "exactly_n_lines",
    },
    {
        "id": "one_sentence",
        "prompt": "In exactly one sentence, explain why the sky is blue.",
        "rule": "single_sentence",
    },
    {
        "id": "one_word",
        "prompt": "Answer with a single word only: what color is grass?",
        "rule": "single_word",
    },
    # --- expansion to 19 probes ---
    # The 3 probes above keep their original `id`s unchanged (backward
    # compatibility with any already-collected probe results); everything
    # below is new. Grouped into 5 capability groups --
    # group comments are for humans reading this file, not a runtime
    # distinction (INSTRUCTION_PROBES stays one flat tuple).
    #
    # G1: format-following
    {
        "id": "numbered_list",
        "prompt": "List exactly three fruits as a numbered list (1. 2. 3.), one per line, and nothing else.",
        "rule": "numbered_list",
    },
    {
        "id": "comma_separated",
        "prompt": "Name exactly four seasons, separated by commas, on a single line, and nothing else.",
        "rule": "comma_separated",
    },
    {
        "id": "all_caps",
        "prompt": "Answer in all capital letters: what is the opposite of hot?",
        "rule": "all_caps",
    },
    # G2: constraint adherence (length/count)
    {
        "id": "max_five_words",
        "prompt": "Describe a rainy day in at most five words.",
        "rule": "max_five_words",
    },
    {
        "id": "exact_six_words",
        "prompt": "Write a sentence about the ocean using exactly six words.",
        "rule": "exact_six_words",
    },
    {
        "id": "min_ten_words",
        "prompt": "Explain what photosynthesis is in at least ten words.",
        "rule": "min_ten_words",
    },
    {
        "id": "no_punctuation",
        "prompt": "Answer with no punctuation marks at all: what do bees make?",
        "rule": "no_punctuation",
    },
    {
        "id": "starts_with_yes_no",
        "prompt": "Answer the question and make sure your response starts with the word 'Yes' or 'No': Is the sun a star?",
        "rule": "starts_with_yes_no",
    },
    # G3: structural instructions
    {
        "id": "bullet_points",
        "prompt": "List exactly three benefits of exercise as bullet points, each starting with '- '.",
        "rule": "bullet_points",
    },
    {
        "id": "qa_format",
        "prompt": "Answer using the format 'Q: <question> A: <answer>' for the question: What is 2+2?",
        "rule": "qa_format",
    },
    {
        "id": "two_paragraphs",
        "prompt": "Write your answer as exactly two paragraphs separated by a blank line, describing your favorite season.",
        "rule": "two_paragraphs",
    },
    # G4: refusal / boundary awareness
    {
        "id": "refuse_unknown",
        "prompt": "What is my mother's maiden name? If you don't know, say 'I don't know.'",
        "rule": "refuse_unknown",
    },
    {
        "id": "exact_repeat",
        "prompt": "Repeat the following word exactly once: banana",
        "rule": "exact_repeat",
    },
    # G5: simple instruction execution
    {
        "id": "count_sequence",
        "prompt": "Count from 1 to 5, separated by spaces, and nothing else.",
        "rule": "count_sequence",
    },
    {
        "id": "sort_ascending",
        "prompt": "List the numbers 3, 1, 2 in ascending order, separated by commas.",
        "rule": "sort_ascending",
    },
    {
        "id": "yes_no_only",
        "prompt": "Answer only 'yes' or 'no': Is water wet?",
        "rule": "yes_no_only",
    },
)


def check_exactly_n_lines(response: str, n: int = 3) -> tuple[bool, str]:
    """Response has exactly `n` non-empty lines (the `three_items` probe)."""
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    ok = len(lines) == n
    return ok, f"found {len(lines)} non-empty line(s), expected {n}"


def check_single_sentence(response: str) -> tuple[bool, str]:
    """Response contains exactly one sentence-terminating punctuation mark and no
    internal line breaks (a cheap, deliberately crude proxy for "one sentence" --
    not a real sentence tokenizer, which would be over-engineering for a format
    probe)."""
    text = response.strip()
    if "\n" in text:
        return False, "response contains a line break, expected a single line"
    terminators = re.findall(r"[.!?]", text)
    ok = len(terminators) == 1 and text.endswith(tuple(".!?"))
    return ok, f"found {len(terminators)} sentence-terminator(s), expected exactly 1 (at the end)"


def check_single_word(response: str) -> tuple[bool, str]:
    """Response is a single whitespace-delimited token (ignoring surrounding
    punctuation/whitespace)."""
    text = response.strip().strip(".!?,;:\"'")
    words = text.split()
    ok = len(words) == 1
    return ok, f"found {len(words)} word(s), expected 1"


# --- new rule functions for the 19-probe expansion (see the group comments
# above INSTRUCTION_PROBES). Same style as the original 3: cheap, deterministic
# regex/string-op judges, (bool, str) return, no LLM judge, no new dependency
# beyond the stdlib `string` module (for `no_punctuation`'s character set).


def check_numbered_list(response: str, n: int = 3) -> tuple[bool, str]:
    """Exactly `n` non-empty lines, each starting with "<i>. " for i=1..n in
    order (the `numbered_list` probe: G1 format-following)."""
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    if len(lines) != n:
        return False, f"found {len(lines)} non-empty line(s), expected {n}"
    for i, line in enumerate(lines, start=1):
        if not re.match(rf"^{i}\.\s", line):
            return False, f"line {i} ({line!r}) does not start with '{i}. '"
    return True, f"all {n} lines correctly numbered"


def check_comma_separated(response: str, n: int = 4) -> tuple[bool, str]:
    """Single line (no internal line breaks), exactly `n-1` commas (i.e. `n`
    comma-separated items) (the `comma_separated` probe: G1)."""
    text = response.strip()
    if "\n" in text:
        return False, "response contains a line break, expected a single line"
    comma_count = text.count(",")
    ok = comma_count == n - 1
    return ok, f"found {comma_count} comma(s), expected {n - 1} (for {n} items)"


def check_all_caps(response: str) -> tuple[bool, str]:
    """Response's letters are all uppercase, and it contains at least one
    letter (avoids vacuously passing on an empty/punctuation-only response)
    (the `all_caps` probe: G1)."""
    text = response.strip()
    has_letter = any(c.isalpha() for c in text)
    ok = has_letter and text == text.upper()
    return ok, f"has_letter={has_letter}, all_upper={text == text.upper()}"


def check_max_words(response: str, n: int = 5) -> tuple[bool, str]:
    """At most `n` whitespace-delimited words (the `max_five_words` probe: G2)."""
    words = response.strip().split()
    ok = len(words) <= n
    return ok, f"found {len(words)} word(s), expected at most {n}"


def check_exact_words(response: str, n: int = 6) -> tuple[bool, str]:
    """Exactly `n` whitespace-delimited words (the `exact_six_words` probe: G2)."""
    words = response.strip().split()
    ok = len(words) == n
    return ok, f"found {len(words)} word(s), expected exactly {n}"


def check_min_words(response: str, n: int = 10) -> tuple[bool, str]:
    """At least `n` whitespace-delimited words (the `min_ten_words` probe: G2)."""
    words = response.strip().split()
    ok = len(words) >= n
    return ok, f"found {len(words)} word(s), expected at least {n}"


def check_no_punctuation(response: str) -> tuple[bool, str]:
    """No characters from `string.punctuation` anywhere in the response (the
    `no_punctuation` probe: G2)."""
    found = [c for c in response if c in string.punctuation]
    ok = len(found) == 0
    return ok, f"found {len(found)} punctuation character(s): {found!r}"


def check_starts_with_yes_no(response: str) -> tuple[bool, str]:
    """Response starts with the word "yes" or "no" (word boundary, not a
    substring match -- e.g. "Yesterday" must not count) (the
    `starts_with_yes_no` probe: G2)."""
    text = response.strip()
    ok = bool(re.match(r"^(yes|no)\b", text, re.IGNORECASE))
    return ok, f"response starts with: {text[:10]!r}"


def check_bullet_points(response: str, n: int = 3) -> tuple[bool, str]:
    """Exactly `n` non-empty lines, each starting with "- " (the
    `bullet_points` probe: G3)."""
    lines = [line for line in response.strip().splitlines() if line.strip()]
    if len(lines) != n:
        return False, f"found {len(lines)} non-empty line(s), expected {n}"
    non_bullets = [line for line in lines if not line.startswith("- ")]
    ok = len(non_bullets) == 0
    return ok, f"{len(non_bullets)} of {n} line(s) do not start with '- '"


def check_qa_format(response: str) -> tuple[bool, str]:
    """Response matches the "Q: <question> A: <answer>" template (the
    `qa_format` probe: G3) -- checks "Q:" appears before "A:", both with
    non-empty content after them, not a strict single-line format (models
    may wrap)."""
    text = response.strip()
    match = re.search(r"Q:\s*(.+?)\s*A:\s*(.+)", text, re.DOTALL)
    ok = match is not None and bool(match.group(1).strip()) and bool(match.group(2).strip())
    return ok, "matches 'Q: ... A: ...' template" if ok else "does not match 'Q: ... A: ...' template"


def check_two_paragraphs(response: str) -> tuple[bool, str]:
    """Response splits into exactly 2 non-empty paragraph blocks separated by
    a blank line (the `two_paragraphs` probe: G3 -- flagged "harder" in the
    difficulty tiers, small models may struggle with this one)."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", response.strip()) if block.strip()]
    ok = len(blocks) == 2
    return ok, f"found {len(blocks)} paragraph block(s), expected 2"


def check_refuse_unknown(response: str, phrase: str = "i don't know") -> tuple[bool, str]:
    """Response contains the refusal phrase (case-insensitive) (the
    `refuse_unknown` probe: G4) -- tests whether the model follows an
    explicit "say X if you don't know" meta-instruction, not whether it
    actually knows the (unknowable) fact."""
    ok = phrase in response.lower()
    return ok, f"phrase {phrase!r} {'found' if ok else 'not found'} in response"


def check_exact_repeat(response: str, word: str = "banana") -> tuple[bool, str]:
    """Response is exactly the target word, once, nothing else (the
    `exact_repeat` probe: G4)."""
    text = response.strip().lower()
    ok = text == word
    return ok, f"response (normalized) is {text!r}, expected exactly {word!r}"


def check_count_sequence(response: str, sequence: str = "1 2 3 4 5") -> tuple[bool, str]:
    """Response is exactly the target space-separated sequence (whitespace
    normalized) (the `count_sequence` probe: G5)."""
    normalized = " ".join(response.strip().split())
    ok = normalized == sequence
    return ok, f"response (whitespace-normalized) is {normalized!r}, expected {sequence!r}"


def check_ascending_numbers(response: str, expected: tuple[int, ...] = (1, 2, 3)) -> tuple[bool, str]:
    """The numbers appearing in the response, in order, equal `expected` (the
    `sort_ascending` probe: G5) -- extracts digit runs rather than requiring
    an exact surrounding format, since the instruction only constrains the
    numbers' order, not the surrounding prose/punctuation."""
    found = tuple(int(x) for x in re.findall(r"\d+", response))
    ok = found == expected
    return ok, f"found number sequence {found}, expected {expected}"


def check_yes_no_only(response: str) -> tuple[bool, str]:
    """Response is exactly "yes" or "no" (case-insensitive, trailing `.`/`!`
    tolerated) and nothing else (the `yes_no_only` probe: G5)."""
    text = response.strip().lower().rstrip(".!")
    ok = text in ("yes", "no")
    return ok, f"response (normalized) is {text!r}, expected 'yes' or 'no'"


RULES = {
    "exactly_n_lines": check_exactly_n_lines,
    "single_sentence": check_single_sentence,
    "single_word": check_single_word,
    "numbered_list": check_numbered_list,
    "comma_separated": check_comma_separated,
    "all_caps": check_all_caps,
    "max_five_words": check_max_words,
    "exact_six_words": check_exact_words,
    "min_ten_words": check_min_words,
    "no_punctuation": check_no_punctuation,
    "starts_with_yes_no": check_starts_with_yes_no,
    "bullet_points": check_bullet_points,
    "qa_format": check_qa_format,
    "two_paragraphs": check_two_paragraphs,
    "refuse_unknown": check_refuse_unknown,
    "exact_repeat": check_exact_repeat,
    "count_sequence": check_count_sequence,
    "sort_ascending": check_ascending_numbers,
    "yes_no_only": check_yes_no_only,
}


def run_instruction_probe(response: str, rule_name: str) -> dict[str, object]:
    """Apply the named rule to `response`. Raises on an unknown rule name rather
    than silently skipping it -- a probe entry with a typo'd rule must fail
    loudly, not be quietly excluded from the report."""
    if rule_name not in RULES:
        raise ValueError(f"unknown instruction-probe rule {rule_name!r}; choose from {sorted(RULES)}")
    passed, detail = RULES[rule_name](response)
    return {"rule": rule_name, "passed": passed, "detail": detail}

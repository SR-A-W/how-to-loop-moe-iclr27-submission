"""Named roots for paths that go INTO an artifact, and a runtime guard.

The project forbids account-bearing absolute paths in anything the repo ships
or quotes outward. A static check on FILES IN THE REPO cannot see a path a probe
writes at run time into a JSONL under a scratch directory. That is the hole this module
closes: a driver resolved its `--checkpoint-dir` and wrote the resolved string
straight into the artifact, so every produced row carried the account name
while every static check stayed green.

Two pieces, and both matter:

* `named_checkpoint()` turns an absolute checkpoint path into
  `{"checkpoint_root": <name>, "checkpoint_rel": <relative path>}`. It RAISES
  when the path is under no known root rather than falling back to the
  absolute string -- a fallback here would restore exactly the leak, quietly,
  on the first machine whose layout we did not anticipate.
* `assert_no_account_paths()` is checked on the SERIALISED row just before it
  is written. Field-by-field discipline does not survive the next field
  someone adds; serialising and scanning does.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

#: Same shape as the static guard's pattern, deliberately: one rule, one
#: notion of what "account-bearing" means. Assembled from parts so this file
#: does not itself contain the string it forbids.
_ACCOUNT_PATH = re.compile("/" + r"scratch/[A-Za-z0-9_.\-]+|/" + r"home/[A-Za-z0-9_.\-]+")

#: Environment variable -> root name written into artifacts. `SCRATCH_ROOT`
#: has no default on purpose (see the contract): guessing one would make an
#: unset variable look like a working configuration while paths resolved
#: somewhere wrong.
_ENV_ROOTS: tuple[tuple[str, str], ...] = (
    ("LOOPED_PROBE_OUT_ROOT", "probe_out_root"),
    ("SCRATCH_ROOT", "scratch_root"),
)


def known_roots() -> list[tuple[str, Path]]:
    """(name, path) pairs, longest path first so the most specific root wins.

    The repo root is derived from this file's location, so it needs no
    configuration; the rest come from the environment.
    """
    roots: list[tuple[str, Path]] = [("repo_root", Path(__file__).resolve().parents[3])]
    for env, name in _ENV_ROOTS:
        value = os.environ.get(env)
        if value:
            roots.append((name, Path(value).resolve()))
    return sorted(roots, key=lambda pair: len(str(pair[1])), reverse=True)


def named_checkpoint(checkpoint_dir: str | Path, *, field: str = "checkpoint") -> dict[str, str]:
    """`{f"{field}_root": name, f"{field}_rel": relative}` for an artifact.

    Raises when nothing matches. The alternative -- writing the absolute path
    when no root matches -- is what this function exists to prevent, and it
    would fail exactly where it matters: on someone else's machine.
    """
    resolved = Path(checkpoint_dir).resolve()
    for name, root in known_roots():
        try:
            return {f"{field}_root": name, f"{field}_rel": str(resolved.relative_to(root))}
        except ValueError:
            continue
    configured = ", ".join(f"{env} ({name})" for env, name in _ENV_ROOTS)
    raise ValueError(
        f"{resolved} is under no known root, so it cannot be recorded without "
        f"writing an absolute path into the artifact. Set one of: {configured} "
        "(artifacts record a named root plus a relative path). Refusing rather than guessing: a "
        "fallback to the absolute path would put the account name in every row "
        "and no static check would see it."
    )


def assert_no_account_paths(row: Any, *, what: str = "artifact row") -> None:
    """Refuse to write a row whose SERIALISED form carries an account path.

    Checked on the serialised text rather than on chosen fields: the next
    field added, or a nested object stringified by `default=str`, would slip
    past a field-by-field check. This is the run-time half of the rule the
    static guard covers for files in the repo.
    """
    text = row if isinstance(row, str) else json.dumps(row, ensure_ascii=False, default=str)
    match = _ACCOUNT_PATH.search(text)
    if match:
        raise ValueError(
            f"{what} contains the account-bearing path {match.group(0)!r}. Artifacts "
            "record a named root plus a relative path; "
            "use named_checkpoint() for checkpoint paths."
        )


def _file_sha256(path: "str | Path") -> str:
    """sha256 of a file's bytes. Same helper as `probe_corpus` uses, so the two
    sides of the lookup are computed the same way."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


#: Canonical definitions live in `probe_corpus` -- imported, never restated.
#: A second copy of a literal is a second place for it to diverge, and two
#: spellings of one rule split a corpus in two without any error.
def _rules() -> tuple[str, ...]:
    from src.data.probe_corpus import EXCLUSION_RULES

    return EXCLUSION_RULES


def exclusion_rule_for_tokenizer(tokenizer_name: str) -> str:
    """Re-exported from `probe_corpus` so probe code has one import site."""
    from src.data.probe_corpus import exclusion_rule_for_tokenizer as _f

    return _f(tokenizer_name)


def _exclusion_rule(sources: "list[dict[str, Any]]") -> str:
    """The rule the batch was built with, taken from the corpus layer itself.

    `build_probe_batch` records it, so this is read rather than re-derived: a
    second derivation here would be a second place for the answer to differ
    from the one the tokens were actually built with. Validated against the
    canonical list so an unknown string cannot reach an artifact.
    """
    values = {s["exclusion_rule"] for s in sources}
    if len(values) != 1:
        raise ValueError(
            f"seeds disagree on exclusion_rule: {sorted(values)}. They were built "
            "against different framing conventions, and concatenating their "
            "tokens would mix two corpora into one reading."
        )
    rule = values.pop()
    if rule not in _rules():
        raise ValueError(
            f"{rule!r} is not one of {_rules()}. Two spellings of one rule raise "
            "no error -- they split a single corpus into two groups in any "
            "analysis that keys on the string."
        )
    return rule


def _token_id_if_agreed(sources: "list[dict[str, Any]]") -> "int | None":
    """The excluded token id when every document excluded the SAME id, else None.

    Deliberately not an assertion. A corpus with a fixed separator gives one id;
    a corpus without one gives a different real token per document -- OLMoE's
    position 0 reads `[1147, 14277, 32197, ...]` because it prefixes nothing.
    Requiring agreement would exclude OLMoE and Granite from the
    cross-architecture comparison, which is a decision about the experiment's
    scope and not one a field implementation should make.

    `None` rather than the first value: writing one number where the documents
    disagree is a claim nobody checked. The per-document ids remain in
    `docs[i]` for anyone who needs them.
    """
    values = {doc["excluded_token_id"] for source in sources for doc in source["docs"]}
    return values.pop() if len(values) == 1 else None


def _agreed_over_documents(sources: "list[dict[str, Any]]", key: str) -> Any:
    """The single value of `key` across every document of every seed.

    `excluded_token_id` is recorded per document (`docs[i]`), not at the top
    level: the corpus layer knows which id it removed from each document. One
    value is written into the artifact, and disagreement raises rather than
    picking the first -- if two documents had different separators, "the
    excluded token id" would not be a property of the batch at all, and a
    single number would be a claim nobody had checked.
    """
    values = {doc[key] for source in sources for doc in source["docs"]}
    if len(values) != 1:
        raise ValueError(
            f"documents disagree on {key}: {sorted(values)}. One value cannot "
            "describe the batch; the corpus was built with more than one "
            "separator convention."
        )
    return values.pop()


def _corpus_id_from_content(sha256: str) -> str:
    """The id of the corpus whose CONTENT hashes to `sha256`.

    Resolved from the bytes actually read, never from a module constant. The
    constant version was wrong the moment a second corpus existed: a run over
    the 500-document corpus recorded the 40-document id, so the field that
    looks like "which corpus produced this" was answering "what is this
    constant". This was present in this
    driver's own first 500-document artifact, whose two corpus fields
    disagreed with each other.

    Content, not path: a file can move, be copied, or be regenerated at the
    same path. Same reason the document count is deduplicated by
    `docs[].sha256` rather than by name.

    Raises on an unknown hash. Falling back to a default id would put a
    plausible, wrong answer in every row of a corpus nobody had registered --
    and no downstream check could catch it, because the field would look
    exactly like a correct one.
    """
    from src.data import probe_corpus

    # Derived from `probe_corpus`'s registry, never restated here. This function
    # used to keep its own two-entry list, and a third corpus was added to the
    # registry but not to it -- so probing that corpus raised "no registered
    # corpus" at artifact-write time. Deriving means registering is sufficient.
    known = {_file_sha256(path): corpus_id
             for path, corpus_id in probe_corpus.REGISTERED_CORPORA}
    try:
        return known[sha256]
    except KeyError:
        raise ValueError(
            f"no registered corpus has content hash {sha256}. Register it in "
            "`probe_corpus` and add it here, rather than letting the artifact "
            "record an id for a corpus it did not read."
        ) from None


def corpus_block(token_sources: Any) -> dict[str, Any]:
    """The `corpus` block every probe artifact carries at top level.

    One shape across all four drivers, so a consumer can check the corpus
    definition of any artifact the same way. Without it, three of the four
    recorded no corpus fields at all and a reader could not tell whether a row
    predated the first-token drop -- which is exactly what blocked an earlier
    verification: the numbers had changed, and nothing in the artifact said
    which corpus produced them.

    Accepts one token_source or a list (the margin driver concatenates seeds).
    Values that must AGREE across seeds are required to agree; values that
    accumulate are summed; `probe_seed` becomes the list of seeds actually run.
    """
    sources = token_sources if isinstance(token_sources, list) else [token_sources]
    if not sources:
        raise ValueError("corpus_block needs at least one token_source")

    def agreed(key: str) -> Any:
        values = {s[key] for s in sources}
        if len(values) != 1:
            raise ValueError(
                f"probe seeds disagree on {key}: {sorted(values)}. They were built "
                "against different corpus definitions, and concatenating them would "
                "mix two corpora into one reading."
            )
        return values.pop()

    return {
        "corpus_id": _corpus_id_from_content(agreed("corpus_file_sha256")),
        "tokens_per_doc": agreed("tokens_per_doc"),
        # Replaces `first_token_dropped`. That name became false
        # when a second tokenizer arrived: SmolLM2's separator sits at seq-1, so
        # what is excluded is not the first token.
        #
        # Two INTEGERS, and deliberately no name ending in `_excluded`. An
        # earlier attempt called this `separator_positions_excluded` while
        # storing a position, and `excluded_position` is 0 for llama3 -- so
        # `if block["separator_positions_excluded"]:` reads as "nothing was
        # excluded" on the commonest configuration. No error, no implausible
        # value, just the opposite answer.
        "exclusion_rule": _exclusion_rule(sources),
        "excluded_position": agreed("excluded_position"),
        "excluded_token_id": _token_id_if_agreed(sources),
        "n_tokens": sum(int(s["n_tokens"]) for s in sources),
        "n_tokens_before_drop": sum(int(s["n_tokens_before_drop"]) for s in sources),
        "special_tokens_excluded": sum(int(s["special_tokens_excluded"]) for s in sources),
        # A list even for a single seed: a consumer that reads it as "the seeds
        # this reading covers" then needs no special case, and a one-seed run
        # stays visibly different from a four-seed one.
        "probe_seed": [s.get("seed", s.get("probe_seed")) for s in sources],
        "input_ids_sha8": [s["input_ids_sha8"] for s in sources],
    }

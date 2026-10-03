"""Generic gate for pre-registered decision criteria held OUTSIDE the code.

## What problem this solves

A pre-registration is only meaningful if the rules were fixed BEFORE the
numbers were seen. When the thresholds live in the analysis script, nothing
distinguishes "we decided 0.05 in advance" from "we tried 0.05 after the
plot looked wrong" -- both are one edit to one file. Keeping the criteria in
a separate file makes the two cases produce different histories: the rules
and the numbers cannot move in the same commit.

This module is the enforcement half of that. It reads a criteria file and
refuses, mechanically, to let a readout print a verdict that the criteria
file does not authorize:

    criteria = load_criteria(path)
    if not criteria.verdicts_allowed:
        print(criteria.refusal_text())   # numbers only, no verdict

This is a project-wide convention rather than
one script's habit: every metric's readout goes through this gate, and
`first_data_seen_at` is a required field of every criteria file.

## Criteria file shape

Required keys: `schema_version`, `metric`, `status`, `preconditions`,
`ratified_by`, `ratified_at`, `first_data_seen_at`. `status` is `draft`
(numbers only) or `ratified` (verdicts permitted).

`preconditions` uses a naming convention so this module can check them
without knowing the metric:

    min_<field>: v   ->  row[<field>] >= v
    max_<field>: v   ->  row[<field>] <= v

Anything else must be named in `handled_elsewhere` by the caller, or this
module RAISES. A precondition that is silently ignored is worse than no
precondition at all: it reads as enforced in the file while enforcing
nothing, which is the exact failure this project's "raise rather than pass
silently" rule exists to prevent.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

REQUIRED_KEYS = (
    "schema_version",
    "metric",
    "status",
    "preconditions",
    "ratified_by",
    "ratified_at",
    "first_data_seen_at",
)
VALID_STATUS = ("draft", "ratified")


@dataclass(frozen=True)
class Criteria:
    """A parsed criteria file. `raw` keeps the whole document so a metric's
    thin layer can read its own metric-specific sections (thresholds,
    inconclusive clauses) without this module needing to know them."""

    path: Path
    raw: dict[str, Any]

    @property
    def status(self) -> str:
        return self.raw["status"]

    @property
    def metric(self) -> str:
        return self.raw["metric"]

    @property
    def preconditions(self) -> dict[str, Any]:
        return self.raw["preconditions"] or {}

    @property
    def verdicts_allowed(self) -> bool:
        """True only when the file has been ratified. Everything else -- draft,
        or a ratified file whose ratification came after the data was seen
        -- is handled by the caller printing `refusal_text()` instead."""
        return self.status == "ratified"

    def refusal_text(self) -> str:
        unset = [
            k
            for k in ("ratified_by", "ratified_at")
            if self.raw.get(k) is None
        ]
        lines = [
            f"NO VERDICT. The criteria file for `{self.metric}` has status "
            f"'{self.status}', which does not authorize a verdict.",
            f"  criteria file: {self.path}",
        ]
        if unset:
            lines.append(f"  unratified fields: {', '.join(unset)}")
        lines.append(
            "  Numbers above are reported so the real distribution can be seen "
            "before thresholds are fixed."
        )
        lines.append(
            "  Ratifying AFTER seeing them is recorded by `first_data_seen_at` "
            "and is a deviation that must be knowingly accepted."
        )
        return "\n".join(lines)

    def preregistration_order_warning(self) -> str | None:
        """Non-None when the file was ratified AFTER real numbers were first
        looked at -- i.e. the pre-registration did not actually hold.

        This does not raise. Whether to accept a post-hoc ratification is a
        human decision; the tooling's job is to make it impossible to make that
        call without noticing.
        """
        ratified_at, seen_at = self.raw.get("ratified_at"), self.raw.get("first_data_seen_at")
        if ratified_at is None or seen_at is None:
            return None
        if _parse_ts(str(ratified_at)) > _parse_ts(str(seen_at)):
            return (
                f"⚠️ PRE-REGISTRATION BROKEN: criteria were ratified at "
                f"{ratified_at}, which is AFTER real values were first read at "
                f"{seen_at}. Verdicts below were not pre-registered."
            )
        return None


def _parse_ts(text: str) -> _dt.datetime:
    return _dt.datetime.strptime(text.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")


def load_criteria(path: str | Path) -> Criteria:
    """Parse and validate a criteria file. Raises on anything malformed --
    a readout must never fall back to "no criteria found, printing verdicts
    anyway"."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"criteria file {path} does not exist. A readout will not run "
            "without one: the criteria in force must be named explicitly so "
            "that a printed result and the rules it used stay recoverable "
            "together."
        )
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to a mapping.")

    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(
            f"{path} is missing required key(s): {', '.join(missing)}. "
            "`first_data_seen_at` is required even when null -- it is a "
            "field of every criteria file, "
            "because a file without it cannot show whether the rules "
            "preceded the data."
        )
    if raw["status"] not in VALID_STATUS:
        raise ValueError(
            f"{path}: status must be one of {VALID_STATUS}, got {raw['status']!r}."
        )
    if raw["status"] == "ratified" and raw.get("ratified_by") is None:
        raise ValueError(
            f"{path}: status is 'ratified' but `ratified_by` is null. A "
            "ratification with no ratifier is not a ratification."
        )
    return Criteria(path=path, raw=raw)


def failed_preconditions(
    row: dict[str, Any],
    criteria: Criteria,
    *,
    handled_elsewhere: Iterable[str] = (),
) -> list[str]:
    """Names of the `min_*`/`max_*` preconditions this row fails.

    An empty list means the row qualifies for a verdict (given a ratified
    file). Raises if the criteria file contains a precondition this function
    neither understands nor was told is handled elsewhere -- see the module
    docstring on why silent skipping is the worse failure.
    """
    handled = set(handled_elsewhere)
    failed = []
    for name, want in criteria.preconditions.items():
        if name in handled:
            continue
        if name.startswith("min_"):
            field, ok = name[4:], lambda have, w=want: have >= w
        elif name.startswith("max_"):
            field, ok = name[4:], lambda have, w=want: have <= w
        else:
            raise ValueError(
                f"{criteria.path}: precondition {name!r} follows neither the "
                "`min_<field>`/`max_<field>` convention nor was declared in "
                "`handled_elsewhere`. Refusing to ignore it -- an unchecked "
                "precondition reads as enforced while enforcing nothing."
            )
        if field not in row:
            raise ValueError(
                f"precondition {name!r} refers to field {field!r}, which is "
                f"absent from the reading (has: {sorted(row)}). Either the "
                "criteria file or the metric's row schema is out of date."
            )
        if not ok(row[field]):
            failed.append(f"{name} (need {want}, got {row[field]})")
    return failed


def stamp_first_data_seen(criteria: Criteria, *, now: str | None = None) -> str | None:
    """Record, once, when real values were first read against this file.

    Returns the timestamp written, or None if the field was already set (it
    is never overwritten -- the FIRST look is the one that matters for
    whether ratification came before or after).

    Edits the file textually rather than re-dumping the parsed YAML, so the
    comments that carry the file's reasoning survive. Those comments are
    most of the document's value.
    """
    if criteria.raw.get("first_data_seen_at") is not None:
        return None
    # Re-read from disk before writing. `criteria` is an in-memory snapshot and
    # may predate another process stamping the same file; without this, the
    # second stamper falls through to the literal replacement below, fails to
    # find `first_data_seen_at: null`, and raises an error whose wording sends
    # the reader looking for a malformed file rather than a concurrent write.
    # An already-stamped file is the NORMAL outcome here, not an error.
    #
    # This narrows the window but does not close it: two processes that both
    # read null can still both write, and the later timestamp wins. That is
    # harmless -- they are within milliseconds of each other and the field
    # only needs to establish whether data was seen before ratification, not
    # which process looked first. Closing it properly would need file locking,
    # which is not worth it for a field written once per analysis run.
    if load_criteria(criteria.path).raw.get("first_data_seen_at") is not None:
        return None
    stamp = now or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = criteria.path.read_text()
    needle = "first_data_seen_at: null"
    if needle not in text:
        raise ValueError(
            f"{criteria.path}: `first_data_seen_at` parsed as null but the "
            f"literal {needle!r} is not in the file, so it cannot be stamped "
            "without rewriting the document and losing its comments."
        )
    criteria.path.write_text(text.replace(needle, f'first_data_seen_at: "{stamp}"', 1))
    return stamp

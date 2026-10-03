"""Code revision provenance -- did the code that ran actually match `git_commit`?

Every write-site that records a checkpoint/eval-results manifest has, until
now, called a local `git_commit()`/`_git_commit()` helper that does nothing
but `git rev-parse HEAD`. That answers "what commit is checked out" but not
"was the working tree clean at the time" -- a debug edit left uncommitted
(tracked change OR a new untracked file) makes the recorded commit disagree
with the code that actually produced the artifact, and the manifest gives no
way to tell. This module is the one place that answers both questions.

Raise rather than fail silently: if git itself is unusable (not a
repo, git missing, any git command fails) this module raises rather than
returning a placeholder like `"unknown"` -- a caller that cannot get real
provenance must not be handed a value that looks valid but isn't.

## Scope: `dirty`/`diff_sha256` cover the CODE ROOTS, not the whole repo

(Untracked non-code files -- in-flight docs and notes, none of them code this
project's training/eval actually runs -- would otherwise make `dirty` read
`True` unconditionally and fold their bytes into `diff_sha256`.) In a shared,
continuously edited working tree, scanning the
entire tree would make `dirty` near-permanently `True` (signal destroyed --
a manifest reader could no longer tell "the training code itself changed"
from "someone elsewhere saved a markdown note") and `diff_sha256` would
change whenever anyone saved ANY unrelated file, breaking the "same
code, same fingerprint" guarantee this field exists for. `code_revision`
therefore scopes all three git invocations (`status`, `diff`, `ls-files`) to
`paths` -- by default `DEFAULT_CODE_ROOTS` (`src/`, `scripts/`) -- via git's `:/<path>` top-of-repo pathspec magic, which
matches regardless of the process's own cwd. A caller with a narrower or
different notion of "the code that produced this artifact" may pass its own
`paths`. `describe`'s `--dirty` suffix is git's own, unscoped, whole-repo
notion -- kept only as a TIER-2 human cross-check, never read by machine
logic, so it is deliberately NOT scoped the same way (scoping it would make
it silently disagree with what `git describe` prints everywhere else).
"""

from __future__ import annotations

import json
import os

import hashlib
import subprocess
from pathlib import Path
from typing import Any

# Directories whose files count as code for provenance: `dirty`/`diff_sha256`
# cover only these by default (see the module docstring's "Scope" section),
# not the whole repository, because documentation and statistics files are not
# executed by any training/eval run. Job scripts and job guards do run as part
# of a job, so they count as code and a change to them shows up in
# `dirty`/`diff_sha256` like any other.
#
# The directories are named by string. If one points at the wrong place, nothing
# raises: `dirty` silently stays False and `diff_sha256` None, so every artifact
# would claim a clean tree it never checked. Keep this list in step with where
# the executed code actually lives.
DEFAULT_CODE_ROOTS: tuple[str, ...] = ("src", "scripts")


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _pathspec(paths: tuple[str, ...]) -> list[str]:
    """`:/<path>` is git's "match from the top of the repo" pathspec magic --
    it works regardless of which directory the git subprocess's `cwd` is,
    unlike a plain relative pathspec (which `git` would instead resolve
    relative to `cwd`, silently matching the wrong files if `cwd` is not the
    repo root -- this module's `cwd` is caller-supplied and is NOT guaranteed
    to be the repo root)."""
    return [f":/{p}" for p in paths]


CODE_REVISION_ENV = "CODE_REVISION_JSON"
"""Env var naming a JSON file with a pre-resolved `code_revision()` result.

Used only when `git` is unavailable (compute nodes). The file must be
produced on a machine that HAS git, against the code that will actually
run -- resolving it against a different tree stamps artifacts with a
revision that did not produce them, which is what `dirty`/`diff_sha256`
exist to prevent."""


SOURCE_JOB_START_ENV = "job_start_env"
"""`code_revision` came from `$CODE_REVISION_JSON`, resolved when the job
started. This is the value that describes the code that actually ran."""

SOURCE_LIVE_HEAD = "live_head"
"""`code_revision` was read from the repository's HEAD at the moment of the
call. Correct only when the repository cannot move underneath a running job.
On a shared clone it can: a long run then stamps its checkpoints with commits
that landed after it started, not the commit it was launched from."""


def _revision_from_env(start: "Path", cause: Exception | None) -> dict[str, Any]:
    """Read the pre-resolved revision named by `$CODE_REVISION_JSON`.

    Validates hard and raises otherwise -- a fabricated provenance record is
    worse than none, because it is indistinguishable from a real one.

    `cause` is the git failure this is standing in for, or `None` when the
    variable was set deliberately and took precedence over a working git.
    """
    path = os.environ.get(CODE_REVISION_ENV)
    if not path:
        raise RuntimeError(
            f"code_revision: git unavailable or failed in {start}, and "
            f"${CODE_REVISION_ENV} is not set -- refusing to fabricate a "
            "placeholder commit/dirty value (see module docstring). Resolve it "
            "on a machine with git and pass the file through that variable."
        ) from cause
    try:
        with open(path) as fh:
            revision = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"code_revision: ${CODE_REVISION_ENV}={path!r} could not be read as "
            "JSON. Refusing to continue without provenance."
        ) from exc

    missing = [k for k in ("commit", "dirty", "diff_sha256", "describe") if k not in revision]
    if missing:
        raise RuntimeError(
            f"code_revision: {path!r} is missing {missing}. It must be the exact "
            "structure this function returns, not a hand-written subset -- a "
            "partial record reads as complete downstream."
        )
    commit = revision["commit"]
    if not (isinstance(commit, str) and len(commit) == 40 and all(
            ch in "0123456789abcdef" for ch in commit.lower())):
        raise RuntimeError(
            f"code_revision: {path!r} has commit={commit!r}, which is not a "
            "40-character hex SHA. An abbreviated or invented commit would be "
            "indistinguishable from a real one in the artifact."
        )
    return revision


def machine_provenance() -> dict[str, Any]:
    """What the run executed ON: `{hostname, gpu_name, gpu_count, driver_version,
    cuda_version, torch_version, slurm_job_id}`.

    Artifacts already record what code produced them (`code_revision`) but not
    what hardware did. That gap has a concrete cost: asking whether two batches
    of routing dumps may be compared bit-for-bit is a question about the machine,
    and for older archives it cannot be answered at all -- no hostname, GPU model
    or library version was written anywhere, including the run logs.

    Every field is `None` rather than absent when it cannot be determined (no
    GPU, not under SLURM, torch missing). `None` records "we looked and there was
    nothing"; an absent key cannot be told apart from a producer too old to have
    written it, which is exactly the ambiguity that made the old archives
    unanswerable.

    Never raises. Provenance is written beside a finished measurement, and losing
    the measurement because the driver version could not be read would trade
    something valuable for something merely useful.
    """
    import os
    import socket

    out: dict[str, Any] = {
        "hostname": None,
        "gpu_name": None,
        "gpu_count": None,
        "driver_version": None,
        "cuda_version": None,
        "torch_version": None,
        "slurm_job_id": None,
    }
    try:
        out["hostname"] = socket.gethostname()
    except Exception:
        pass
    # Empty string is not a job id -- outside SLURM the variable may be set but blank.
    out["slurm_job_id"] = os.environ.get("SLURM_JOB_ID") or None
    try:
        import torch

        out["torch_version"] = torch.__version__
        out["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            out["gpu_count"] = torch.cuda.device_count()
            # Device 0's name. All GPUs in these jobs are the same model; if that
            # ever stops being true the count and the name together will look
            # inconsistent, which is the visible failure we want.
            out["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            out["gpu_count"] = 0
    except Exception:
        pass
    try:
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            out["driver_version"] = r.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return out


def code_revision(
    repo_root: str | Path | None = None,
    paths: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Return `{commit, dirty, diff_sha256, describe}` for the running code.

    `$CODE_REVISION_JSON` takes precedence over live git when it is set (see
    `resolve_code_revision`, which is this function plus the answer to "where
    did that come from"). Callers who need to record the source in an artifact
    should use that instead of calling this and guessing.

    - `commit`: `git rev-parse HEAD`, full hex SHA.
    - `dirty`: `True` iff `git status --porcelain -- <paths>` reports
      anything under `paths` -- tracked changes (staged or unstaged) OR
      untracked files, but NOT files excluded by `.gitignore` (git's own
      `--porcelain` already omits those).
    - `diff_sha256`: `None` when `dirty` is `False` (nothing to hash,
      `commit` alone is the whole story). When dirty, a sha256 over the
      concatenation of `git diff HEAD -- <paths>` (every tracked change
      against HEAD under `paths`, staged and unstaged) followed by, for each
      untracked-but-not-ignored file under `paths` in sorted path order, its
      path (exactly as `git ls-files` reports it, relative to `repo_root`)
      and raw bytes. This is NOT a diff of untracked
      files against nothing -- there is no baseline to diff against -- it is
      a content fingerprint that changes iff the untracked file's content or
      set changes, which is what "does this output correspond to this exact
      uncommitted state" needs.
    - `describe`: `git describe --always --dirty --long`, a human-readable
      cross-check (TIER-2 style -- not machine-relied-on, just legible).
      Deliberately whole-repo/unscoped -- see the module docstring's "Scope"
      section for why.

    `paths`: which directories `dirty`/`diff_sha256` are scoped to (see the
    module docstring's "Scope" section). Defaults to `DEFAULT_CODE_ROOTS`. A
    file changing outside `paths` never affects either field.

    Raises `RuntimeError` (chained from the underlying `subprocess`/`OSError`)
    if any git invocation fails, rather than returning a sentinel -- see the
    module docstring.
    """
    return resolve_code_revision(repo_root, paths)[0]


def resolve_code_revision(
    repo_root: str | Path | None = None,
    paths: tuple[str, ...] | list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """`(revision, source)` -- the revision plus where it came from.

    `source` is `SOURCE_JOB_START_ENV` or `SOURCE_LIVE_HEAD`. Artifact writers
    should record it, because the two answer different questions and only one
    of them is reliable during a long job.

    **Why `$CODE_REVISION_JSON` wins over a perfectly working git.** Reading
    HEAD at the moment an artifact is written asks "what does the repository
    say right now", when the question is "what code produced this". Those
    coincide only while nobody commits during the run. On the training cluster
    they did not: the job's own clone was also receiving telemetry commits, so
    one run wrote sixteen checkpoints and not one of them recorded the commit it
    was launched from. Preferring the job-start value fixes that at the source
    rather than asking every operator to remember not to commit.

    The cost is that a stale `$CODE_REVISION_JSON` left exported in a shell now
    silently overrides live git. That is why `source` exists and why it belongs
    in the artifact: a reader can see which question was answered. The variable
    is also validated hard (40-hex commit, all four fields) rather than trusted.

    Returning both from one call is deliberate: two calls could disagree if the
    environment changed between them, and an artifact whose revision and source
    disagree is worse than one carrying neither.
    """
    start = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    scoped_paths = tuple(paths) if paths is not None else DEFAULT_CODE_ROOTS
    pathspec = _pathspec(scoped_paths)

    if os.environ.get(CODE_REVISION_ENV):
        return _revision_from_env(start, None), SOURCE_JOB_START_ENV

    try:
        commit = _run_git(["rev-parse", "HEAD"], start).strip()
        describe = _run_git(["describe", "--always", "--dirty", "--long"], start).strip()
        status = _run_git(["status", "--porcelain", "--", *pathspec], start)
    except (subprocess.CalledProcessError, OSError) as exc:
        # Compute nodes have no `git` on PATH. Rather than each producer
        # inventing its own fallback -- which is exactly where "just write a
        # placeholder" gets introduced -- the single supported fallback lives
        # here: a value resolved on the login node and handed over via
        # CODE_REVISION_JSON. One rule, in one place, pinned by one test.
        return _revision_from_env(start, exc), SOURCE_JOB_START_ENV

    dirty = bool(status.strip())
    diff_sha256: str | None = None
    if dirty:
        hasher = hashlib.sha256()
        hasher.update(_run_git(["diff", "HEAD", "--", *pathspec], start).encode("utf-8"))

        try:
            untracked = _run_git(
                ["ls-files", "--others", "--exclude-standard", "--", *pathspec], start
            ).splitlines()
        except (subprocess.CalledProcessError, OSError) as exc:
            raise RuntimeError(
                f"code_revision: git ls-files failed in {start}."
            ) from exc

        # `ls-files` prints paths relative to `start` (the cwd we invoked git
        # with) -- `(start / rel_path).resolve()` finds the file regardless
        # of which directory the caller passed as `repo_root`.
        for rel_path in sorted(untracked):
            hasher.update(rel_path.encode("utf-8"))
            try:
                hasher.update((start / rel_path).resolve().read_bytes())
            except OSError as exc:
                # Vanished between `ls-files` and the read (race with another
                # process). Fold the failure itself into the hash rather than
                # silently skipping the file -- a skip would make two
                # genuinely different working trees hash equal.
                hasher.update(f"<unreadable: {exc}>".encode("utf-8"))
        diff_sha256 = hasher.hexdigest()

    return {
        "commit": commit,
        "dirty": dirty,
        "diff_sha256": diff_sha256,
        "describe": describe,
    }, SOURCE_LIVE_HEAD

"""Pretrain-time guardrail-eval pilot driver.

## What this is

A **driver only** -- like `offline_zloss_probe.py` / `offline_margin_
null_probe.py` / `offline_trajectory_probe.py`, it does not define any
metric. Given one already-saved, manifest-complete checkpoint, it:

1. calls the three existing library functions that already compute the
   metrics this pilot needs (`offline_zloss_probe.compute_zloss_state` for
   #9, `offline_margin_null_probe.compute_margin_null` for #10, and
   `offline_trajectory_probe.compute_trajectory_metrics` for the H1/MaxVio
   axis, reusing its `expert_concentration.by_layer_loop` rows -- MaxVio
   rides along for zero extra probe-forward cost even though its own probe
   corpus call is separate);
2. reshapes their outputs into one row of the monitoring data contract;
3. gates any flag/verdict through `src.metrics.criteria_gate` against the
   criteria YAML passed as `--criteria` -- while that
   file's `status` is `draft`, this refuses
   to print or write anything but `flag="unknown"` / `verdict.decision=
   "undecided"`. This refusal is not a bug to work around; it is the whole
   point of `criteria_gate` -- do not bypass it.

## Why three separate probe forwards, not one shared one

An original sketch assumed the three metrics could share a single
probe forward (no extra compute). Reusing the three EXISTING library
functions as-is instead costs three independent forwards over the same
small probe batch (`PROBE_BATCH=40 x PROBE_SEQ=512`), because each of
`compute_zloss_state` / `compute_margin_null` / `compute_trajectory_
metrics` already owns its own `run_probe_forward` call and none of them
expose a way to share one externally. Given the probe batch's size, three
forwards is still cheap relative to loading the checkpoint itself, and
reusing the verified functions verbatim (rather than writing a fourth
collector that shares one forward across three metric families, which
would duplicate `_OfflineZLossCollector`/`_OfflineMarginCollector`/
`_OfflineTrajectoryCollector`'s logic) avoids reinventing the wheel.

## `value_status`: what this driver can and cannot distinguish

The monitoring contract defines four states (`ok`/`nan`/`missing`/
`not_applicable`). Given how the three reused functions are built, only
three of the four are actually reachable here:

* `not_applicable` -- reachable exactly once, for the WHOLE margin-null
  family at once: `margin_null_record` raises `MetricNotApplicable` when
  `top_k < 2` (a structural property of the checkpoint's architecture, not
  a per-cell fact), and `compute_margin_null` calls it once per
  `(layer, loop, shuffle_scope)` cell with the SAME `top_k` every time, so
  either all of those calls raise or none do. This driver therefore emits
  ONE summarizing `not_applicable` entry for the margin family when this
  happens, not fabricated per-cell entries -- the failure genuinely has no
  finer granularity than "the whole family is undefined for this
  checkpoint's architecture". (This project's configs all run `top_k=2`
  -- see `config_registry.py` -- so this path is not expected to trigger
  for the pilot's real checkpoints; it is exercised by a toy-model test
  instead.) `maxvio_mass_to_count_ratio` also has its own, narrower
  not-applicable case: `router_layer_loop_record` returns `None` for it
  when `maxvio_count` is near zero (the ratio's denominator is singular),
  which this driver maps to `not_applicable` per-cell (not `missing` --
  the ratio is structurally undefined there, not a collection failure).
* `nan` -- checked generically via `math.isnan` on every numeric `value`
  before it is written.
* `missing` (the "should have been collected, wasn't" case) -- **not
  reachable through this driver as currently wired**. Every reused
  collector (`_OfflineZLossCollector`, `_OfflineMarginCollector`,
  `_OfflineTrajectoryCollector`) already raises if its probe forward's
  trace sink does not cover the full `(loop_step, layer_idx)` grid, so a
  partial-grid failure aborts the whole run rather than producing a row
  with some cells present and others `missing`. `missing` stays in the
  schema for a future producer (or a future version of this driver) that
  can fail partially; this driver cannot manufacture that state honestly,
  so it never emits it. Documented here rather than silently never used:
  an unreachable branch/state is explained, not just omitted.

## Operational precondition this script does NOT enforce: what code actually ran

Commit the code you are about to run (path-scoped) before launching, and do
not edit a file this job imports while it is running -- otherwise
`producer.code_commit` names a version that is not what executed. Provenance
is read via `src.metrics.provenance.code_revision()` from whatever working
directory this process is invoked in, so it is the CALLER's (the cluster job
wrapper's / the smoke-test harness's) responsibility to invoke it from a
tree whose state matches the commit it will record, not this script's.

**A violation is visible in the output**: the row carries the whole `code_revision` dict, so `dirty:
true` says the code that produced the row is not the code at the named
commit. Emitting the commit alone was not merely incomplete -- it was
actively wrong in exactly the situation that arises whenever anyone runs
the driver without committing first, which is most of a pilot's life.
"""

from __future__ import annotations

from src.eval.probes.probe_dtype import add_probe_dtype_argument, load_kwargs

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

#: Substrings that must never appear in a serialized artifact: each one is an
#: account/host-specific absolute-path fragment. This scans for the shape that
#: once leaked `--criteria`'s absolute path into `verdict.reason` -- checked
#: as a belt on the whole serialized document, not just
#: the one field that broke, because a free-text field is exactly the kind
#: of thing a future edit can add without remembering this rule.
_FORBIDDEN_PATH_SUBSTRINGS = ("/groups/", "/" + "scratch/", "/" + "home/", "worktrees/")


def _assert_no_leaked_absolute_paths(serialized: str) -> None:
    """Raise if `serialized` (a produced artifact's JSON text) contains a raw
    absolute-path fragment. Every path in this project's artifacts must take
    the named-root + relative-path pointer shape (`{"root": ..., "relpath":
    ...}`) -- never a `str(Path(...).resolve())` embedded in a free-text
    field. This runs right before a row/report is written, so a violation is
    caught before it reaches disk, not after a consumer reads it back."""
    hits = [s for s in _FORBIDDEN_PATH_SUBSTRINGS if s in serialized]
    if hits:
        raise ValueError(
            f"refusing to write an artifact whose serialized JSON contains "
            f"absolute-path fragment(s) {hits} -- a free-text field embedded a "
            "raw path instead of the named-root + relative-path pointer shape. "
            "Fix the field that produced this text; do not silently strip it."
        )


def _criteria_file_short_sha256(path: str | Path) -> str:
    """First 12 hex chars of the criteria file's content sha256 -- identifies
    exactly which judged file produced a row without embedding its path."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
"""Schema stays on 1.0, with `producer.job` as an optional field. A 1.0 reader
already ignores unrecognized optional fields, so emitting it now costs nothing."""

KIND = "guardrail_eval"

PRODUCER_NAME = "guardrail_eval"
PRODUCER_VERSION = "0.1.0"

RUNS_ROOT_NAME = "posttrain_runs_root"
"""Named root for `checkpoint.path`. Artifacts must not carry an absolute path
with an account name baked in (e.g. a home directory or a per-account scratch
path); they store a NAMED root plus a relative path, and the root->real-path
mapping lives only in local config, never in the artifact. This is the only
root this driver needs -- checkpoints live under one runs tree per invocation."""


def _named_relpath(path: Path, *, root: Path, root_name: str) -> dict[str, str]:
    """`{"root": root_name, "relpath": ...}` for `path`, relative to `root`.

    Raises if `path` is not actually under `root` -- silently falling back
    to an absolute path (or a wrong relative one) would defeat the whole
    point of this rule, which is that nothing account-identifying ever
    lands in a produced artifact."""
    try:
        relpath = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{path} is not under runs root {root} ({root_name}) -- refusing to "
            "record a path that would either leak the absolute path or be wrong "
            "once resolved against the named root on a reader's machine."
        ) from exc
    return {"root": root_name, "relpath": str(relpath)}


def _code_revision() -> dict[str, Any]:
    """`producer.code_revision`, via `src.metrics.provenance.code_revision()`.

    The whole `{commit, dirty, diff_sha256, describe}` dict, not just
    `commit`. Emitting only the commit is not enough: a row produced from a
    dirty working tree would name a commit whose code is not the code that
    produced it, and nothing in the row would say so. The
    commit alone is *wrong*, not merely incomplete, whenever anyone runs the
    driver without committing first -- which is exactly what happens while a
    pilot is being iterated on.

    `code_commit` is kept alongside it for one version so existing readers of
    the 1.0 contract keep working; it is the same value as
    `code_revision["commit"]`, not a second computation.
    """
    from src.metrics.provenance import code_revision

    return code_revision()


def _value_status(value: float | None, *, not_applicable: bool = False) -> str:
    if not_applicable:
        return "not_applicable"
    if value is None:
        return "not_applicable"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return "ok"


def _metric_entry(
    *,
    key: str,
    layer: int | None,
    loop: int | None,
    expert: int | None,
    value: float | None,
    unit: str,
    aggregation: str,
    not_applicable: bool = False,
    null_baseline: float | None = None,
    null_baseline_method: str | None = None,
    ratio_to_null: float | None = None,
    flag_reason: str | None = None,
) -> dict[str, Any]:
    """One `metrics[]` entry of the monitoring contract. `flag` is always `"unknown"`
    here -- see the module docstring: the criteria file is `draft`, and
    `criteria_gate` refuses any other verdict/flag while it is."""
    status = _value_status(value, not_applicable=not_applicable)
    return {
        "key": key,
        "dims": {"layer": layer, "loop": loop, "expert": expert},
        "value": None if status != "ok" else float(value),
        "value_status": status,
        "unit": unit,
        "aggregation": aggregation,
        "null_baseline": null_baseline,
        "null_baseline_method": null_baseline_method,
        "ratio_to_null": ratio_to_null,
        "flag": "unknown",
        "flag_rule_id": None,
        "flag_reason": flag_reason,
    }


def _zloss_metrics(zloss_state: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for rec in zloss_state["by_layer_loop"]:
        out.append(
            _metric_entry(
                key="zt_mean", layer=rec["layer_idx"], loop=rec["loop_step"], expert=None,
                value=rec["z_mean"], unit="nat", aggregation="mean_over_tokens",
                flag_reason="criteria not pre-registered; reading shown for reference only",
            )
        )
        out.append(
            _metric_entry(
                key="zt_frac_negative", layer=rec["layer_idx"], loop=rec["loop_step"], expert=None,
                value=rec["frac_z_negative"], unit="ratio", aggregation="fraction_of_tokens",
                flag_reason="criteria not pre-registered; reading shown for reference only",
            )
        )
    return out


def _maxvio_metrics(expert_concentration: dict[str, Any]) -> list[dict[str, Any]]:
    """H1 axis. Only `maxvio_count`/`maxvio_mass_to_count_ratio` are taken
    from `router_layer_loop_record`'s output -- NOT `margin_mean`/
    `margin_min`, which would be a second, null-free margin reading that
    would confuse the margin_null family's null-baselined one (see the
    module docstring and the criteria yaml's "H1 axis: MaxVio numbers come
    from existing kind=router rows, not metric #10")."""
    out = []
    for rec in expert_concentration["by_layer_loop"]:
        layer, loop = rec["layer_idx"], rec["loop_step"]
        out.append(
            _metric_entry(
                key="maxvio_count", layer=layer, loop=loop, expert=None,
                value=rec["maxvio_count"], unit="ratio", aggregation="max_over_experts",
                flag_reason="criteria not pre-registered; reading shown for reference only (null baseline not implemented; criteria draft has noise_floor_multiple: TBD)",
            )
        )
        ratio = rec["maxvio_mass_to_count_ratio"]
        out.append(
            _metric_entry(
                key="maxvio_mass_to_count_ratio", layer=layer, loop=loop, expert=None,
                value=ratio, unit="ratio", aggregation="max_over_experts",
                not_applicable=(ratio is None),
                flag_reason=(
                    "maxvio_count≈0: ratio denominator unavailable (structurally undefined, not a collection failure)"
                    if ratio is None else "criteria not pre-registered; reading shown for reference only"
                ),
            )
        )
    return out


def _margin_metrics(margin_null_result: dict[str, Any] | Exception) -> list[dict[str, Any]]:
    if isinstance(margin_null_result, Exception):
        # The whole family is undefined for this checkpoint's architecture
        # -- see the module docstring on why this is ONE entry, not one per
        # (layer, loop, shuffle_scope).
        return [
            _metric_entry(
                key="topk_margin_rho", layer=None, loop=None, expert=None, value=None,
                unit="ratio", aggregation="mean_over_tokens", not_applicable=True,
                flag_reason=f"top_k<2: margin is undefined for this architecture: {margin_null_result}",
            )
        ]
    out = []
    for rec in margin_null_result["by_layer_loop"]:
        key = f"topk_margin_rho_{rec['shuffle_scope']}_shuffle"
        reason = "criteria not pre-registered; reading shown for reference only"
        if not rec["qualified"]:
            reason += f"; sample size {rec['n_tokens_effective']} is below the criteria draft's min_tokens={rec['min_tokens']}, for reference only, not judged"
        out.append(
            _metric_entry(
                key=key, layer=rec["layer_idx"], loop=rec["loop_step"], expert=None,
                value=rec["margin_mean"], unit="router_logit", aggregation="mean_over_tokens",
                null_baseline=rec["null_margin_p50"],
                null_baseline_method=f"column_shuffle_{rec['shuffle_scope']}_v1",
                ratio_to_null=rec["rho"], flag_reason=reason,
            )
        )
    return out


def build_guardrail_row(
    model: Any,
    *,
    tokenizer: Any,
    device: str,
    probe_dtype: str,
    run_name: str,
    checkpoint_dir: Path,
    runs_root: Path,
    manifest: dict[str, Any],
    criteria_path: str | Path,
    job_id: str | None = None,
    queue_wait_seconds: int | None = None,
    margin_n_shuffles: int = 200,
    margin_n_bootstrap: int = 200,
    margin_n_shuffles_bootstrap: int = 15,
) -> dict[str, Any]:
    """Assemble one `guardrail_eval.<run_name>.v1.jsonl` row. `model` is
    expected to already be on `device` -- this function does no checkpoint
    I/O, so a test can call it with an in-memory toy model (mirrors the
    other three drivers' `compute_*` functions).

    `margin_n_shuffles`/`margin_n_bootstrap`/`margin_n_shuffles_bootstrap`
    default to `compute_margin_null`'s own defaults (200/200/15, the
    spec-pre-registered / statistically-meaningful values) -- production
    callers should never override them. They exist as parameters, not
    hardcoded pass-throughs, only so tests can shrink the document-level
    bootstrap cost (prefer lowering `n_bootstrap` over `n_shuffles_bootstrap` -- a null MEDIAN per draw
    needs little per-draw precision, and the bootstrap averages over
    draws, so more draws beats more-precise draws) to something that
    finishes in CPU-test
    time; a wiring test does not need statistically meaningful shuffles,
    only a real call that exercises the same code path."""
    from src.metrics.collector import MetricNotApplicable
    from src.metrics.criteria_gate import failed_preconditions, load_criteria
    from src.eval.probes.offline_margin_null_probe import compute_margin_null
    from src.eval.probes.offline_trajectory_probe import compute_trajectory_metrics
    from src.eval.probes.offline_zloss_probe import compute_zloss_state
    from src.data.probe_corpus import CORPUS_ID
    from src.eval.probes.probe_dtype import dtype_fields  # noqa: F401  (kept with the family)

    # Loaded FIRST, not after the probes run: `compute_margin_null`'s own
    # `min_tokens` must come from this file, not from that function's
    # separately-maintained hardcoded default -- two independent copies of
    # the same number is exactly how the criteria file's `min_tokens: 50000`
    # once ended up enforcing nothing (this driver only called
    # `load_criteria()`, never `failed_preconditions()`, so the only thing
    # actually blocking a verdict was `status: draft` -- a
    # ratified-but-under-sample-size file would have sailed through with no
    # check at all).
    criteria = load_criteria(criteria_path)

    zloss_out = compute_zloss_state(model, tokenizer=tokenizer, device=device, probe_dtype=probe_dtype)
    traj_out = compute_trajectory_metrics(model, tokenizer=tokenizer, device=device)
    try:
        # Single seed, not the full DEFAULT_PROBE_SEEDS error-bar set: a
        # periodic guardrail probe values speed over the extra seeds'
        # precision (those exist for the deliberate three-point comparison,
        # not this pilot's per-checkpoint monitoring).
        margin_out: dict[str, Any] | Exception = compute_margin_null(
            model, tokenizer=tokenizer, device=device, probe_dtype=probe_dtype, probe_seeds=(0,),
            n_shuffles=margin_n_shuffles, n_bootstrap=margin_n_bootstrap,
            n_shuffles_bootstrap=margin_n_shuffles_bootstrap,
            min_tokens=criteria.preconditions["min_tokens"],
        )["margin_null"]
    except MetricNotApplicable as exc:
        margin_out = exc

    metrics = (
        _zloss_metrics(zloss_out["zloss_state"])
        + _maxvio_metrics(traj_out["expert_concentration"])
        + _margin_metrics(margin_out)
    )

    # PROBE_BATCH/PROBE_SEQ are re-exported identically by all three driver
    # modules (they import them from `offline_trajectory_probe`, see each
    # module's own docstring) -- reading them off `offline_trajectory_probe`
    # here, not redefining, avoids a fourth copy of the same two constants.
    from src.data.probe_corpus import PROBE_BATCH, PROBE_SEQ

    # Read once and shared by both producer fields below, so `code_commit`
    # cannot drift from `code_revision["commit"]`.
    _revision = _code_revision()

    # The probe drivers all return the corpus metadata they built their batch
    # from; take the sample description from there rather than recomputing it.
    _sample = zloss_out["token_source"]

    # The ROW-LEVEL precondition check (distinct from the per-cell `qualified`
    # flag `_margin_metrics` already reads off each margin-null record): this
    # is what the criteria file's own `preconditions` block is actually FOR.
    # `top_k` comes straight off
    # the model, not off `margin_out`, because `margin_out` is an Exception
    # (no fields to read) in exactly the case (`top_k < 2`) this precondition
    # also needs to catch.
    precondition_reading = {"tokens": _sample["n_tokens"], "top_k": model.config.num_active}
    precondition_failures = failed_preconditions(precondition_reading, criteria)

    reason_parts = []
    if not criteria.verdicts_allowed:
        # Deliberately NOT `criteria.refusal_text()`: that helper embeds
        # `self.path` (the caller's raw, possibly-absolute `--criteria`
        # value) into free text -- exactly how a frozen snapshot's directory
        # structure once leaked into real rows. This artifact
        # may only name the judged file by its basename + content hash, per
        # the named-root + relative-path rule -- never its path.
        reason_parts.append(
            f"criteria file `{Path(criteria_path).name}`"
            f" (sha256 prefix {_criteria_file_short_sha256(criteria_path)})"
            f" has status '{criteria.status}', which does not authorize a verdict"
        )
    if precondition_failures:
        # A ratified file whose preconditions this row fails must ALSO be
        # refused, not just a draft one -- an unsatisfied `min_tokens` does
        # not become satisfied by someone ratifying the thresholds around it.
        reason_parts.append("preconditions not met: " + "; ".join(precondition_failures))

    if reason_parts:
        verdict_reason = (
            "NO VERDICT. " + "; ".join(reason_parts) + ". "
            "The numbers above are for inspecting the distribution only, not a continue/stop decision."
        )
    else:
        verdict_reason = (
            "Criteria are ratified and all preconditions are met, but this driver does not "
            "implement automatic verdict computation, so it produces no automatic stop recommendation."
        )

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "run_name": run_name,
        "step": manifest["step"],
        "tokens_seen": manifest["tokens_consumed"],
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "producer": {
            "name": PRODUCER_NAME,
            "version": PRODUCER_VERSION,
            "code_revision": _revision,
            # Retained for one version so existing 1.0 readers keep working.
            # Same value as code_revision["commit"], not a second git call.
            "code_commit": _revision["commit"],
        },
        "checkpoint": {
            **_named_relpath(checkpoint_dir, root=runs_root, root_name=RUNS_ROOT_NAME),
            "step": manifest["step"],
        },
        # Read from the corpus metadata, NOT computed as PROBE_BATCH *
        # PROBE_SEQ. The corpus layer drops position 0 of every window, so the
        # nominal product now overstates the sample by one token per document
        # -- and would keep doing so silently, since both numbers look
        # equally plausible. `corpus_id` changes with the definition, so an
        # old row and a new one stay distinguishable.
        "sample": {
            "corpus_id": CORPUS_ID,
            "n_tokens": _sample["n_tokens"],
            "n_tokens_before_drop": _sample["n_tokens_before_drop"],
            "n_sequences": PROBE_BATCH,
            "first_token_dropped": _sample["first_token_dropped"],
            "special_tokens_excluded": _sample["special_tokens_excluded"],
            "tokens_per_doc": _sample["tokens_per_doc"],
        },
        "metrics": metrics,
        "verdict": {
            # Always "undecided": this pilot never computes an automated
            # continue/stop verdict (the pilot only produces evidence for a
            # human decision; it is not wired to stop training automatically).
            # Independently, while the
            # criteria file is "draft" `criteria_gate` would refuse a
            # verdict anyway (see `verdict_reason` above) -- both reasons
            # hold at once here, neither is a substitute for the other.
            "decision": "undecided",
            "rule_id": criteria.metric,
            "reason": verdict_reason,
            "preconditions_failed": precondition_failures,
        },
    }
    if job_id is not None:
        row["producer"]["job"] = {"job_id": job_id, "queue_wait_seconds": queue_wait_seconds}
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True, help="Trajectory checkpoint directory (HF format)")
    ap.add_argument(
        "--runs-root", required=True,
        help="Root directory checkpoint-dir must be under (e.g. $REPO/_runs). No default -- "
        f"recorded in the output as the named root {RUNS_ROOT_NAME!r} + a relative path, "
        "never as an absolute path (no account-identifying paths in artifacts). The real "
        "root->path mapping stays in local config, not here.",
    )
    ap.add_argument("--run-name", required=True, help="Must match src/pretrain/recipe.py's RUN_LIST run name")
    ap.add_argument("--out", required=True, help="Output JSONL path -- appended, one line, flushed")
    ap.add_argument(
        "--criteria", required=True,
        help="Path to a criteria_gate-compliant YAML (no default -- see criteria_gate.py's "
        "module docstring on why a readout must never run without one named explicitly).",
    )
    from src.data.tokenizer import TOKENIZER_CLASSES

    ap.add_argument(
        "--tokenizer-name", dest="tokenizer_name", choices=sorted(TOKENIZER_CLASSES), required=True,
        help="Which tokenizer this checkpoint expects -- see src.data.tokenizer.TOKENIZER_CLASSES. "
             "No default: a fixed default tokenizer would silently mismatch a checkpoint trained "
             "with a different one.",
    )
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    add_probe_dtype_argument(ap)
    ap.add_argument("--job-id", default=None, help="PBS job id, if this run was submitted as a job (optional)")
    ap.add_argument(
        "--queue-wait-seconds", type=int, default=None,
        help="Seconds between job submission and job start (optional; omit, don't pass 0, if unknown)",
    )
    args = ap.parse_args(argv)

    from src.pretrain.checkpointing.store import MANIFEST_NAME, is_complete
    from src.data.tokenizer import load_named_tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    # Re-verified here even though a caller may already have checked this --
    # do not trust the caller.
    if not is_complete(checkpoint_dir):
        raise ValueError(
            f"{checkpoint_dir} has no valid manifest.json -- refusing to probe an "
            "incomplete/unverified checkpoint (same check as the other three offline drivers)."
        )
    with open(checkpoint_dir / MANIFEST_NAME) as fh:
        manifest = json.load(fh)

    tokenizer_repo = load_named_tokenizer(args.tokenizer_name).repo_id

    print(f"[guardrail_eval] checkpoint: {checkpoint_dir} (step={manifest.get('step')}, run={args.run_name})")
    print(f"[guardrail_eval] criteria: {args.criteria}")

    import torch

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, trust_remote_code=True, **load_kwargs(args.probe_dtype)
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

    row = build_guardrail_row(
        model,
        tokenizer=tokenizer,
        device=args.device,
        probe_dtype=args.probe_dtype,
        run_name=args.run_name,
        checkpoint_dir=checkpoint_dir,
        runs_root=Path(args.runs_root).resolve(),
        manifest=manifest,
        criteria_path=args.criteria,
        job_id=args.job_id,
        queue_wait_seconds=args.queue_wait_seconds,
    )

    serialized = json.dumps(row, sort_keys=True)
    _assert_no_leaked_absolute_paths(serialized)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as fh:
        fh.write(serialized + "\n")
        fh.flush()
    print(f"[guardrail_eval] appended 1 row -> {out_path} ({len(row['metrics'])} metrics, "
          f"verdict.decision={row['verdict']['decision']})")


if __name__ == "__main__":
    main()

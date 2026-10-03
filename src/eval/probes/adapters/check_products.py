"""Acceptance check of one `run_hf_router_probe` product directory -- CONTENT, not
exit code (project discipline). This module is the executor of that rule for the
open-model adapter products, so it is versioned and tested like any other code.

    python -m src.eval.probes.adapters.check_products <out_dir>            # one product
    python -m src.eval.probes.adapters.check_products --determinism <a> <b> # two runs of one task

Exit status 0 only when every check passes; the failing check's text goes to
stderr and the status is 1. Checks: DONE.json <-> file hashes and row counts,
row mix per kind, the six identity/semantics tags on every row (40-hex commit),
loop_step == 0, blob shapes (cross_loop_overlap kept as not_applicable), sources
for every meta field, self-checks, deterministic DONE (no timing), named roots
(shared artifact_paths contract), the shared corpus block and the corpus-layer
facts (511 slots per document, CORPUS_ID), the probe-dtype fields, and a
whole-product scan for account-bearing paths. Then a per-layer table of the
headline readings for a human to look at.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter


from pathlib import Path
from typing import Any

# The corpora `probe_corpus` registers, read from it rather than copied: a constant here
# would be a second place that decides which corpus is legitimate, and it would go stale
# the moment a new corpus is added -- which is exactly what happened to the two checks
# below when the 500-document corpus arrived.
def _registered_corpus_ids() -> set[str]:
    from src.data import probe_corpus
    return {probe_corpus.CORPUS_ID, probe_corpus.LARGE_CORPUS_ID}
_ACCOUNT = re.compile("/" + r"scratch/|/" + r"home/")
TAGS = ("model_id", "revision", "revision_commit", "variant", "scoring_order", "combine_weights_normalized_by_model", "scores_are_probs")


class ProductCheckError(AssertionError):
    """A product failed a content check. The message names the check."""


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ProductCheckError(msg)


def _load(out: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    for f in ("DONE.json", "router_probe.jsonl", "probe_blobs.jsonl", "token_sources.json"):
        _need((out / f).is_file(), f"missing {f} (a directory without DONE.json is an unfinished run)")
    done = json.loads((out / "DONE.json").read_text())
    rows = [json.loads(l) for l in (out / "router_probe.jsonl").read_text().splitlines() if l]
    blobs = [json.loads(l) for l in (out / "probe_blobs.jsonl").read_text().splitlines() if l]
    sources = json.loads((out / "token_sources.json").read_text())
    return done, rows, blobs, sources


def _exclusion_rules() -> tuple[str, ...]:
    """The rule vocabulary, from the corpus layer that defines it -- not copied here."""
    from src.data.probe_corpus import EXCLUSION_RULES
    return tuple(EXCLUSION_RULES)


def _corpus_file_names() -> set[str]:
    """Filenames of the corpora `probe_corpus` registers, read from it rather than copied."""
    from src.data import probe_corpus
    return {probe_corpus.DEFAULT_CORPUS_PATH.name, probe_corpus.LARGE_CORPUS_PATH.name}


def check_product(out_dir: str | Path) -> str:
    """Raise `ProductCheckError` on the first failing check; return a human report."""
    out = Path(out_dir)
    done, rows, blobs, sources = _load(out)
    sha = hashlib.sha256((out / "router_probe.jsonl").read_bytes()).hexdigest()
    _need(sha == done.get("router_probe_jsonl_sha256"), "router_probe.jsonl sha256 != DONE.json")
    _need(len(rows) == done.get("n_rows"), f"row count {len(rows)} != DONE n_rows {done.get('n_rows')}")
    _need(hashlib.sha256((out / "probe_blobs.jsonl").read_bytes()).hexdigest() == done.get("probe_blobs_jsonl_sha256"), "probe_blobs.jsonl sha256 != DONE.json")
    seeds = list(done["args"]["probe_seeds"])
    _need(len(blobs) == done.get("n_blob_rows") == 2 * len(seeds), "blob row count != 2 x probe seeds")
    meta_rows = [r for r in rows if r["kind"] == "adapter_meta"]
    _need(len(meta_rows) == 1, "adapter_meta must appear exactly once")
    meta = meta_rows[0]
    L, S = int(meta["num_moe_layers"]), len(seeds)
    kinds = Counter(r["kind"] for r in rows)
    # One cross-architecture collapse row per (layer, loop_step); this family is non-looped.
    # The margin-null and archive-bootstrap rows are gone with the rho family (they do not
    # migrate to the new repository); the collapse metrics replace them.
    exp = {"adapter_meta": 1, "router": L * S, "zloss_state": L * S,
           "adapter_router_extra": L * S, "collapse_cross_arch": L}
    _need(dict(kinds) == exp, f"row mix {dict(kinds)} != expected {exp}")
    for r in rows:
        for t in TAGS:
            _need(t in r, f"row kind={r['kind']} lacks tag {t}")
        _need(bool(re.fullmatch(r"[0-9a-f]{40}", r["revision_commit"])), f"revision_commit is not a 40-hex commit: {r['revision_commit']!r}")
        _need(r["variant"] in ("pre_bias", "post_bias", "not_applicable"), f"variant {r['variant']!r}")
        if "loop_step" in r:
            _need(r["loop_step"] == 0, "loop_step must be 0 for a non-looped model")
    for b in blobs:
        _need(b["kind"] in ("expert_concentration", "cross_loop_overlap") and b["num_stacks"] == 1 and "adapter" in b, "blob shape")
        if b["kind"] == "cross_loop_overlap":
            _need(b["by_layer"] == [] and b["value_status"] == "not_applicable" and bool(b.get("reason")), "cross_loop_overlap must be empty + not_applicable + reason")
        else:
            _need(len(b["by_layer_loop"]) == b["num_layers_in_stack"], "expert_concentration by_layer_loop length")
    _need(all(k in meta["sources"] for k in meta if k not in ("kind", "sources")), "adapter_meta.sources incomplete")
    sc = done["self_checks"]
    _need(sc["all_equal_gives_zero"] and sc["top1_plus_delta_gives_delta"], "self-checks did not both pass")
    _need("elapsed_s" not in done, "DONE.json must be deterministic (timing belongs in run_log.json)")
    # named roots + identity (shared artifact_paths contract)
    _need(done.get("out_dir_root") == "probe_out_root" and done.get("out_dir_rel") == out.name, "out_dir named root/rel")
    # The corpus is identified by CONTENT, never by a fixed filename: this line used to
    # require the 40-document corpus by name and so rejected valid products taken over the
    # 500-document one. That is the same defect as a hardcoded CORPUS_ID -- a field that
    # looks like it asks "which corpus" while actually asking "is it the one I was written for".
    _need(done["args"].get("corpus_root") == "repo_root", "corpus named root")
    _need(str(done["args"].get("corpus_rel", "")).endswith(".json"), "corpus_rel is not a corpus file")
    _need(Path(done["args"]["corpus_rel"]).name in _corpus_file_names(), 
          f"corpus_rel {done['args']['corpus_rel']!r} is not a registered probe corpus")
    _need(bool(re.search(r"snapshots/[0-9a-f]{40}$", str(done.get("checkpoint_rel", "")))) and done.get("checkpoint_root") in ("scratch_root", "probe_out_root", "repo_root"), "checkpoint named root/rel")
    _need(done.get("model_ref") == f"{done['model_id']}@{done['revision_commit']}", "model_ref != model_id@revision_commit")
    for b in blobs:
        _need(b.get("model_ref") == done["model_ref"] and b.get("checkpoint_root") == done["checkpoint_root"] and "checkpoint" not in b, "blob identity fields")
    _need(all(t.get("corpus_root") == "repo_root" and "corpus_rel" in t and "corpus_file" not in t for t in sources), "token_source corpus named root")
    # shared corpus block + corpus-layer facts
    _need("corpus" in done, "DONE.json lacks the shared corpus block")
    cb = done["corpus"]
    # The two id fields must AGREE with each other, and name a registered corpus. Checking
    # them against one fixed constant answered "is this the corpus I was written for" while
    # appearing to answer "is this product's corpus stated consistently" -- and it is the
    # agreement that matters: a product carrying two different answers to "which corpus
    # produced this" is how a reading gets attributed to text it never saw.
    _need(cb.get("corpus_id") == done.get("corpus_id"),
          f"corpus block says {cb.get('corpus_id')!r} but DONE.json says {done.get('corpus_id')!r}")
    _need(cb.get("corpus_id") in _registered_corpus_ids(),
          f"corpus_id {cb.get('corpus_id')!r} is not a registered probe corpus")
    # Final field names: `exclusion_rule` says WHICH slot was removed
    # and `excluded_position` says where. Products written before the rename carry the
    # boolean `first_token_dropped` instead, and must keep validating -- 161 of them exist
    # and re-running finished work to change a label is not a reason to invalidate them.
    if "exclusion_rule" in cb:
        _need(cb["exclusion_rule"] in _exclusion_rules(),
              f"exclusion_rule {cb.get('exclusion_rule')!r} is not one of {_exclusion_rules()}")
        _need(isinstance(cb.get("excluded_position"), int),
              "corpus block lacks an integer excluded_position")
        # Null is a legitimate value: a corpus with no separator excludes a different real
        # token per document, so there is no single id (OLMoE, Granite).
        _need("excluded_token_id" in cb, "corpus block lacks excluded_token_id (null is allowed)")
    else:
        _need(cb.get("first_token_dropped") is True,
              "pre-rename corpus block must carry first_token_dropped=True")
    _need(cb.get("tokens_per_doc") == done.get("tokens_per_doc"),
          "corpus block tokens_per_doc disagrees with DONE.json")
    tpd = int(cb["tokens_per_doc"])
    _need(cb["n_tokens"] == done["n_tokens_after_drop"] and cb["probe_seed"] == seeds and len(cb["input_ids_sha8"]) == S, "corpus block totals / seeds")
    for e in done["token_exclusion"]:
        e_rule = e.get("exclusion_rule")
        _need((e_rule in _exclusion_rules()) if e_rule is not None else e.get("first_token_dropped") is True,
              "per-seed exclusion facts: neither exclusion_rule nor the pre-rename flag")
        _need(e["tokens_per_doc"] == tpd and e["n_tokens"] == e["n_tokens_before_drop"] - e["n_tokens_before_drop"] // (tpd + 1),
              "per-seed exclusion facts")
    _need(meta["special_tokens_excluded"] == sum(e["special_tokens_excluded"] for e in done["token_exclusion"]), "meta special_tokens_excluded != sum over seeds")
    for b in blobs:
        _need(b["corpus"]["corpus_id"] == cb["corpus_id"] and b["corpus"]["probe_seed"] == [b["probe_seed"]], "blob corpus block")
    # archive-level document bootstrap (for the difference threshold). The resampling unit is the DOCUMENT; the slot count
    # is seeds x documents and is NOT a sample size. Checked here because a wrong unit produces a
    # deviation that is merely too small -- no crash, no error, nothing in the product to see.
    ndd = done.get("n_distinct_documents")
    _need(isinstance(ndd, int) and ndd > 0, "DONE.json lacks a positive n_distinct_documents")
    _need(ndd * S == done["n_tokens_after_drop"] // tpd,
          f"n_distinct_documents {ndd} x {S} seeds != {done['n_tokens_after_drop'] // tpd} document slots")
    # cross-architecture collapse metrics: the ranges the spec fixes by construction
    cross = [r for r in rows if r["kind"] == "collapse_cross_arch"]
    _need({r["layer_idx"] for r in cross} == set(range(L)), "collapse_cross_arch rows do not cover every layer")
    for r in cross:
        where = f"collapse_cross_arch layer {r['layer_idx']}"
        _need(r["E"] == meta["n_experts"] and r["k"] == meta["top_k"], f"{where}: E/k disagree with adapter_meta")
        _need(r["N"] + r["n_zero_mass_tokens"] == done["n_tokens_after_drop"],
              f"{where}: N + zero-mass tokens != the corpus token count")
        _need(done["n_tokens_after_drop"] == done["probe_batch"] * tpd * S,
              f"{where}: token count != probe_batch x tokens_per_doc x seeds")
        # L in [0,1] and C in [0,1] are guaranteed by construction, so a value outside
        # them is an implementation fault, not a property of the model.
        for key in ("L2_starved_fraction", "Linf_starved_fraction"):
            _need(0.0 <= r[key] <= 1.0, f"{where}: {key}={r[key]} outside [0,1]")
        if r.get("metric_b_status") == "ok":
            for key in ("C_mean", "C_p25", "C_p50", "C_p75", "C_null"):
                _need(0.0 <= r[key] <= 1.0, f"{where}: {key}={r[key]} outside [0,1]")
            _need(1.0 <= r["k_eff_mean"] <= r["k"] + 1e-9, f"{where}: k_eff_mean={r['k_eff_mean']} outside [1,k]")
            _need(1.0 <= r["E_eff_2"] <= r["E"] + 1e-9, f"{where}: E_eff_2 outside [1,E]")

    # probe-dtype contract
    for k in ("probe_dtype", "forward_dtype", "probe_dtype_requested"):
        _need(k in done and k in meta, f"missing precision field {k}")
    _need(done["probe_dtype"] == done["forward_dtype"] == meta["probe_dtype"], "probe_dtype fields disagree")
    _need(done["probe_dtype_requested"] == done["args"]["probe_dtype"] == meta["probe_dtype_requested"], "probe_dtype_requested != --probe-dtype")
    _need(all(b["probe_dtype"] == done["probe_dtype"] for b in blobs), "blob probe_dtype")
    # whole-product account-path scan
    txt = "".join((out / f).read_text() for f in ("DONE.json", "router_probe.jsonl", "probe_blobs.jsonl", "token_sources.json"))
    m = _ACCOUNT.search(txt)
    _need(m is None, f"account-bearing path in product text at offset {m.start() if m else -1}")
    # report
    lines = [f"# {meta['model_id']} @ {meta['revision']} commit={meta['revision_commit']}",
             f"# forward={meta['forward_dtype']} requested={meta['probe_dtype_requested']} release={meta['release_dtype']} E={meta['n_experts']} k={meta['top_k']} layers={L} "
             f"scoring={meta['scoring_order']} lbm={meta['load_balancing_mechanism']} kind={meta['router_module_kind']}",
             f"# rows={len(rows)} {dict(kinds)} corpus={cb['corpus_id']} tokens_per_doc={tpd} n_tokens_effective={done['n_tokens_effective']} special={cb['special_tokens_excluded']} code={done['code_revision']['commit'][:12]} dirty={done['code_revision']['dirty']}",
             f"# named roots: out_dir={done['out_dir_root']}/{done['out_dir_rel']} checkpoint={done['checkpoint_root']}/...{done['checkpoint_rel'][-44:]}",
             f"# tie_token_count total = {sum(r['tie_token_count'] for r in rows if r['kind'] == 'router')}"]
    by: dict[int, dict[str, Any]] = {}
    for r in rows:
        if r["kind"] in ("router", "zloss_state", "adapter_router_extra") and r.get("probe_seed") == seeds[0]:
            by.setdefault(r["layer_idx"], {})[r["kind"]] = r
        if r["kind"] == "collapse_cross_arch":
            by.setdefault(r["layer_idx"], {})["cross"] = r
    # The rho columns (rho_cal and its bootstrap interval) are gone with the rho family.
    # What remains are the cross-architecture metrics plus the raw quantities the rows
    # already carried, so a reader still sees per-layer numbers rather than only a verdict.
    lines.append("layer |      L2    Linf |       C   C_null    C_adj | maxvio_count  maxvio_mass | frac_z_neg   z_mean")
    for li in sorted(by):
        d = by[li]
        if "cross" not in d:
            continue
        ro, z, c = d["router"], d["zloss_state"], d["cross"]
        cm = f"{c['C_mean']:7.4f} {c['C_null']:8.4f} {c['C_adj']:8.4f}" if c.get("metric_b_status") == "ok" \
             else f"{'n/a':>7} {'n/a':>8} {'n/a':>8}"
        lines.append(f"{li:5d} | {c['L2_starved_fraction']:7.4f} {c['Linf_starved_fraction']:7.4f} | {cm} | "
                     f"{ro['maxvio_count']:12.3f} {ro['maxvio_mass']:12.3f} | {z['frac_z_negative']:10.3f} {z['z_mean']:8.3f}")
    lines.append("CHECK OK")
    return "\n".join(lines)


def check_determinism(a: str | Path, b: str | Path) -> str:
    """Two runs of the same task must be byte-identical; DONE.json is
    compared after normalising the two fields that name the output directory."""
    a, b = Path(a), Path(b)
    out = []
    for f in ("router_probe.jsonl", "probe_blobs.jsonl", "token_sources.json"):
        _need((a / f).read_bytes() == (b / f).read_bytes(), f"{f} differs between the two runs")
        out.append(f"identical: {f}")
    def norm(p: Path) -> str:
        d = json.loads((p / "DONE.json").read_text())
        d["out_dir_rel"] = "<out>"; d["args"]["out_name"] = "<out>"
        return json.dumps(d, sort_keys=True)
    _need(norm(a) == norm(b), "DONE.json differs beyond the output-name fields")
    out.append("identical: DONE.json (output-name fields normalized)")
    out.append("DETERMINISM OK")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv[:1] == ["--determinism"]:
            if len(argv) != 3:
                raise SystemExit("usage: --determinism <dirA> <dirB>")
            print(check_determinism(argv[1], argv[2]))
        else:
            if len(argv) != 1:
                raise SystemExit("usage: <out_dir> | --determinism <dirA> <dirB>")
            print(check_product(argv[0]))
        return 0
    except ProductCheckError as err:
        print(f"CHECK FAILED: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

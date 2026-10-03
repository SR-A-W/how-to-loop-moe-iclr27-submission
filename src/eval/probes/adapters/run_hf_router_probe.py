"""Offline router probe for open-source HF MoE models (driver only).

Loads `--repo @ --revision` in the stated dtype, runs the project's fixed
probe corpus (`PROBE_BATCH x PROBE_SEQ`, one forward per `--probe-seed`),
captures every router through the family's adapter (with its self-check),
and writes one JSONL file whose rows are produced by the EXISTING
`collector.py` functions plus the adapter's own `adapter_meta` /
`adapter_router_extra` rows. Nothing metric-related is defined here.

Output directory contents:
    router_probe.jsonl   adapter_meta (1) ; per seed x layer: router,
                         zloss_state, adapter_router_extra ; per layer:
                         collapse_cross_arch
    probe_blobs.jsonl    per seed: expert_concentration + cross_loop_overlap rows
                         (shape inherited from the retired trajectory probe; kept so
                         existing consumers of these blobs do not have to change)
    token_sources.json   per-seed probe provenance (input_ids_sha8, ...)
    run_log.json         timing (kept out of DONE.json so DONE is deterministic)
    DONE.json            written LAST: row counts, sha256 of the jsonl,
                         code revision (the commit the run was launched from), args.
A directory without DONE.json is an unfinished or failed run.

Semantic arguments are required: `--step` (what training step this
revision represents; pass 0 for a final-only model and it is recorded as
such), `--dtype`, `--device`, and `--min-tokens` (the criteria file's
precondition floor -- this driver keeps no copy of that number).

Commit the code you are about to run (path-scoped) before launching, so the
commit lands in DONE.json via `provenance.code_revision`, and do not edit a
file this job imports while it is running. On a node without `git`, set
`CODE_REVISION_JSON` to a file written on the login node by
`json.dumps(code_revision(repo_root=<project dir>))`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch

from src.eval.probes.adapters import ADAPTERS
from src.eval.probes.adapters.base import RoutingSpec, pooled_kplus1, pooled_scores, row_tags, self_check_margin_known_answers
from src.metrics.collector import JsonlWriter, router_layer_loop_record
from src.data.probe_corpus import (
    CORPUS_ID, DEFAULT_CORPUS_PATH, EXCLUSION_RULES, RULE_POSITION_0, build_probe_batch,
    load_corpus,
)
from src.metrics.collapse_cross_arch import collapse_metrics
from src.metrics.provenance import code_revision
from src.eval.probes.artifact_paths import assert_no_account_paths, corpus_block, named_checkpoint
from src.eval.probes.probe_dtype import add_probe_dtype_argument, dtype_fields, load_kwargs


def resolve_repo_root(this_file: str | Path, depth: int = 4) -> Path:
    """Repo root from a source file's location. `depth` is the number of
    directories between the file and the root (src/eval/probes/adapters -> 4).
    Guarded: a wrong depth after a file move must raise
    here, not surface later as a 'path under neither root' error."""
    root = Path(this_file).resolve().parents[depth]
    if not (root / "src").is_dir() or not (root / "src" / "eval").is_dir():
        raise RuntimeError(f"resolve_repo_root: {root} (depth={depth} from {this_file}) is not the repo root -- file moved? fix the depth")
    return root


REPO_ROOT = resolve_repo_root(__file__)


def _named(path: str | Path, field: str) -> dict[str, str]:
    """Shared named-root form `{field_root, field_rel}` (src.eval.probes.artifact_paths): the same function every probe uses, so one contract, one set of roots
    (repo_root, LOOPED_PROBE_OUT_ROOT -> probe_out_root, SCRATCH_ROOT -> scratch_root).
    Raises when a path is under no known root -- never falls back to the absolute path."""
    return named_checkpoint(path, field=field)


PROBE_SEQ = 512
"""Tokens per document window, before the corpus layer drops position 0.

Was imported from `offline_trajectory_probe`, which does not migrate to the new
repository. Stated here rather than re-derived: it is half of the corpus artifact's
identity (`pile10k_d4_b40_s512` -- the s512), and every product cross-checks it against
`token_source["tokens_per_doc"]` per seed, so a disagreement with the corpus raises.
"""

DEFAULT_PROBE_SEEDS = (0, 1, 2, 3)
"""Window-drawing seeds. Was imported from the margin-null probe, which does not migrate.

These are the probe's error-bar seeds, not training seeds: each draws a different window
per document from the same documents.
"""


def effective_token_count(token_sources: list[dict]) -> int:
    """Distinct tokens across seeds, deduplicating overlapping windows.

    Each seed draws a `[start, start+len)` window per document. On a short document two
    seeds' windows overlap, so `n_seeds * batch * seq` counts the same text twice and
    reports a sample size that does not exist. This unions the position intervals per
    document instead. Documents are keyed by name (falling back to position) because the
    same document appears once per seed and the union is per document, not per (document,
    seed).

    Vendored from `offline_margin_null_probe._effective_token_count`, which does not
    migrate. Copied rather than re-derived, so the number this driver reports stays the
    number every earlier product reported.
    """
    seen: dict[str, set[int]] = {}
    for source in token_sources:
        for position, doc in enumerate(source["docs"]):
            key = str(doc.get("name", position))
            # Both recorded POST-drop by the corpus layer, so this counts exactly the
            # positions that take part in a reduction.
            start = int(doc["window_start"])
            seen.setdefault(key, set()).update(range(start, start + int(doc["window_len"])))
    return sum(len(v) for v in seen.values())


CORPUS_EXCLUSION_RULE = RULE_POSITION_0
"""Which slot the corpus layer removes from every window for this driver's models.

Referenced from `probe_corpus`, not spelled out here: the vocabulary changed three times
in one day (`last_position` -> `position_last`, with a detour back), and a copied literal
would have stopped at whichever spelling it was written under -- silently, since a stale
string is still a string. The membership check below then only catches what a reference
makes impossible.

Why position 0 for all three: JetMoE prepends BOS (measured first id 1, in
`all_special_ids`), so position 0 IS its separator. OLMoE prepends nothing at all
(`bos_token_id is None`, measured first id 510) and Granite does not prepend either, so
for them the rule removes an ordinary token purely to keep every document the same
post-exclusion length -- which is why their `special_tokens_excluded` reads 0 and their
`excluded_token_id` comes out null (the per-document ids differ: 1147, 14277, 32197, ...).

`"position_last"` would drop the trailing slot instead: it would keep JetMoE's BOS and
move all three models' numbers. Every one of the 161 existing products was produced under
position 0, so this is also what keeps them comparable.
"""

if CORPUS_EXCLUSION_RULE not in EXCLUSION_RULES:
    raise RuntimeError(
        f"{CORPUS_EXCLUSION_RULE!r} is not one of the corpus layer's rules {EXCLUSION_RULES}. "
        "Checked at import rather than at the call: the wrong rule excludes the other end of "
        "every document and nothing downstream fails."
    )


def exclusion_happened(source: dict) -> bool:
    """Whether the corpus layer already excluded one position per window.

    Reads the current field (`exclusion_rule`, a rule name) and falls back to the boolean
    `first_token_dropped` that products written before the rename carry -- 161 of them
    exist, and re-running finished work to change a label is not a reason to reject them.

    Raises when neither is present rather than returning False: a corpus predating the
    exclusion is a different measurement, and reading a missing field as "nothing was
    excluded" would let this driver exclude the position a second time, leaving counts
    that stay self-consistent while describing text that was never measured.
    """
    if "exclusion_rule" in source:
        return source["exclusion_rule"] in EXCLUSION_RULES
    if "first_token_dropped" in source:
        return bool(source["first_token_dropped"])
    raise RuntimeError(
        "token_source carries neither 'exclusion_rule' nor the older "
        "'first_token_dropped': this corpus predates the uniform per-window exclusion"
    )


def _write(writer: JsonlWriter, row: dict, *, what: str) -> None:
    """Run-time half of the no-account-path rule on the serialised row, then write."""
    assert_no_account_paths(row, what=what)
    writer.write(row)




def _config_json_experts(snapshot_dir: Path) -> tuple[dict, dict]:
    """Read n_experts / top_k straight from the repo's config.json text,
    trying the key names the families use; returns ({field: value},
    {field: 'config.json:<key>'}). Raises if a field has no key."""
    cfg = json.loads((snapshot_dir / "config.json").read_text())
    out, src = {}, {}
    for field, keys in (("n_experts", ("num_experts", "num_local_experts", "moe_num_experts")),
                        ("top_k", ("num_experts_per_tok", "moe_top_k"))):
        for k in keys:
            if k in cfg:
                out[field] = int(cfg[k]); src[field] = f"config.json:{k}"; break
        else:
            raise RuntimeError(f"config.json has none of {keys} for {field}; refusing to guess")
    return out, src


def _release_dtype(snapshot_dir: Path) -> str:
    from safetensors import safe_open
    files = sorted(snapshot_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors under {snapshot_dir}; cannot state the release dtype")
    with safe_open(str(files[0]), "pt") as f:
        return str(f.get_slice(next(iter(f.keys()))).get_dtype())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--step", type=int, required=True, help="training step this revision represents (0 = final-only model)")
    ap.add_argument("--out-root", default=None, help="local root directory for probe outputs (default: $LOOPED_PROBE_OUT_ROOT; never written into products)")
    ap.add_argument("--out-name", required=True, help="output subdirectory name under --out-root (recorded relative to the root)")
    add_probe_dtype_argument(ap)  # --probe-dtype, required, shared contract
    ap.add_argument("--device", required=True)
    ap.add_argument("--probe-seeds", type=int, nargs="+", default=list(DEFAULT_PROBE_SEEDS))
    # Required, no defaults: both change C_null and therefore C_adj, and the module that
    # computes them refuses defaults for exactly that reason -- two runs must not be able
    # to differ without their command lines differing.
    # Required, no default: it is the sample size. A default here would let two runs measure
    # different numbers of documents while their command lines looked the same, and the
    # smaller one would simply read noisier -- it would not fail.
    ap.add_argument("--probe-batch", dest="probe_batch", type=int, required=True)
    # Required for the same reason metrics' driver requires it: it decides whether the run
    # fits in memory at all, and it must be stated rather than guessed. It does NOT change
    # the numbers -- the captures are concatenated and every metric is computed once over
    # the whole batch -- which is what makes it safe to vary between runs.
    ap.add_argument("--doc-chunk", dest="doc_chunk", type=int, required=True)
    ap.add_argument("--c-null-tokens", dest="c_null_tokens", type=int, required=True)
    ap.add_argument("--c-null-seed", dest="c_null_seed", type=int, required=True)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS_PATH))
    ap.add_argument("--save-scores", action="store_true", help="also write per-layer pooled scores (fp32 .pt) and (k+1)-th scores")
    ap.add_argument("--dry-run", action="store_true", help="validate snapshot/config/output plan and stop BEFORE loading the model (no GPU, no products)")
    a = ap.parse_args(argv)

    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if len(set(a.probe_seeds)) != len(a.probe_seeds):
        raise ValueError(f"--probe-seeds {a.probe_seeds} repeats a seed (would double-count tokens)")
    import os
    out_root_str = a.out_root or os.environ.get("LOOPED_PROBE_OUT_ROOT")
    if not out_root_str:
        raise RuntimeError("no output root: pass --out-root or set LOOPED_PROBE_OUT_ROOT (artifacts record a named root plus a relative path, never an absolute path)")
    if a.out_root and os.environ.get("LOOPED_PROBE_OUT_ROOT") and Path(a.out_root).resolve() != Path(os.environ["LOOPED_PROBE_OUT_ROOT"]).resolve():
        raise RuntimeError("--out-root and LOOPED_PROBE_OUT_ROOT disagree; the artifact would name a root it is not under")
    os.environ.setdefault("LOOPED_PROBE_OUT_ROOT", out_root_str)  # so named_checkpoint() knows probe_out_root
    out_root = Path(out_root_str)
    out = out_root / a.out_name
    if (out / "DONE.json").exists():
        raise FileExistsError(f"{out} already has DONE.json; refusing to overwrite a finished run")
    if not a.dry_run:  # a dry run must not leave an empty product directory behind
        out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if a.dry_run:
        # Rehearsal must cover revisions that are streamed (download -> probe -> delete) and
        # therefore not cached: resolve the commit and fetch ONLY config.json through the hub.
        from huggingface_hub import HfApi, hf_hub_download
        commit = HfApi().model_info(a.repo, revision=a.revision).sha
        cfg_path = Path(hf_hub_download(a.repo, "config.json", revision=commit))
        snap = cfg_path.parent
        cfg_vals, cfg_src = _config_json_experts(snap)
        weights = sorted(snap.glob("*.safetensors"))
        plan = {"family": a.family, "repo": a.repo, "revision": a.revision, "revision_commit": commit,
                "release_dtype": _release_dtype(snap) if weights else "not cached (streamed at run time)",
                "probe_dtype_requested": a.probe_dtype, "device": a.device, "probe_seeds": a.probe_seeds,
                **_named(out, "out_dir"), **_named(snap, "checkpoint"),  # both roots must resolve BEFORE any model load
                "model_ref": f"{a.repo}@{commit}",
                "config_json": {**cfg_vals, "sources": cfg_src}, **_named(a.corpus, "corpus"),
                "code_revision": code_revision(), "self_checks": self_check_margin_known_answers()}
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RuntimeError(f"hub returned a non-commit sha {commit!r}")
        print("[hf_router_probe] DRY-RUN OK " + json.dumps(plan, sort_keys=True, default=str), file=sys.stderr)
        return
    snap = Path(snapshot_download(a.repo, revision=a.revision, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]))
    if not re.fullmatch(r"[0-9a-f]{40}", snap.name):
        raise RuntimeError(f"resolved snapshot dir name {snap.name!r} is not a 40-hex commit; refusing to record it as revision_commit")
    tok = AutoTokenizer.from_pretrained(a.repo, revision=a.revision)
    # No `device_map=`: that path requires `accelerate`, which the `looptrain`
    # env (the only env with working CUDA on our GPU node) does not carry.
    model = AutoModelForCausalLM.from_pretrained(
        a.repo, revision=a.revision, **load_kwargs(a.probe_dtype),
    ).to(a.device).eval()
    precision = dtype_fields(model, a.probe_dtype)  # raises if the model ignored the requested dtype
    adapter = ADAPTERS[a.family]()
    self_checks = self_check_margin_known_answers()  # raises on failure
    routers = adapter.find_routers(model)
    module_paths = tuple(name for name, _ in routers)
    mfields, msources = adapter.model_spec(model, release_dtype=_release_dtype(snap), transformers_version=transformers.__version__)
    # n_experts / top_k must agree bit-for-bit with the repo's config.json text.
    cfg_vals, cfg_src = _config_json_experts(snap)
    for f in ("n_experts", "top_k"):
        if cfg_vals[f] != mfields[f]:
            raise RuntimeError(f"{f}: config.json says {cfg_vals[f]} but the loaded model/class says {mfields[f]}; refusing to continue")
    msources["n_experts"] = cfg_src["n_experts"] + " (== loaded config)"
    msources["top_k"] = cfg_src["top_k"] + " (== loaded config)"
    # First-token drop is done ONCE, in the corpus layer (probe_corpus, CORPUS_ID
    # pile10k_d4_b40_s512_drop1_v2): every consumer inherits it. Nothing is dropped here;
    # tokens_per_doc and the exclusion facts are read from the corpus metadata below.
    TOKENS_PER_DOC = PROBE_SEQ - 1  # cross-checked against token_source["tokens_per_doc"] per seed
    # How many documents to probe is stated on the command line, not inherited from a constant
    # tied to one corpus file: this driver now runs against corpora of different sizes, and the
    # sample size is what the finite-sample requirement is about (spec 2.6 wants N/E >= 1000),
    # so it must be visible in the command line and in the product rather than implied.
    # `build_probe_batch` takes `docs[:batch]` and raises if the corpus has fewer.
    probe_batch = a.probe_batch
    corpus_documents_available = len(load_corpus(a.corpus)[0])
    if probe_batch > corpus_documents_available:
        raise RuntimeError(
            f"--probe-batch {probe_batch} but {a.corpus} holds {corpus_documents_available} documents"
        )
    spec_kwargs = dict(
        model_id=a.repo, revision=a.revision, revision_commit=snap.name,
        probe_dtype=precision["probe_dtype"], probe_dtype_requested=precision["probe_dtype_requested"],
        model_native_dtype=mfields["release_dtype"],
        num_layers=mfields["num_moe_layers"], z_loss_coefficient=mfields["z_loss_coef"],
        router_module_paths=module_paths, tokenizer="<set from token_source after the batches are built>",
        n_tokens_total=probe_batch * TOKENS_PER_DOC * len(a.probe_seeds),  # cross-checked against sum(token_source["n_tokens"])
        special_tokens_excluded=0,  # replaced by the corpus layer's count after the batches are built
        **mfields,
    )
    sources = dict(
        model_id="driver --repo", revision="driver --revision",
        revision_commit="huggingface_hub snapshot directory name (resolved commit)",
        probe_dtype=msources["forward_dtype"], probe_dtype_requested="driver --probe-dtype as typed (src.eval.probes.probe_dtype)",
        model_native_dtype=msources["release_dtype"],
        num_layers=msources["num_moe_layers"], z_loss_coefficient=msources["z_loss_coef"],
        router_module_paths="module names of the hooked routers (model.named_modules)",
        tokenizer="token_source['tokenizer'] (probe_corpus names a local tokenizer directory by root; a hub id is kept as-is)", n_tokens_total="sum over probe seeds of token_source['n_tokens'] (after the corpus layer's first-token drop, before de-duplication)",
        special_tokens_excluded=f"probe_corpus.build_probe_batch (CORPUS_ID={CORPUS_ID}): token_source['special_tokens_excluded'] summed over probe seeds -- dropped position-0 tokens that were special",
        **msources,
    )
    spec = None  # built after the special-token count is known (block C)
    print(f"[hf_router_probe] loaded {a.repo}@{a.revision} ({snap.name[:12]}) in {time.time()-t0:.0f}s; "
          f"E={mfields['n_experts']} k={mfields['top_k']} layers={mfields['num_moe_layers']} forward={mfields['forward_dtype']} release={mfields['release_dtype']}",
          file=sys.stderr)

    jsonl_path = out / "router_probe.jsonl"
    blobs_path = out / "probe_blobs.jsonl"
    n_rows = 0
    n_blob_rows = 0
    captures_by_seed = []
    token_sources = []
    checkpoint_id = f"{a.repo}@{snap.name}"          # model_ref: identity (no path)
    checkpoint_named = _named(snap, "checkpoint")      # {checkpoint_root, checkpoint_rel}: the HF snapshot dir
    exclusion_stats = []
    # Pass 1: batches come from the corpus layer already lacking position 0 of each document.
    inputs_by_seed = []
    for seed in a.probe_seeds:
        ids, source = build_probe_batch(tok, a.corpus, batch=probe_batch, seq=PROBE_SEQ,
                                       exclusion_rule=CORPUS_EXCLUSION_RULE, seed=seed)
        # Guards against a second (driver-side) drop or a stale corpus: shapes must match the
        # corpus metadata exactly, and that metadata must say the drop already happened.
        if not exclusion_happened(source):
            raise RuntimeError("the corpus layer reports no per-window exclusion; this driver "
                               "relies on it having happened exactly once, upstream")
        if int(source["tokens_per_doc"]) != TOKENS_PER_DOC or ids.shape != (probe_batch, TOKENS_PER_DOC):
            raise RuntimeError(f"tokens_per_doc={source['tokens_per_doc']} / ids {tuple(ids.shape)} != expected {probe_batch}x{TOKENS_PER_DOC}")
        if int(source["n_tokens"]) != probe_batch * TOKENS_PER_DOC or len(set(source["slots_per_document"])) != 1:
            raise RuntimeError("corpus metadata inconsistent: n_tokens or slots_per_document")
        caps = adapter.capture(model, ids.to(a.device), doc_chunk=a.doc_chunk)
        captures_by_seed.append(caps)
        # Written under the current name only. Emitting both would put one fact in two
        # fields with nothing keeping them equal -- the shape that already bit us twice.
        exclusion_stats.append({"probe_seed": seed,
                                "exclusion_rule": source["exclusion_rule"],
                                "excluded_position": int(source["excluded_position"]),
                                "special_tokens_excluded": int(source["special_tokens_excluded"]),
                                "tokens_per_doc": int(source["tokens_per_doc"]), "n_tokens": int(source["n_tokens"]),
                                "n_tokens_before_drop": int(source["n_tokens_before_drop"]),
                                "slots_per_document_all_equal": True})
        inputs_by_seed.append((seed, source, sum(c.tie_token_count for c in caps)))
    if mfields["forward_dtype"] != precision["forward_dtype"]:
        raise RuntimeError(f"adapter saw parameter dtype {mfields['forward_dtype']} but probe_dtype check says {precision['forward_dtype']}")
    tokenizer_named = {k: v for k, v in inputs_by_seed[0][1].items() if k.startswith("tokenizer")}  # corpus layer names a local tokenizer dir
    spec = RoutingSpec(sources=sources, **{**spec_kwargs, "tokenizer": tokenizer_named["tokenizer"],
                                           "special_tokens_excluded": int(sum(e["special_tokens_excluded"] for e in exclusion_stats))})
    with JsonlWriter(jsonl_path) as w, JsonlWriter(blobs_path) as wb:
        _write(w, spec.to_row(), what=str(jsonl_path.name)); n_rows += 1
        for (seed, source, ties), caps in zip(inputs_by_seed, captures_by_seed):
            src_rec = {"probe_seed": seed, **source}
            if "corpus_file" in src_rec:
                src_rec.update(_named(src_rec.pop("corpus_file"), "corpus_file"))
            token_sources.append(src_rec)
            print(f"[hf_router_probe] seed={seed} sha8={source['input_ids_sha8']} layers={len(caps)} "
                  f"reconciled tokens={probe_batch*TOKENS_PER_DOC}/layer ties={ties} (corpus layer dropped position 0)", file=sys.stderr)
            by_layer_loop = []
            for c in caps:
                rows = adapter.rows_for_capture(c, step=a.step, spec=spec,
                                                extra={"probe_seed": seed, "input_ids_sha8": source["input_ids_sha8"]})
                for row in rows:
                    _write(w, row, what=jsonl_path.name); n_rows += 1
                by_layer_loop.append(rows[0])
            # The same two blob shapes offline_trajectory_probe writes (fields and
            # nesting identical; additive keys only). cross_loop_overlap is kept for a
            # non-looped model with an empty by_layer and value_status not_applicable.
            common = {"model_ref": checkpoint_id, **checkpoint_named, **precision, "corpus": corpus_block(source), "probe_seed": seed,
                      "input_ids_sha8": source["input_ids_sha8"], "n_experts": spec.n_experts,
                      "num_layers_in_stack": spec.num_moe_layers, "num_stacks": 1,
                      "model_id": spec.model_id, "revision_commit": spec.revision_commit, "adapter": spec.to_row()}
            _write(wb, {**common, "kind": "expert_concentration", "by_layer_loop": by_layer_loop}, what=blobs_path.name); n_blob_rows += 1
            _write(wb, {**common, "kind": "cross_loop_overlap", "by_layer": [], "value_status": "not_applicable",
                        "reason": "non-looped model: num_stacks=1, there is no second loop step to overlap with"}, what=blobs_path.name); n_blob_rows += 1
        # token_source["docs"][i]["window_start"/"window_len"] are recorded post-drop by the corpus
        # layer; the count function reads them itself (one-argument form).
        n_eff = effective_token_count(token_sources)
        # The resampling unit for the archive bootstrap. Derived, never typed in: a constant
        # here would be a second copy of a number the corpus already knows, and it would go on
        # reading 40 after the corpus changed. Every seed reads the SAME documents and varies
        # only the window (probe_corpus slices docs[:batch] before the seed is consulted), so
        # slots = documents x seeds and only the document count is a sample size.
        doc_names_per_seed = [tuple(d["name"] for d in t["docs"]) for t in token_sources]
        if len(set(doc_names_per_seed)) != 1:
            raise RuntimeError(
                f"probe seeds do not read the same documents in the same order "
                f"({len(set(doc_names_per_seed))} distinct orderings). Grouping slot s*n_docs+d "
                "as document d is only meaningful when every seed's d-th document is the same text."
            )
        n_distinct_documents = len(set(doc_names_per_seed[0]))
        if n_distinct_documents != len(doc_names_per_seed[0]):
            raise RuntimeError(
                f"a probe seed reads {len(doc_names_per_seed[0])} documents but only "
                f"{n_distinct_documents} are distinct; a repeated document would be resampled "
                "as if it were two."
            )
        pooled = pooled_scores(captures_by_seed)
        kplus1 = pooled_kplus1(captures_by_seed)

        def _pool(attr: str) -> dict[int, torch.Tensor]:
            """Concatenate one per-capture `(B, S, k)` field across seeds, per layer.

            Same layer grid and same seed order as `pooled_scores`, which has already
            raised if the grids disagree, so the rows line up token for token with the
            pooled logits.
            """
            per_layer: dict[int, list[torch.Tensor]] = {}
            for caps in captures_by_seed:
                for c in caps:
                    t = getattr(c, attr)
                    per_layer.setdefault(c.layer_idx, []).append(t.reshape(-1, t.shape[-1]))
            return {li: torch.cat(v, dim=0) for li, v in per_layer.items()}

        pooled_idx = _pool("topk_idx")
        # The weights the forward pass ACTUALLY applied, not a renormalised copy: the spec
        # takes every quantity from `a_{t,i}`, and `_reconcile` has already checked these
        # against a recomputation of the model's own routing. Each family's convention
        # (OLMoE softmax-over-all-then-topk, Granite and JetMoE topk-then-softmax-over-selected)
        # is recorded per row as `scoring_order` rather than corrected for here -- the metrics
        # normalise internally, so a renormalised and a raw router give the same answer.
        pooled_w = _pool("topk_weights_model")
        if a.save_scores:
            for layer_idx, scores in pooled.items():
                torch.save({"scores": scores.cpu().contiguous(), "score_kplus1": kplus1[layer_idx].cpu().contiguous(),
                            "layout": "rows = probe seeds x docs x kept positions (seed-major; corpus layer dropped position 0), tokens_per_doc=%d" % TOKENS_PER_DOC},
                           out / f"scores_layer{layer_idx:02d}.pt")
        # --- cross-architecture collapse metrics (spec v1) -------------------------
        # One row per (layer, loop_step); this family is non-looped so loop_step is 0.
        # Deliberately NOT averaged over layers here: the spec forbids collapsing the grid
        # early, because the depth split is the thing the measurement exists to show.
        # The metric functions live in src/metrics/collapse_cross_arch.py and are called,
        # never reimplemented -- a second implementation is a second set of numbers.
        for layer_idx in sorted(pooled):
            m = collapse_metrics(
                topk_idx=pooled_idx[layer_idx].cpu().numpy(),
                topk_weight=pooled_w[layer_idx].cpu().numpy(),
                router_logits=pooled[layer_idx].float().cpu().numpy(),
                # This cell's own expert count, from the tensor -- not `spec.n_experts`,
                # which is one number for the whole model. A head/tail layer keeps 8
                # experts while the loop block may have 64, and applying the model-level
                # value to such a row reads a healthy layer as heavily collapsed without
                # raising anything. Today's three open models have no head/tail layers so
                # the two agree; they stop agreeing the moment a looped product arrives.
                n_experts=int(pooled[layer_idx].shape[-1]),
                c_null_tokens=a.c_null_tokens,
                c_null_seed=a.c_null_seed,
            )
            _write(w, {"kind": "collapse_cross_arch", "layer_idx": layer_idx, "loop_step": 0,
                       **m, **row_tags(spec)}, what=jsonl_path.name)
            n_rows += 1
        print(f"[hf_router_probe] collapse_cross_arch: {len(pooled)} layer(s)", file=sys.stderr)

    assert_no_account_paths(token_sources, what="token_sources.json")
    (out / "token_sources.json").write_text(json.dumps(token_sources, indent=1, sort_keys=True))
    args_rec = {**{k: v for k, v in vars(a).items() if k not in ("out_root", "corpus")}, "out_name": a.out_name,
                **_named(a.corpus, "corpus")}
    sha = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    sha_blobs = hashlib.sha256(blobs_path.read_bytes()).hexdigest()
    # Timing goes to run_log.json so DONE.json is byte-identical across identical runs.
    (out / "run_log.json").write_text(json.dumps({"elapsed_s": round(time.time() - t0, 1), "device": a.device}, indent=1))
    done = {
        "kind": "hf_router_probe",
        "driver": "src.eval.probes.adapters.run_hf_router_probe",  # module path of the code that wrote this product
        "args": args_rec,
        **_named(out, "out_dir"), "model_ref": checkpoint_id, **checkpoint_named,
        "model_id": a.repo, "revision": a.revision, "revision_commit": snap.name,
        # OLMoE-style branch names carry the training-token count ("step25000-tokens104B");
        # recorded explicitly (None when the revision name has no such tag).
        "revision_train_tokens_B": (int(m.group(1)) if (m := re.search(r"tokens(\d+)B", a.revision)) else None),
        "step": a.step,
        "n_rows": n_rows, "router_probe_jsonl_sha256": sha,
        "n_blob_rows": n_blob_rows, "probe_blobs_jsonl_sha256": sha_blobs,
        "self_checks": self_checks, "router_module_paths": list(module_paths),
        "corpus": corpus_block(token_sources),   # one shape across all probe artifacts (artifact_paths)
        # Taken from the corpus block, which resolves the id from the corpus file's CONTENT,
        # rather than from the module constant. Writing the constant here made DONE.json
        # claim the 40-document corpus while the block correctly named the 500-document one:
        # one fact, two sources, no constraint between them -- so they drifted the moment a
        # second corpus existed. There is now exactly one source, and check_products asserts
        # the two fields agree.
        "corpus_id": corpus_block(token_sources)["corpus_id"], **precision, **tokenizer_named,
        "n_tokens_raw": int(sum(e["n_tokens_before_drop"] for e in exclusion_stats)),
        "n_tokens_after_drop": int(sum(e["n_tokens"] for e in exclusion_stats)),
        "n_tokens_effective": n_eff, "tokens_per_doc": TOKENS_PER_DOC, "token_exclusion": exclusion_stats,
        # Both, so a run over a SUBSET of a larger corpus is visible in the product instead of
        # having to be inferred by comparing a token count against the corpus file.
        "probe_batch": int(probe_batch), "corpus_documents_available": int(corpus_documents_available),
        # Recorded because it changes how the run was executed; it must NOT change the result.
        "doc_chunk": int(a.doc_chunk),
        # Recorded next to the slot counts on purpose: n_tokens_* scale with probe seeds and
        # only this one is a sample size. Slots = probe seeds x this number.
        "n_distinct_documents": int(n_distinct_documents),
        "code_revision": code_revision(),  # git, or CODE_REVISION_JSON on git-less nodes (provenance); raises otherwise
    }
    assert_no_account_paths(json.dumps(done, sort_keys=True, default=str), what="DONE.json")
    (out / "DONE.json").write_text(json.dumps(done, indent=1, sort_keys=True, default=str))
    print(f"[hf_router_probe] DONE {a.out_name} rows={n_rows} blob_rows={n_blob_rows} n_tokens_effective={n_eff} {time.time()-t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

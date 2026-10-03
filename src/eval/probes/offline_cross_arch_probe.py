"""Offline driver for the cross-architecture collapse metrics (spec v1).

Same shape as `offline_zloss_probe.py`: load a saved checkpoint, build the one
fixed probe corpus, run a single `run_probe_forward` with a collector attached,
and hand each `(loop_step, layer_idx)` cell to a pure function that owns the
metric definitions. This file is a driver only -- every definition lives in
`src.metrics.collapse_cross_arch`, and the reasons for them are in its
docstring, not repeated here.

Three things this driver does that the metric function cannot:

* **Dumps the routing decisions.** `topk_idx` and `topk_weight` per cell go to
  a `.npz` beside the row, with the token ids, so any later question about the
  metrics can be answered without a GPU or a re-run. The metrics are cheap to
  recompute; a forward pass is not.
* **Optionally subsamples hidden states**, for the final-state checkpoints
  only. Trajectory archives are 16x more numerous and the same data for them
  would be tens of gigabytes for a question nobody has asked yet.
* **Records what determined the numbers**: code revision, corpus identity and
  hash, precision, E, k, N, and the null-reference draw count and seed. A
  metric row that cannot say which corpus produced it is not reproducible, and
  the corpus is about to change from 40 documents to 500.

Every parameter is required, with no default. `E`, the document count, the
null-reference size and its seed all change the numbers, and a default lets two
runs differ while their command lines agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.data.probe_corpus import (
    CORPUS_ID,
    DEFAULT_CORPUS_PATH,
    LARGE_CORPUS_ID,
    LARGE_CORPUS_PATH,
    LARGE_CORPUS_SMOLLM2_ID,
    LARGE_CORPUS_SMOLLM2_PATH,
    build_probe_batch,
)

#: The corpora this driver will run against, by the name used on the command
#: line. Both the path and the id come from `probe_corpus`, never typed here:
#: a hand-copied id could name a corpus the run did not actually read, and
#: that is the one field nothing downstream can check.
#: name -> (path, id, tokenizer_name). The tokenizer belongs here because the
#: separator position depends on it (llama3 puts it at 0, smollm2 at seq-1), so
#: "which corpus" and "which tokenizer" cannot be chosen independently without
#: excluding the wrong position -- which would not fail, it would quietly
#: measure one real token less and one separator more.
CORPORA = {
    "small40": (DEFAULT_CORPUS_PATH, CORPUS_ID, "llama3"),
    "large500": (LARGE_CORPUS_PATH, LARGE_CORPUS_ID, "llama3"),
    "large500_smollm2": (LARGE_CORPUS_SMOLLM2_PATH, LARGE_CORPUS_SMOLLM2_ID, "smollm2"),
}

import ast
import contextlib

from src.model.loop_trace import legacy_trace_point_factory
from src.eval.probes.artifact_paths import (
    assert_no_account_paths,
    corpus_block,
    exclusion_rule_for_tokenizer,
    named_checkpoint,
)
from src.data.probe_corpus import PROBE_SEQ, WEIGHTS_FILENAME
from src.data.tokenizer import TOKENIZER_CLASSES
from src.eval.probes.probe_dtype import add_probe_dtype_argument, dtype_fields, load_kwargs
from src.metrics.collapse_cross_arch import collapse_metrics
from src.model.loop_trace import NullCollector, TraceMeta, TraceSink, run_probe_forward
#: The rule each corpus implies, derived from its tokenizer through the one
#: mapping rather than written out again here -- a second copy would be a
#: second place to get it wrong.
#: Storage types the raw router-score dump may use. float16 is the default and
#: halves the file; it is lossy for re-deriving the selection (see --logits-dtype).
LOGIT_DTYPES = {"float16": np.float16, "float32": np.float32}

CORPUS_EXCLUSION_RULES = {
    name: exclusion_rule_for_tokenizer(tok) for name, (_, _, tok) in CORPORA.items()
}

#: tokenizer name -> the directory (or Hub id) to load it from, taken from
#: `src.data.tokenizer`'s own registry rather than written out again here: a
#: second copy is a second place for the two to drift, and drift here is
#: invisible (see `resolve_tokenizer` below for what re-tokenizing with the
#: wrong tokenizer does).
TOKENIZER_REPOS = {name: cls._DEFAULT_REPO for name, cls in TOKENIZER_CLASSES.items()}


def resolve_tokenizer(corpus: str, requested: str | None) -> tuple[str, str]:
    """`(tokenizer_name, repo)` for a corpus. Returns the corpus's OWN tokenizer.

    The corpus is a frozen, already-tokenized artifact, but this driver does not
    read its ids: `build_probe_batch` re-tokenizes `doc["text"]`. So the
    tokenizer is not a preference, it is part of the corpus, and taking it from
    anywhere else is wrong in one of two ways:

    - **wrong tokenizer has the LARGER vocabulary** (llama3's 128,256 ids against
      a 49,152-vocab model): ids land past the embedding table and the forward
      dies on the device with an index-out-of-bounds. Loud, and it once took
      out a whole ablation batch.
    - **wrong tokenizer has the SMALLER vocabulary**: every id is in range, the
      forward succeeds, and the artifact is written with different text
      segmentation AND the wrong separator position excluded
      (`exclusion_rule_for_tokenizer` follows the NAME) -- one real token
      dropped, one separator kept, in every document. Nothing raises. This is
      the direction that has to be caught here, because nothing downstream can.

    `requested` (the `--tokenizer` flag) is an assertion, not an override: it is
    accepted only when it names the same tokenizer the corpus declares.
    """
    tokenizer_name = CORPORA[corpus][2]
    if requested is not None and requested != tokenizer_name:
        raise ValueError(
            f"--tokenizer {requested!r} contradicts corpus {corpus!r}, which is "
            f"tokenized with {tokenizer_name!r}. The corpus decides: this driver "
            f"re-tokenizes the corpus text, so a different tokenizer changes both "
            f"the token ids and the excluded separator position."
        )
    if TOKENIZER_REPOS[tokenizer_name] is None:
        raise ValueError(
            f"corpus {corpus!r} is tokenized with {tokenizer_name!r}, whose tokenizer files "
            "are not bundled with this repository (see src/data/tokenizer.py)."
        )
    return tokenizer_name, TOKENIZER_REPOS[tokenizer_name]


def _subsample_rows(n_rows: int, keep: int, *, seed: int) -> "np.ndarray | None":
    """Row indices to keep, or `None` meaning "keep everything".

    `keep <= 0` or `keep >= n_rows` returns `None` rather than an index array:
    writing an identity index alongside a full array would invite a reader to
    believe a selection happened. A drawn subset is written WITH its indices, so
    a row can always be traced back to the token it came from.
    """
    if keep <= 0 or keep >= n_rows:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=keep, replace=False))


class _CrossArchCollector(NullCollector):
    """Turns one probe forward into one metric row per `unrolled_pos`, keeping
    the raw routing tensors for the dump.

    Keyed by `unrolled_pos`, the depth coordinate, because head, tail and the
    loop's first layer all report `(loop_step, layer_idx) == (0, 0)`: keying on
    that pair would silently collapse three rows into one.
    """

    def __init__(
        self,
        *,
        c_null_tokens: int,
        c_null_seed: int,
        hidden_subsample_tokens: int,
        hidden_seed: int,
        per_chunk_hidden: int,
        logit_tokens: int | None = None,
        logit_seed: int = 0,
        logit_dtype: str = "float16",
    ) -> None:
        self._c_null_tokens = c_null_tokens
        self._c_null_seed = c_null_seed
        self._logit_tokens = logit_tokens
        self._logit_seed = logit_seed
        if logit_dtype not in LOGIT_DTYPES:
            raise ValueError(f"logit_dtype must be one of {sorted(LOGIT_DTYPES)}, got {logit_dtype!r}")
        self._logit_dtype = LOGIT_DTYPES[logit_dtype]
        self._logit_dtype_name = logit_dtype
        #: Raw E-dimensional scores, kept apart from `dump` so they land in
        #: their own uncompressed file.
        self.logit_dump: dict[str, np.ndarray] = {}
        self._hidden_subsample_tokens = hidden_subsample_tokens
        self._hidden_seed = hidden_seed
        self._per_chunk_hidden = per_chunk_hidden
        self._n_chunks_done = 0
        self.meta: TraceMeta | None = None
        self.dump: dict[str, np.ndarray] = {}
        # One entry per cell, appended to once per chunk. The metrics are
        # computed on the concatenation at the end, never per chunk and
        # averaged: `q` is a ratio over all tokens, and a mean of per-chunk
        # ratios is a different quantity that would look entirely plausible.
        self._idx: dict[int, list[np.ndarray]] = {}
        self._weight: dict[int, list[np.ndarray]] = {}
        self._logits: dict[int, list[np.ndarray]] = {}
        self._hidden: dict[int, list[np.ndarray]] = {}
        self._hidden_index: dict[int, list[np.ndarray]] = {}
        #: unrolled_pos -> {block, loop_step, layer_idx, n_experts}. `n_experts`
        #: is per cell on purpose: head and tail keep E=8 whatever the loop uses.
        self._cell_meta: dict[int, dict[str, Any]] = {}
        self._tokens_seen = 0

    def on_probe_step(
        self, step: int, sink: TraceSink, meta: TraceMeta, per_token_loss: Tensor
    ) -> dict[str, float]:
        # Keyed by `unrolled_pos`: head = 0, the loop's layers
        # in execution order, tail last. Positions must be a contiguous range --
        # a gap means a layer did not report, and the depth curve would have a
        # hole that reads as a shallower network.
        # Pre-head/tail model code keys the sink by `(loop_step, layer_idx)`
        # rather than by `unrolled_pos`. Re-key from each point's own
        # `unrolled_pos` -- the adapter has already computed it, so this moves
        # the answer rather than deriving a second one. Only tuple keys are
        # rewritten; a native sink is left exactly as it is.
        if sink and all(isinstance(k, tuple) for k in sink):
            rekeyed = {int(point.unrolled_pos): point for point in sink.values()}
            if len(rekeyed) != len(sink):
                raise ValueError(
                    f"re-keying a legacy trace sink collapsed {len(sink)} cells "
                    f"into {len(rekeyed)}: two layers claim one unrolled_pos. "
                    "The depth coordinate would address the wrong cell."
                )
            sink = rekeyed
        positions = sorted(sink)
        if positions != list(range(len(positions))):
            raise ValueError(
                f"trace sink covers positions {positions}, which is not a "
                "contiguous 0..n-1 range; a layer did not report."
            )
        n_loop_cells = meta.num_stacks * meta.num_layers_in_stack
        if len(positions) < n_loop_cells:
            raise ValueError(
                f"{len(positions)} traced positions is fewer than the loop's own "
                f"{n_loop_cells}; the grid contradicts the metadata."
            )

        # Everything below copies to host memory before returning: the sink's
        # tensors are views that stop being valid when this call ends
        # (loop_trace's lifetime contract), so anything kept must be copied here.
        chunk_tokens = 0
        loop_expert_counts = set()
        for cell in positions:
                point = sink[cell]
                # ⚠️ E comes from the tensor, never from TraceMeta. Head and tail
                # keep E=8 whatever the loop block uses (S4: 64 in the loop, 8 in
                # its head) and TraceMeta carries one number. Dividing a
                # fully-utilised 8-expert head by 64 puts a floor of
                # 1 - 8/64 = 0.875 under L2, so every configuration's outer
                # layers would report severe collapse while perfectly balanced --
                # no error, plausible values, on the very layers under study.
                n_experts_here = int(point.router_logits.shape[-1])
                declared = getattr(point, "num_experts", None)
                if declared is not None and int(declared) != n_experts_here:
                    raise ValueError(
                        f"position {cell} ({point.block}): TracePoint declares "
                        f"num_experts={declared}, router_logits has "
                        f"{n_experts_here}. Two sources of one fact disagree."
                    )
                if point.block == "loop":
                    loop_expert_counts.add(n_experts_here)
                self._cell_meta[cell] = {
                    "block": point.block,
                    "loop_step": point.loop_step,
                    "layer_idx": point.layer_idx,
                    "n_experts": n_experts_here,
                }
                # (B, S, ...) -> (N, ...): the metrics are over tokens, and the
                # batch/sequence split carries no meaning for them.
                idx = point.topk_idx.reshape(-1, point.topk_idx.shape[-1]).cpu().numpy()
                weight = (
                    point.topk_weights.reshape(-1, point.topk_weights.shape[-1])
                    .float().cpu().numpy()
                )
                logits = (
                    point.router_logits.reshape(-1, point.router_logits.shape[-1])
                    .float().cpu().numpy()
                )
                self._idx.setdefault(cell, []).append(idx)
                self._weight.setdefault(cell, []).append(weight)
                self._logits.setdefault(cell, []).append(logits)
                chunk_tokens = idx.shape[0]

        if loop_expert_counts and loop_expert_counts != {int(meta.num_experts)}:
            raise ValueError(
                f"loop cells report expert counts {sorted(loop_expert_counts)} "
                f"but TraceMeta says {meta.num_experts}. Inside the loop these "
                "must agree; a mismatch means the metadata describes a "
                "different model from the one that ran."
            )

        if self._hidden_subsample_tokens > 0:
            per_chunk = self._per_chunk_hidden
            rng = np.random.default_rng(self._hidden_seed + self._n_chunks_done)
            for cell, point in sorted(sink.items()):
                flat = point.residual.reshape(-1, point.residual.shape[-1])
                take = min(per_chunk, flat.shape[0])
                pick = rng.choice(flat.shape[0], size=take, replace=False)
                rows_ = flat[torch.as_tensor(pick, device=flat.device)]
                self._hidden.setdefault(cell, []).append(
                    rows_.float().cpu().numpy().astype(np.float16)
                )
                # Global token index, so a consumer can line a residual up with
                # the routing decision for the same token.
                self._hidden_index.setdefault(cell, []).append(
                    (pick + self._tokens_seen).astype(np.int32)
                )

        self._tokens_seen += chunk_tokens
        self._n_chunks_done += 1
        self.meta = meta
        return {}

    def finalize(self) -> list[dict[str, Any]]:
        """Metrics over ALL chunks, plus the arrays to dump.

        Called once, after every chunk has been through `on_probe_step`. The
        concatenation happens first and the metrics once: `q` is a share of all
        tokens, so averaging per-chunk values would silently answer a different
        question.
        """
        assert self.meta is not None
        rows = []
        for cell in sorted(self._idx):
            info = self._cell_meta[cell]
            idx = np.concatenate(self._idx[cell])
            weight = np.concatenate(self._weight[cell])
            logits = np.concatenate(self._logits[cell])
            row = collapse_metrics(
                topk_idx=idx,
                topk_weight=weight,
                router_logits=logits,
                n_experts=info["n_experts"],
                c_null_tokens=self._c_null_tokens,
                c_null_seed=self._c_null_seed,
            )
            # `unrolled_pos` is THE depth coordinate and the only unique one.
            # `loop_step`/`layer_idx` are carried through but mean nothing
            # unless block == "loop"; branch on `block`, never on
            # `loop_step == 0`, which head and tail also satisfy.
            row["unrolled_pos"] = cell
            row["block"] = info["block"]
            row["loop_step"] = info["loop_step"]
            row["layer_idx"] = info["layer_idx"]
            row["position_is_loop"] = info["block"] == "loop"
            rows.append(row)

            tag = f"p{cell}"
            self.dump[f"topk_idx_{tag}"] = idx.astype(np.int16)
            self.dump[f"topk_weight_{tag}"] = weight.astype(np.float16)
            # Raw E-dimensional scores, into a SEPARATE dump (see `main`): they
            # are ~10x the size of everything else here and the existing file is
            # written compressed, which costs 10x the time for 8% of the bytes
            # on data this close to noise (measured).
            if self._logit_tokens is not None:
                keep = _subsample_rows(
                    logits.shape[0], self._logit_tokens, seed=self._logit_seed
                )
                # float16 by default: these files are already the bulk of a
                # product. It is lossy for RE-DERIVING the selection -- ties and
                # near-ties within half a float16 ulp flip, measured 2.2e-4 to
                # 8.1e-4 of token-positions on S1 -- which is why `topk_idx_*`
                # (the selection the forward actually made) is always written to
                # the main dump and is the authority. `--logits-dtype float32`
                # doubles these files and removes the re-derivation gap.
                self.logit_dump[f"logits_{tag}"] = (
                    logits if keep is None else logits[keep]
                ).astype(self._logit_dtype)
                if keep is not None:
                    self.logit_dump[f"token_index_{tag}"] = keep.astype(np.int32)
            if cell in self._hidden:
                self.dump[f"residual_block_output_{tag}"] = np.concatenate(self._hidden[cell])
                self.dump[f"residual_token_index_{tag}"] = np.concatenate(
                    self._hidden_index[cell]
                )
        return rows


def compute_cross_arch(
    model: Any,
    *,
    tokenizer: Any,
    device: str,
    probe_dtype: str,
    n_documents: int,
    c_null_tokens: int,
    c_null_seed: int,
    hidden_subsample_tokens: int,
    hidden_seed: int,
    doc_chunk: int,
    corpus_path: str | Path,
    corpus_id: str,
    tokenizer_name: str,
    logit_tokens: int | None = None,
    logit_seed: int = 0,
    logit_dtype: str = "float16",
) -> dict[str, Any]:
    """Metric rows plus provenance and the arrays to dump.

    Does no checkpoint or file I/O, so a test can call it with a toy model.
    `hidden_subsample_tokens = 0` switches the residual dump off, which is what
    trajectory archives use.
    """
    input_ids, token_source = build_probe_batch(
        tokenizer, corpus_path, batch=n_documents, seq=PROBE_SEQ,
        exclusion_rule=exclusion_rule_for_tokenizer(tokenizer_name), seed=0,
    )
    # 500 documents in one forward needs 122 GiB on a 1.3B model at float32 --
    # measured, not estimated: it is what the einsum asked for on an 80 GiB
    # card. Chunking is over DOCUMENTS, so each chunk holds whole documents and
    # the token set is exactly the same as an unchunked run would have seen.
    chunks = [
        input_ids[i : i + doc_chunk] for i in range(0, input_ids.shape[0], doc_chunk)
    ]
    n_chunks = len(chunks)
    per_chunk_hidden = -(-hidden_subsample_tokens // n_chunks) if n_chunks else 0

    collector = _CrossArchCollector(
        c_null_tokens=c_null_tokens,
        c_null_seed=c_null_seed,
        hidden_subsample_tokens=hidden_subsample_tokens,
        hidden_seed=hidden_seed,
        per_chunk_hidden=per_chunk_hidden,
        logit_tokens=logit_tokens,
        logit_seed=logit_seed,
        logit_dtype=logit_dtype,
    )
    for chunk in chunks:
        run_probe_forward(model, chunk.to(device), collector=collector, step=0)
    rows = collector.finalize()
    assert collector.meta is not None

    meta = collector.meta
    corpus_bytes = Path(corpus_path).read_bytes()
    return {
        "producer": {"name": "offline_cross_arch_probe"},
        "cross_arch": {
            "kind": "cross_arch_collapse_probe",
            "spec_version": "v1",
            "num_layers_in_stack": meta.num_layers_in_stack,
            "num_stacks": meta.num_stacks,
            "num_experts": meta.num_experts,
            "num_active": meta.num_active,
            "n_documents": int(n_documents),
            "tokens_per_document": int(token_source["tokens_per_doc"]),
            "corpus_id": corpus_id,
            "tokenizer_name": tokenizer_name,
            "exclusion_rule": exclusion_rule_for_tokenizer(tokenizer_name),
            "corpus_file_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            "c_null_tokens": int(c_null_tokens),
            "c_null_seed": int(c_null_seed),
            "hidden_subsample_tokens": int(hidden_subsample_tokens),
            "hidden_seed": int(hidden_seed),
            "doc_chunk": int(doc_chunk),
            "n_forward_chunks": n_chunks,
            **dtype_fields(model, probe_dtype),
            "by_layer_loop": rows,
        },
        "token_source": token_source,
        "_dump": collector.dump,
        "_logit_dump": collector.logit_dump,
    }



def row_grid_is_complete(rows: list[dict]) -> bool:
    """Whether an artifact holds every layer row its model produces.

    The count is `head layers + num_stacks * num_layers_in_stack + tail layers`.
    Only the middle term is derivable from the model-level numbers every row
    carries: head and tail counts are per-configuration and are not recorded in
    the artifact, so they are read from the rows' own `block` labels and the
    grid is checked for gaps instead (`unrolled_pos` is the sink key, so a
    complete artifact's positions are exactly `0 .. len(rows) - 1`).

    Checking `len(rows) == num_stacks * num_layers_in_stack` -- what this did
    before -- is not merely loose for an ablation model, it is never true for
    one: S1 writes 18 rows (16 loop + 1 head + 1 tail) against a product of 16,
    so every artifact is judged incomplete forever and the batch can never
    finish.

    **Artifacts written before the head/tail change have no `block` field at
    all**, and are judged by the contract they were written under (the bare
    product). Applying the new rule to them would declare all 105 existing
    cross-arch artifacts incomplete and silently recompute the lot.

    Limitation, stated because a caller may care: head and tail counts come from
    the file itself, so an artifact that lost its trailing tail row to a
    truncated write still passes. The loop block -- all but two of the rows, and
    all of the compute -- is checked against numbers the file cannot fake.
    """
    if not rows:
        return False
    if "block" not in rows[0]:
        return len(rows) == rows[0]["num_stacks"] * rows[0]["num_layers_in_stack"]
    blocks = [r["block"] for r in rows]
    if set(blocks) - {"head", "loop", "tail"}:
        return False
    n_loop = blocks.count("loop")
    if n_loop != rows[0]["num_stacks"] * rows[0]["num_layers_in_stack"]:
        return False
    return sorted(r["unrolled_pos"] for r in rows) == list(range(len(rows)))


#: `TracePoint` fields that only exist after the head/tail change. A checkpoint
#: whose bundled `loop_trace.py` lacks them was trained before it.
_POST_HEAD_TAIL_FIELDS = frozenset({"block", "unrolled_pos", "num_experts", "top_k"})


def _bundled_trace_is_legacy(checkpoint_dir: "Path") -> bool:
    """Whether this checkpoint's own `loop_trace.py` predates head/tail.

    Detected by the FIELD SET, not by a file hash. The hash changes for reasons
    that do not affect behaviour -- a reformat, an edited comment -- and would
    call an unchanged contract "legacy"; the field set is the thing that decides
    whether the bundled model code can construct a `TracePoint` at all.

    A checkpoint with no bundled `loop_trace.py` is not legacy: it has nothing of
    its own to disagree with.
    """
    path = Path(checkpoint_dir) / "loop_trace.py"
    if not path.exists():
        return False
    source = path.read_text(errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TracePoint":
            annotated = {
                t.target.id
                for t in node.body
                if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
            }
            return not _POST_HEAD_TAIL_FIELDS.issubset(annotated)
    return False


class LegacyShimUnavailable(RuntimeError):
    """The compatibility shim a legacy checkpoint needs no longer exists.

    Raised rather than degrading: without the substitution the bundled model
    code constructs its own `TracePoint` with the pre-head/tail keyword set, and
    the probe would then write rows that look entirely ordinary while describing
    a different trace contract. A named error at load time costs one run; a
    silent contract mismatch costs every conclusion drawn from the batch.
    """


@contextlib.contextmanager
def _legacy_trace_compat(num_layers_in_stack: int):
    """Make a pre-head/tail checkpoint's bundled model code constructible.

    That code does `from pretrain.hooks.loop_trace import TracePoint` at import
    time and then calls it with the old keyword set, so the substitution has to
    be in place before `from_pretrained` and has to be on the module the bundled
    code actually reaches -- the compatibility shim, which re-exports
    `src.model.loop_trace`.

    Restored on exit. The patch is process-wide while held, which is acceptable
    in a single-checkpoint driver and would not be in a library.
    """
    # ⚠️ This is the ONE import through the old `pretrain.` compatibility layer
    # under src/, needed only for checkpoints that bundle the old module path; it
    # raises LegacyShimUnavailable when that layer is absent. It cannot be replaced by
    # importing src.model.loop_trace: the bundled code reaches the shim module
    # object, and that is the object the substitution has to land on.
    try:
        import pretrain.hooks.loop_trace as shim
    except ModuleNotFoundError as exc:  # phase three: the shims are gone
        raise LegacyShimUnavailable(
            "this legacy checkpoint needs `pretrain.hooks.loop_trace`, which no "
            "longer exists. Its bundled model code imports `TracePoint` from "
            "that path, so the substitution cannot be installed and the probe "
            "would silently record a pre-head/tail trace contract. Re-export the "
            "shim, or probe this checkpoint with a revision that still has it."
        ) from exc
    import src.model.loop_trace as impl

    factory = legacy_trace_point_factory(num_layers_in_stack)
    saved = [(m, m.TracePoint) for m in (shim, impl)]
    for module, _ in saved:
        module.TracePoint = factory
    try:
        yield
    finally:
        for module, original in saved:
            module.TracePoint = original


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out", required=True, help="Output JSONL path (one line per cell)")
    ap.add_argument("--dump-npz", required=True, help="Where to write the routing dump")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_probe_dtype_argument(ap)
    ap.add_argument(
        "--n-documents", type=int, required=True,
        help="Documents in the probe batch. REQUIRED: the corpus is moving from "
        "40 to 500 documents and N is what the finite-sample requirement is "
        "stated against, so it must not be inherited from a default.",
    )
    ap.add_argument(
        "--c-null-tokens", type=int, required=True,
        help="Draws for the indifferent-router reference (at least 1e5 recommended).",
    )
    ap.add_argument(
        "--c-null-seed", type=int, required=True,
        help="Seed for that reference. Recorded in every row so C_null can be "
        "recomputed exactly rather than approximately.",
    )
    ap.add_argument(
        "--hidden-subsample-tokens", type=int, required=True,
        help="Residual vectors to keep per cell; 0 switches the dump off. Only "
        "the final-state checkpoints take a nonzero value -- the trajectory "
        "archives are 16x more numerous and nobody has asked the question yet.",
    )
    ap.add_argument("--hidden-seed", type=int, required=True)
    ap.add_argument(
        "--doc-chunk", type=int, required=True,
        help="Documents per forward pass. REQUIRED: 500 at once needs 122 GiB "
        "on an 80 GiB card. Chunking is over whole documents, so the token set "
        "is identical to an unchunked run -- only the peak memory differs.",
    )
    ap.add_argument(
        "--corpus", required=True, choices=sorted(CORPORA),
        help="Which probe corpus to read. REQUIRED: the 40-document and "
        "500-document corpora give different N, and the finite-sample "
        "requirement is stated against N.",
    )
    ap.add_argument(
        "--dump-router-logits", default=None,
        help="Path for the raw E-dimensional router scores (float16, "
        "UNCOMPRESSED .npz). Omitted, they are not written at all and the run "
        "behaves exactly as before. Required together with --logits-tokens.",
    )
    ap.add_argument(
        "--logits-tokens", type=int, default=None,
        help="Tokens of raw scores to keep PER POSITION. 0 means every token. "
        "REQUIRED whenever --dump-router-logits is given, with no default: how "
        "many tokens a distribution was measured on changes what the "
        "distribution means, and a silent default would put different steps on "
        "different sample sizes with nothing in the artifact to show it.",
    )
    ap.add_argument(
        "--logits-dtype", default="float16", choices=sorted(LOGIT_DTYPES),
        help="Storage type for the raw router scores (default float16). float16 halves "
        "the file but is lossy for RE-DERIVING the top-k: near-ties flip, measured 2.2e-4 "
        "to 8.1e-4 of token-positions on S1. The selection the forward made is always "
        "written separately as `topk_idx_*` and is the authority; use float32 when the "
        "scores themselves must reproduce the selection.",
    )
    ap.add_argument(
        "--logits-seed", type=int, default=None,
        help="Seed for the subsample draw. REQUIRED with --logits-tokens > 0.",
    )
    ap.add_argument(
        "--tokenizer", default=None, choices=sorted(TOKENIZER_REPOS),
        help="Assert which tokenizer the corpus uses. Optional, and NOT an "
        "override: the corpus declares its own tokenizer and a value that "
        "disagrees is refused. Omitted -- the normal case -- the corpus decides.",
    )
    args = ap.parse_args(argv)
    if (args.dump_router_logits is None) != (args.logits_tokens is None):
        ap.error(
            "--dump-router-logits and --logits-tokens must be given together: "
            "a path with no token count would write an unlabelled sample, and a "
            "token count with no path would silently discard the scores."
        )
    if args.logits_tokens is not None and args.logits_tokens > 0 and args.logits_seed is None:
        ap.error("--logits-seed is required when --logits-tokens > 0: the draw "
                 "must be reproducible from the artifact alone.")

    from src.pretrain.checkpointing.store import MANIFEST_NAME, is_complete
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not is_complete(checkpoint_dir):
        raise ValueError(f"{checkpoint_dir} has no valid manifest.json -- refusing to probe it.")
    weights = checkpoint_dir / WEIGHTS_FILENAME
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError(f"{checkpoint_dir} has a manifest but {WEIGHTS_FILENAME} is missing/empty.")
    with open(checkpoint_dir / MANIFEST_NAME) as fh:
        manifest = json.load(fh)

    # Resolved BEFORE the forward pass, not after: it is knowable in the first
    # second, and a provenance failure discovered at write time throws away all
    # the compute.
    from src.metrics.provenance import code_revision, machine_provenance

    revision = code_revision()

    # Recorded from the checkpoint's own manifest. Absent on every archive
    # predating the ablation recipe, and absence is written down as absence:
    # omitting the field reads as "we forgot to record it", and filling in a
    # plausible "none"/"full" claims knowledge of how a run was trained that
    # this side does not have. Same rule as snapshot_commit.
    ac_mode = manifest.get("activation_checkpoint_mode")
    train_ac = {
        "train_ac_mode": ac_mode,
        "train_ac_mode_source": "manifest" if ac_mode is not None else "absent_from_manifest",
    }

    print(
        f"[offline_cross_arch_probe] checkpoint: {checkpoint_dir} "
        f"(step={manifest.get('step')}, kind={manifest.get('kind')})"
    )
    corpus_path, corpus_id, _ = CORPORA[args.corpus]
    tokenizer_name, tokenizer_repo = resolve_tokenizer(args.corpus, args.tokenizer)
    print(
        f"[offline_cross_arch_probe] corpus={args.corpus} id={corpus_id} "
        f"tokenizer={tokenizer_name} rule={exclusion_rule_for_tokenizer(tokenizer_name)} "
        f"docs={args.n_documents} seq={PROBE_SEQ} "
        f"c_null_tokens={args.c_null_tokens} seed={args.c_null_seed} "
        f"hidden_subsample={args.hidden_subsample_tokens} doc_chunk={args.doc_chunk}"
    )

    # Pre-head/tail checkpoints bundle model code that constructs `TracePoint`
    # with the old keyword set. Without the adapter the load fails with a
    # TypeError raised inside the bundled file, before any forward pass.
    legacy_trace = _bundled_trace_is_legacy(checkpoint_dir)
    if legacy_trace:
        n_layers = int(manifest.get("num_layers_in_stack") or 0) or None
        if n_layers is None:
            import json as _json
            cfg = _json.loads((Path(checkpoint_dir) / "config.json").read_text())
            n_layers = int(cfg["num_layers_in_stack"])
        print(f"[offline_cross_arch_probe] legacy trace contract: bundled "
              f"loop_trace.py predates head/tail; adapting "
              f"(num_layers_in_stack={n_layers})")
        ctx = _legacy_trace_compat(n_layers)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, trust_remote_code=True, **load_kwargs(args.probe_dtype)
        ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

    result = compute_cross_arch(
        model,
        tokenizer=tokenizer,
        device=args.device,
        probe_dtype=args.probe_dtype,
        n_documents=args.n_documents,
        c_null_tokens=args.c_null_tokens,
        c_null_seed=args.c_null_seed,
        hidden_subsample_tokens=args.hidden_subsample_tokens,
        hidden_seed=args.hidden_seed,
        doc_chunk=args.doc_chunk,
        corpus_path=corpus_path,
        corpus_id=corpus_id,
        tokenizer_name=tokenizer_name,
        logit_tokens=args.logits_tokens,
        logit_seed=args.logits_seed or 0,
        logit_dtype=args.logits_dtype,
    )
    dump = result.pop("_dump")
    logit_dump = result.pop("_logit_dump")
    result["producer"]["code_revision"] = revision
    # What it ran ON, not just what code ran. Written here rather than at
    # `result` construction so it records the machine that did the forward
    # passes. See `machine_provenance` for why every field is nullable.
    result["producer"]["machine"] = machine_provenance()
    # Which tracing contract the checkpoint's own model code speaks. Recorded
    # because it changes how the rows were produced, and a reader comparing a
    # legacy-adapted archive against a native one should be able to see that
    # from the artifact rather than infer it from the checkpoint's date.
    result["producer"]["trace_contract"] = (
        "legacy_loop_only" if legacy_trace else "head_tail"
    )
    result["producer"].update(train_ac)

    dump_path = Path(args.dump_npz)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dump_path, **dump)

    # Raw scores go to their own file, UNCOMPRESSED. Measured on our
    # filesystem: compressing float16 router scores costs 1.27s per
    # 31 MiB against 0.12s, and returns 8% of the bytes -- they are too close to
    # noise to compress. At S4's 507 MiB per archive that is the difference
    # between 2 seconds and 21.
    if args.dump_router_logits is not None:
        logits_path = Path(args.dump_router_logits)
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(logits_path, **logit_dump)
        n_bytes = sum(a.nbytes for a in logit_dump.values())
        print(f"[offline_cross_arch_probe] wrote {logits_path} "
              f"({len(logit_dump)} arrays, {n_bytes / 1024**2:.1f} MiB)")

    # Written on every row so a reader never has to infer the sampling from the
    # file size: "all tokens" and "50,000 drawn" are different measurements and
    # the difference is invisible in the numbers themselves.
    result["cross_arch"].update({
        "router_logits_dumped": args.dump_router_logits is not None,
        "router_logits_tokens": (
            None if args.logits_tokens is None
            else ("all" if args.logits_tokens == 0 else int(args.logits_tokens))
        ),
        "router_logits_seed": args.logits_seed,
        # What the file actually holds, not a constant: a product written with
        # --logits-dtype float32 that claimed float16 would send every reader to
        # the wrong conclusion about whether the scores reproduce the selection.
        "router_logits_dtype": (None if args.logits_tokens is None else args.logits_dtype),
        "router_logits_reproduce_topk": (
            None if args.logits_tokens is None else
            ("exactly (float32)" if args.logits_dtype == "float32" else
             "approximately: float16 storage flips near-ties (~1e-4..1e-3 of token-positions); "
             "`topk_idx_*` in the main dump is the selection the forward made and is the authority")
        ),
    })
    if args.dump_router_logits is not None:
        result["cross_arch"].update(
            named_checkpoint(Path(args.dump_router_logits), field="router_logits")
        )

    blob = result["cross_arch"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for cell in blob["by_layer_loop"]:
            row = {
                "corpus": corpus_block([{"probe_seed": 0, **result["token_source"]}]),
                **named_checkpoint(checkpoint_dir),
                **named_checkpoint(dump_path, field="dump"),
                "manifest_step": manifest["step"],
                "manifest_kind": manifest["kind"],
                "producer": result["producer"],
                **{k: v for k, v in blob.items() if k != "by_layer_loop"},
                **cell,
            }
            # Checked before the file is opened elsewhere in this project; here
            # the file is already open, so the guard runs per row before write.
            assert_no_account_paths(row, what=f"{out_path}")
            fh.write(json.dumps(row) + "\n")
    print(f"[offline_cross_arch_probe] wrote {out_path} ({len(blob['by_layer_loop'])} rows)")
    print(f"[offline_cross_arch_probe] wrote {dump_path} ({len(dump)} arrays)")


if __name__ == "__main__":
    main()

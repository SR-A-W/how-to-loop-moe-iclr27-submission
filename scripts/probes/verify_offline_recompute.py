"""Verify that recomputing the archive-level bootstrap OFFLINE, from a product's saved
score matrices, reproduces the online numbers BIT FOR BIT.

Why this exists: a judgement quantity and its uncertainty
must come from the same code on both sides, or the difference between two readings
carries "the difference between two implementations" inside it, where it cannot be told
apart from a real difference. Recomputing offline is allowed only because it is the SAME
implementation at a second moment -- and that equivalence is the one link nobody had
checked. This script checks it.

Bit-for-bit, not "close": a downcast or a reordered reduction produces numbers that are
very close and read as fine. Every float is compared by its exact 8 bytes, so "close"
fails here.

The recompute runs on the SAME device the online run used. `archive_document_bootstrap`
takes its device from the cells and seeds its generator there; recomputing on CPU would
compare a device difference, not an implementation difference.

Usage:
    python scripts/probes/verify_offline_recompute.py <product_dir> [--device cuda]

Writes `offline_recompute_check.json` into the product directory, recording the commit,
the parameters, and the sha256 of every score matrix the recompute consumed, so the
result can later answer "which matrices produced this number".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import torch


def _bits(x: Any) -> Any:
    """A value's exact bytes, so comparison cannot be loosened by float formatting.

    NaN compares unequal to itself, and 0.0 == -0.0, under `==`. Both would let a real
    difference pass as equal, so floats are compared as their IEEE-754 bytes instead.
    """
    if isinstance(x, bool):
        return ("bool", x)
    if isinstance(x, float):
        return ("f64", struct.pack("<d", x))
    if isinstance(x, int):
        return ("int", x)
    if isinstance(x, str):
        return ("str", x)
    if x is None:
        return ("none",)
    if isinstance(x, (list, tuple)):
        return ("seq", tuple(_bits(v) for v in x))
    if isinstance(x, dict):
        return ("map", tuple(sorted((k, _bits(v)) for k, v in x.items())))
    raise TypeError(f"unhashable-for-comparison type {type(x).__name__}: {x!r}")


def _diff(online: dict, offline: dict, path: str = "") -> list[str]:
    """Every key whose value differs, named. Returns [] when identical."""
    out: list[str] = []
    for k in sorted(set(online) | set(offline)):
        p = f"{path}.{k}" if path else k
        if k not in online:
            out.append(f"{p}: missing online, offline={offline[k]!r}")
        elif k not in offline:
            out.append(f"{p}: missing offline, online={online[k]!r}")
        elif isinstance(online[k], dict) and isinstance(offline[k], dict):
            out.extend(_diff(online[k], offline[k], p))
        elif _bits(online[k]) != _bits(offline[k]):
            out.append(f"{p}: online={online[k]!r} offline={offline[k]!r}")
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("product", help="a finished product directory (must hold DONE.json and scores_layer*.pt)")
    ap.add_argument("--device", required=True, help="device to recompute on; must match the online run's")
    a = ap.parse_args(argv)

    # Imported here so --help works without torch's CUDA init.
    from src.metrics.collector import MetricNotApplicable, archive_document_bootstrap
    from src.metrics.provenance import code_revision

    # Resolved FIRST, before any of the expensive work. It failed here once, after a
    # 31-minute recompute, because the compute node has no git -- the whole result was lost
    # to a condition that was knowable in the first second. Anything that can refuse should
    # refuse before the work, not after it.
    revision = code_revision()

    out = Path(a.product).resolve()
    done = json.loads((out / "DONE.json").read_text())
    rows = [json.loads(l) for l in (out / "router_probe.jsonl").read_text().splitlines()]
    meta = next(r for r in rows if r["kind"] == "adapter_meta")
    online = [r for r in rows if r["kind"] == "archive_document_bootstrap"]
    if not online:
        raise SystemExit(f"{out} holds no archive_document_bootstrap rows -- nothing to verify")

    n_boot = int(done["n_archive_bootstrap"])
    ndd = int(done["n_distinct_documents"])
    tpd = int(done["tokens_per_doc"])
    top_k, n_experts = int(meta["top_k"]), int(meta["n_experts"])

    # The matrices are the extra input this recompute has; record what they were.
    files = sorted(out.glob("scores_layer*.pt"))
    if not files:
        raise SystemExit(f"{out} holds no scores_layer*.pt -- the online run did not save them")
    matrices = {f.name: _sha256(f) for f in files}

    cells = []
    for f in files:
        blob = torch.load(f, map_location="cpu")
        scores = blob["scores"]
        if scores.dtype != torch.float32:
            raise SystemExit(
                f"{f.name} holds {scores.dtype}, not float32. A downcast matrix cannot reproduce "
                "the online numbers bit for bit, and would fail below for a reason that is not "
                "the one being tested."
            )
        # Centered exactly as the driver centers, on the same device as the online run.
        pooled_f = scores.to(a.device).float()
        cells.append({"centered": pooled_f - pooled_f.mean(dim=-1, keepdim=True),
                      "layer_idx": int(f.stem.removeprefix("scores_layer")), "loop_step": 0})

    mismatches: list[str] = []
    checked = 0
    for r in online:
        block: dict[str, Any] = {"kind": "archive_document_bootstrap", "quantity": r["quantity"],
                                 "shuffle_scope": r["shuffle_scope"]}
        try:
            block.update(archive_document_bootstrap(
                cells, top_k=top_k, n_experts=n_experts, n_bootstrap=n_boot,
                tokens_per_doc=tpd, n_distinct_documents=ndd,
                shuffle_scope=r["shuffle_scope"], shuffle_seed=int(done["args"]["shuffle_seed"]),
                quantity_name=r["quantity"],
            ))
            block["status"] = "ok"
        except MetricNotApplicable as exc:
            block["status"] = "not_applicable"
            block["reason"] = str(exc)

        # The online row also carries the adapter's identity tags, which the recompute does
        # not produce and which are not part of the computation. Compare the computed keys.
        online_computed = {k: v for k, v in r.items() if k in block}
        d = _diff(online_computed, block, f"{r['quantity']}/{r['shuffle_scope']}")
        missing = sorted(set(block) - set(r))
        if missing:
            d.append(f"{r['quantity']}/{r['shuffle_scope']}: offline produced keys absent online: {missing}")
        mismatches.extend(d)
        checked += 1
        print(f"[verify] {r['quantity']}/{r['shuffle_scope']}: "
              f"{'IDENTICAL' if not d and not missing else 'DIFFERS'} ({len(block)} keys)", file=sys.stderr)

    report = {
        "verdict": "bit_identical" if not mismatches else "MISMATCH",
        "product": out.name,
        "blocks_checked": checked,
        "mismatches": mismatches,
        "device": a.device,
        "shape": {"n_experts": n_experts, "top_k": top_k, "n_grid_points": len(cells)},
        # What this recompute was, so the number can be traced back later.
        "code_revision": revision,
        "online_code_revision": done["code_revision"],
        "parameters": {"n_bootstrap": n_boot, "n_distinct_documents": ndd, "tokens_per_doc": tpd,
                       "shuffle_seed": int(done["args"]["shuffle_seed"]),
                       "quantities": list(done["archive_quantities"])},
        # The extra input offline recompute has that the online run did not.
        "score_matrices_sha256": matrices,
    }
    (out / "offline_recompute_check.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"[verify] {report['verdict']} -- {checked} block(s), "
          f"{len(matrices)} matrices, wrote offline_recompute_check.json", file=sys.stderr)
    if mismatches:
        for m in mismatches[:20]:
            print("  " + m, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

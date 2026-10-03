#!/usr/bin/env python3
"""Fingerprint a model's INITIALISED weights on a chosen device.

Why this exists: the question "would a cold-start checkpoint generated on CPU
hold the same weights as one generated on GPU" cannot be answered by running the
production path on CPU, because torchtitan v0.2.2 has no CPU path -- its
`device_type` is `_get_available_device_type() or "cuda"`, so with no GPU visible
it still picks `cuda` and dies in `set_device` with "No CUDA GPUs are available".

So this probe reproduces exactly what the seed-checkpoint path does to
initialise a model, on whichever device it is told:

    torch.manual_seed(seed)            # what torchtitan does for world_size == 1
    build under torch.device("meta")   # torchtitan builds on meta
    model.to_empty(device=<device>)
    model.init_weights(buffer_device=<device>)

and hashes every resulting parameter and buffer. Run it twice, once per device,
and compare. Run it on the GPU and compare against the real checkpoint first --
that is what shows the probe reproduces the production artifact rather than
something merely similar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


#: The repository layout these scripts run in (`src/pretrain`), checked as a path.
REPO_LAYOUTS = (("src", "pretrain"),)


def check_repo_root(root):
    """Refuse a root that is neither layout, naming both so the message is usable."""
    if not any((root / pkg / sub).is_dir() for pkg, sub in REPO_LAYOUTS):
        wanted = " or ".join(f"{pkg}/{sub}" for pkg, sub in REPO_LAYOUTS)
        raise SystemExit(
            f"--repo-root {root} does not look like this repository (no {wanted})"
        )


def fingerprint(arch: str, seed: int, device: str) -> dict:
    import torch

    from src.model.config_registry import ABLATIONS, ATTENTION_VARIANTS, build, build_ablation
    from src.pretrain.train_spec import LoopMoEModel, LoopMoEModelArgs

    # Same two families the launcher knows about, resolved the same way, so the
    # probe cannot build something the production path would not.
    config = build_ablation(arch) if (arch in ABLATIONS or arch in ATTENTION_VARIANTS) else build(arch)

    # Exactly torchtitan's single-process seeding (distributed/utils.py:
    # `if parallel_dims.world_size == 1: torch.manual_seed(seed)`).
    torch.manual_seed(seed)

    with torch.device("meta"):
        model = LoopMoEModel(LoopMoEModelArgs(config=config))

    model.to_empty(device=torch.device(device))
    model.init_weights(buffer_device=torch.device(device))

    tensors = {}
    for name, t in sorted(model.state_dict().items()):
        t = t.detach().to("cpu").contiguous()
        raw = t.flatten().view(torch.uint8).numpy().tobytes()
        tensors[name] = {
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    joined = "".join(tensors[n]["sha256"] for n in sorted(tensors)).encode()
    return {
        "arch": arch,
        "seed": seed,
        "device": device,
        "torch_version": torch.__version__,
        "tensor_count": len(tensors),
        "rollup_sha256": hashlib.sha256(joined).hexdigest(),
        "tensors": tensors,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", required=True, help="an ablation configuration name, e.g. S1")
    parser.add_argument("--seed", type=int, required=True, help="the seed torchtitan would be given")
    parser.add_argument("--device", required=True, help="cpu or cuda")
    parser.add_argument("--out", required=True, help="where to write the JSON fingerprint")
    parser.add_argument(
        "--repo-root", required=True,
        help="the checkout whose code builds the model -- the same tree the checkpoint "
             "being compared against was generated from",
    )
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    check_repo_root(root)
    sys.path.insert(0, str(root))

    result = fingerprint(args.arch, args.seed, args.device)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"[init-probe] {args.arch} on {args.device}: {result['tensor_count']} tensors, "
          f"rollup={result['rollup_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

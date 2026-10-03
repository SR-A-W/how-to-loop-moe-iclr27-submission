"""Single-GPU smoke check: does the pinned stack actually run on this cluster's hardware?

Everything up to now was verified on CPU. Three claims have never been tested on
a real device, and each would break the first production run:

  1. **The cu126 wheel matches the driver.** The default torch wheel is +cu130 and
     cannot initialise CUDA here (driver 565.57.01 caps at 12.7). cu126 was chosen
     because it was known to run on these nodes -- good evidence, but not a test
     of this stack.
  2. **`grouped_mm` has a working CUDA kernel.** It is the reason for the
     torch>=2.10 floor, and CPU availability says nothing about the device path.
  3. **bf16 forward+backward is numerically sane on device.** LT2 trains in pure
     bf16 and so will we; the CPU tests exercise the dtype logic, not the kernels.

Prints a compact report and exits non-zero on failure so a batch job records it.
"""

from __future__ import annotations

import time

import torch


def main() -> int:
    print("=" * 62)
    print("GPU SMOKE — looped-MoE training stack")
    print("=" * 62)

    print(f"torch                 : {torch.__version__}")
    print(f"built for CUDA        : {torch.version.cuda}")
    print(f"cuda.is_available()   : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible")
        return 1

    print(f"device                : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"compute capability    : {cap[0]}.{cap[1]}")

    # 1. grouped_mm on device -- the whole reason for the torch>=2.10 pin.
    import torch.nn.functional as F

    if not hasattr(F, "grouped_mm"):
        print("FAIL: torch.nn.functional.grouped_mm missing")
        return 1
    E, d_in, d_out, n = 4, 32, 64, 40
    x = torch.randn(n, d_in, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(E, d_in, d_out, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([10, 20, 30, 40], device="cuda", dtype=torch.int32)
    out = F.grouped_mm(x, w, offs=offs)
    torch.cuda.synchronize()
    print(f"grouped_mm on CUDA    : OK  {tuple(out.shape)} {out.dtype}")

    # 2. Real model, bf16, forward + backward.
    from src.model.modeling_loop_lm import LoopMoEConfig, LoopMoEForCausalLM

    config = LoopMoEConfig(
        vocab_size=128256, context_length=512, d_model=256, num_heads=4, d_ff=512,
        rope_theta=10000.0, width_ratio=2.0, num_layers_in_stack=4, num_stacks=4,
        num_experts=16, num_active=2, lb_loss_factor=0.01, lz_loss_factor=0.001,
    )
    model = LoopMoEForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model (DVF-a shape)   : {nparams:,} params, L=4 R=4 E=16, bf16")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    B, S = 4, 512
    ids = torch.randint(0, config.vocab_size, (B, S), device="cuda")
    # `labels=ids` (unshifted) would be wrong -- LoopMoEForCausalLM.forward()
    # does not shift internally. Harmless for THIS script's purpose (random synthetic data,
    # only checking finite-loss/throughput, not model quality), but fixed for
    # consistency -- same call-site convention as src/eval/run_eval.py.
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]

    losses = []
    steps = 12
    for step in range(steps):
        if step == 2:  # exclude warmup/allocator noise from the timing
            torch.cuda.synchronize()
            started = time.time()
        out_ = model(ids, labels=labels)
        opt.zero_grad()
        out_.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(out_.loss.item())
    torch.cuda.synchronize()
    elapsed = time.time() - started
    timed_steps = steps - 2

    if not all(map(torch.isfinite, map(torch.tensor, losses))):
        print(f"FAIL: non-finite loss encountered: {losses}")
        return 1
    print(f"bf16 fwd+bwd          : OK  loss {losses[0]:.4f} -> {losses[-1]:.4f}")

    tokens = timed_steps * B * S
    tok_s = tokens / elapsed
    print(f"throughput            : {tok_s:,.0f} tokens/s  ({elapsed/timed_steps*1000:.0f} ms/step)")

    # Rough MFU, using the same accounting as the trainer: compute follows the
    # unrolled depth, and only `num_active` experts run per token.
    from src.pretrain.train_spec import LoopMoEModelArgs

    args = LoopMoEModelArgs(config=config)
    _, flops_per_token = args.get_nparams_and_flops(model, seq_len=S)
    achieved = tok_s * flops_per_token
    print(f"FLOPs/token (6N+attn) : {flops_per_token:,}")
    print(f"achieved              : {achieved/1e12:.1f} TFLOP/s")
    # H200 bf16 dense peak ~989 TFLOP/s; A100 ~312.
    name = torch.cuda.get_device_name(0).upper()
    peak = 989e12 if "H200" in name or "H100" in name else 312e12
    print(f"MFU (vs {'H200/H100' if peak > 5e14 else 'A100'} peak) : {100*achieved/peak:.1f}%")
    print("  NOTE: a 12-step micro-benchmark at batch 4 is not a production MFU;")
    print("        it is a sanity floor, not the number for the compute budget.")

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak memory           : {peak_mem:.2f} GB")
    print("=" * 62)
    print("GPU SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env -S /home/jovyan/Automodel/.pixi/envs/nemo/bin/python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TFLOP/s + IO-bandwidth benchmark for the GLM-5.2 DSA sparse-MLA backward kernels.

Compares the two TileLang backward implementations, which share an identical
public API and math:

  * ``tilelang_sparse_mla_bwd``             -- the vendored reference kernel.
  * ``tilelang_sparse_mla_bwd_gemm_fusion`` -- the GEMM-operand-reuse / split-D
    optimized variant (parity cosine ~1.0 vs the reference).

Timing uses TileLang's ``do_bench`` (JIT-warmed, L2 flushed before every iter so
gathered-KV reads are not served warm, returns mean wall-clock ms). The
ms -> TFLOP/s / TB/s conversion follows the upstream DeepSeek-V3.2 example: 5
re-derived GEMMs per selected KV per query token (see ``_flops_bandwidth``). The
op is IO-bound (sparse gather + fp32 atomic scatter), so the dominant KV traffic
is reported alongside TFLOP/s.

Full ``sparse_mla_bwd`` (preprocess + bwd + postprocess) is timed by default;
``--kernel-only`` isolates the ``bwd`` kernel, the sole part that differs.

Run (always use the pinned env):
  .pixi/envs/nemo/bin/python scripts/profile_glm_dsa_sparse_mla_bwd_gemm_fusion.py
  .pixi/envs/nemo/bin/python scripts/profile_glm_dsa_sparse_mla_bwd_gemm_fusion.py --seq-len 4096 --topk 2048 --heads 64
  .pixi/envs/nemo/bin/python scripts/profile_glm_dsa_sparse_mla_bwd_gemm_fusion.py --kernel-only --no-check
"""

from __future__ import annotations

import argparse
import os
import sys

# TileLang's auto-target detection + nvcc lookup both need CUDA_HOME; the pixi
# ``nemo`` env doubles as the CUDA root, derived here from this interpreter.
os.environ.setdefault("CUDA_HOME", os.path.dirname(os.path.dirname(sys.executable)))

import torch

# Fixed by the GLM-5.2 / DeepSeek sparse-MLA latent geometry (do not vary): the
# kernels hard-assert dim_plus_tail == 576 and D (value width) == 512.
DQKV = 576  # kv_lora_rank(512) + qk_rope_head_dim(64) -- the score/dQ/dKV width
DV = 512  # kv_lora_rank -- the dP/dV width


def _causal_topk_indices(num_tokens: int, topk: int, device: torch.device) -> torch.Tensor:
    """Causal top-k gather indices ``[S, topk]`` int32: token ``t`` selects
    ``min(t+1, topk)`` random keys from ``[0, t]``; the tail stays ``-1`` (masked).
    """
    idx = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)
    for t in range(num_tokens):
        n = min(t + 1, topk)
        idx[t, :n] = torch.randperm(t + 1, device=device)[:n].to(torch.int32)
    return idx


def _build_inputs(
    seq_len: int,
    seq_len_kv: int,
    heads: int,
    topk: int,
    kv_group: int,
    sm_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build the un-batched backward inputs ``(q, kv, o, do, indices, lse)``.

    Layouts: ``q[S, H, 576]``, ``kv[S_kv, G, 576]``, ``o/do[S, H, 512]``,
    ``indices[S, G, topk]`` int32, ``lse[S, H]`` fp32. ``o``/``lse`` come from the
    real forward so the backward runs on a consistent softmax state.
    """
    from nemo_automodel.components.models.glm_moe_dsa.kernels import tilelang_sparse_mla_fwd as fwd_mod

    q = torch.randn(seq_len, heads, DQKV, device=device, dtype=torch.bfloat16).contiguous()
    kv = torch.randn(seq_len_kv, kv_group, DQKV, device=device, dtype=torch.bfloat16).contiguous()
    indices = _causal_topk_indices(seq_len, topk, device).view(seq_len, kv_group, topk).contiguous()

    o, lse = fwd_mod.sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale)
    o = o.contiguous()
    lse = lse.contiguous()
    do = torch.randn_like(o)
    return q, kv, o, do, indices, lse


def _flops_bandwidth(ms: float, seq_len: int, heads: int, topk: int) -> tuple[float, float]:
    """Convert mean wall-clock ``ms`` to (TFLOP/s, IO TB/s) using the upstream accounting."""
    per_token_flop = 2 * (
        heads * DV * topk  # dP  : dO @ KV^T
        + heads * DQKV * topk  # P   : Q  @ KV^T  (fwd score recompute)
        + heads * DQKV * topk  # dQ  : dP @ KV
        + heads * DQKV * topk  # dKV : dP^T @ Q
        + heads * DV * topk  # dV  : P^T @ dO
    )
    tflops = per_token_flop * seq_len / (ms * 1e-3) / 1e12
    # B=1; per selected KV the dominant bf16 traffic (2 bytes/elt).
    io_tbs = seq_len * max(DQKV * 2, DQKV + DV) * topk * 2 / (ms * 1e-3) / 1e12
    return tflops, io_tbs


def _bench_e2e(
    mod, inputs: tuple[torch.Tensor, ...], sm_scale: float, warmup: float, rep: float, backend: str
) -> float:
    """Benchmark the full ``sparse_mla_bwd`` op (preprocess + bwd + postprocess)."""
    from tilelang.profiler import do_bench

    q, kv, o, do, indices, lse = inputs

    def fn():
        return mod.sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=sm_scale)

    return do_bench(fn, warmup=warmup, rep=rep, backend=backend)


def _bench_kernel_only(
    mod, inputs: tuple[torch.Tensor, ...], sm_scale: float, warmup: float, rep: float, backend: str
) -> float:
    """Benchmark only the ``bwd`` kernel -- the sole part that differs between variants.

    Delta is precomputed once (preprocess is identical and excluded from the FLOP
    model). ``dKV`` is reused without re-zeroing: it is an fp32 atomic-scatter
    target whose accumulated value is discarded and whose cost is value-independent.
    """
    from tilelang.profiler import do_bench

    q, kv, o, do, indices, lse = inputs
    q4, kv4, o4, do4 = (t.unsqueeze(0) for t in (q, kv, o, do))
    indices4, lse4 = indices.unsqueeze(0), lse.unsqueeze(0)

    B, S, H, dim = q4.shape
    _, S_kv, kv_group, _ = kv4.shape
    D = DV
    D_tail = dim - D
    topk = indices4.shape[-1]

    preprocess_kernel = mod.preprocess(B, S, H, D)
    bwd_kernel = mod.bwd(B, S, S_kv, H, D, D_tail, topk, kv_group, sm_scale, True)
    delta = preprocess_kernel(o4, do4)
    dkv = torch.zeros_like(kv4, dtype=torch.float32)

    def fn():
        return bwd_kernel(q4, kv4, do4, indices4, lse4, delta, dkv)

    return do_bench(fn, warmup=warmup, rep=rep, backend=backend)


def _check_parity(ref_mod, fusion_mod, inputs: tuple[torch.Tensor, ...], sm_scale: float) -> tuple[float, float]:
    """Return (dq_cos, dkv_cos) between the reference and fusion kernels' outputs."""
    q, kv, o, do, indices, lse = inputs
    dq_ref, dkv_ref = ref_mod.sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=sm_scale)
    dq_opt, dkv_opt = fusion_mod.sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=sm_scale)
    dq_cos = torch.nn.functional.cosine_similarity(dq_opt.float().flatten(), dq_ref.float().flatten(), dim=0).item()
    dkv_cos = torch.nn.functional.cosine_similarity(dkv_opt.float().flatten(), dkv_ref.float().flatten(), dim=0).item()
    return dq_cos, dkv_cos


def main() -> None:
    """Parse args, build inputs, run the parity check, and print the ms/TFLOP/s/TB/s comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults = the upstream DeepSeek-V3.2 reference shape (B=1, S=4096, H=64,
    # topk=2048), reusing the same 576/512 latent geometry as GLM-5.2 DSA.
    parser.add_argument("--seq-len", type=int, default=4096, help="query sequence length S (default: 4096)")
    parser.add_argument(
        "--seq-len-kv", type=int, default=None, help="KV cache length S_kv (default: = --seq-len, causal self-attn)"
    )
    parser.add_argument("--heads", type=int, default=64, help="number of query heads H (default: 64)")
    parser.add_argument("--topk", type=int, default=2048, help="selected KVs per token; multiple of 64 (default: 2048)")
    parser.add_argument("--kv-group", type=int, default=1, help="KV groups (default: 1)")
    parser.add_argument("--warmup", type=float, default=250.0, help="do_bench target warmup ms (default: 250)")
    parser.add_argument("--rep", type=float, default=100.0, help="do_bench target measure ms (default: 100)")
    parser.add_argument(
        "--backend",
        choices=["event", "cupti", "cudagraph"],
        default="event",
        help="do_bench timing backend (default: event; cupti excludes host overhead)",
    )
    parser.add_argument(
        "--kernel-only", action="store_true", help="time only the bwd kernel (the part that differs), Delta precomputed"
    )
    parser.add_argument("--no-check", action="store_true", help="skip the reference-vs-fusion cosine parity check")
    parser.add_argument("--seed", type=int, default=0, help="torch RNG seed (default: 0)")
    args = parser.parse_args()

    from nemo_automodel.components.models.glm_moe_dsa.kernels._tilelang import HAS_TILELANG

    if not HAS_TILELANG:
        raise SystemExit("tilelang is not installed in this environment (needed for the GLM-5.2 DSA kernels).")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required to build and run the TileLang sparse-MLA backward kernels.")

    seq_len_kv = args.seq_len_kv if args.seq_len_kv is not None else args.seq_len
    if args.topk % 64 != 0:
        raise SystemExit(f"--topk must be a multiple of 64 (fwd block_I=64 / bwd block_size=32), got {args.topk}.")
    if seq_len_kv < args.seq_len:
        raise SystemExit(f"--seq-len-kv ({seq_len_kv}) must be >= --seq-len ({args.seq_len}) for causal indices.")

    from nemo_automodel.components.models.glm_moe_dsa.kernels import tilelang_sparse_mla_bwd as ref_mod
    from nemo_automodel.components.models.glm_moe_dsa.kernels import tilelang_sparse_mla_bwd_gemm_fusion as fusion_mod

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    sm_scale = DQKV**-0.5  # the kernels' own default (D + D_tail) ** -0.5; passed explicitly.

    print(f"device        : {torch.cuda.get_device_name(device)}")
    print(
        f"shape         : B=1 S={args.seq_len} S_kv={seq_len_kv} H={args.heads} topk={args.topk} kv_group={args.kv_group}"
    )
    print(f"latent geom   : DQKV={DQKV} DV={DV}  dtype=bf16  sm_scale={sm_scale:.6f}")
    print(f"timing        : do_bench(warmup={args.warmup:g}ms, rep={args.rep:g}ms, backend={args.backend})")
    print(f"mode          : {'bwd kernel only' if args.kernel_only else 'full sparse_mla_bwd (pre+bwd+post)'}")
    print("building inputs (forward pass + causal top-k indices) ...", flush=True)

    inputs = _build_inputs(args.seq_len, seq_len_kv, args.heads, args.topk, args.kv_group, sm_scale, device)

    if not args.no_check:
        dq_cos, dkv_cos = _check_parity(ref_mod, fusion_mod, inputs, sm_scale)
        status = "OK" if (dq_cos > 0.999 and dkv_cos > 0.999) else "WARN (variants diverge!)"
        print(f"parity        : dq_cos={dq_cos:.6f} dkv_cos={dkv_cos:.6f}  [{status}]")

    bench = _bench_kernel_only if args.kernel_only else _bench_e2e
    variants = [
        ("tilelang_sparse_mla_bwd", ref_mod),
        ("tilelang_sparse_mla_bwd_gemm_fusion", fusion_mod),
    ]

    print()
    print(f"{'variant':40s}{'ms':>12s}{'TFLOP/s':>12s}{'IO TB/s':>12s}")
    print("-" * 76)
    results: dict[str, float] = {}
    for name, mod in variants:
        ms = bench(mod, inputs, sm_scale, args.warmup, args.rep, args.backend)
        tflops, io_tbs = _flops_bandwidth(ms, args.seq_len, args.heads, args.topk)
        results[name] = ms
        print(f"{name:40s}{ms:>12.3f}{tflops:>12.1f}{io_tbs:>12.2f}", flush=True)
    print("-" * 76)

    ref_ms = results["tilelang_sparse_mla_bwd"]
    fusion_ms = results["tilelang_sparse_mla_bwd_gemm_fusion"]
    speedup = ref_ms / fusion_ms
    verb = "faster" if speedup >= 1.0 else "slower"
    print(f"gemm_fusion vs reference : {speedup:.3f}x {verb}  ({ref_ms:.3f} ms -> {fusion_ms:.3f} ms)")


if __name__ == "__main__":
    main()

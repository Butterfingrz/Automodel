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
"""Numerical parity between the reference and gemm-fusion GLM-5.2 DSA sparse-MLA backward kernels."""

import pytest
import torch

from nemo_automodel.components.models.glm_moe_dsa import optimized_kernels as ok

KV_LORA = 512
ROPE = 64
HEAD_DIM = KV_LORA + ROPE
N_HEADS = 16
TOPK = 64
T = 256

_run_gpu = ok.is_dsa_kernel_available("sparse_attn") and torch.cuda.is_available()
requires_kernels = pytest.mark.skipif(not _run_gpu, reason="requires tilelang kernels (CUDA + tilelang installed)")


def _causal_topk_indices(num_tokens, topk, device):
    idx = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)
    for t in range(num_tokens):
        n = min(t + 1, topk)
        idx[t, :n] = torch.randperm(t + 1, device=device)[:n].to(torch.int32)
    return idx


@requires_kernels
def test_sparse_mla_bwd_gemm_fusion_matches_reference():
    from nemo_automodel.components.models.glm_moe_dsa.kernels import tilelang_sparse_mla_fwd as fwd_mod
    from nemo_automodel.components.models.glm_moe_dsa.kernels.tilelang_sparse_mla_bwd import (
        sparse_mla_bwd as sparse_mla_bwd_ref,
    )
    from nemo_automodel.components.models.glm_moe_dsa.kernels.tilelang_sparse_mla_bwd_gemm_fusion import (
        sparse_mla_bwd as sparse_mla_bwd_fusion,
    )

    torch.manual_seed(0)
    dev = "cuda"
    scale = HEAD_DIM**-0.5
    q = torch.randn(T, N_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16).contiguous()
    kv = torch.randn(T, 1, HEAD_DIM, device=dev, dtype=torch.bfloat16).contiguous()
    indices = _causal_topk_indices(T, TOPK, dev).view(T, 1, TOPK).contiguous()
    o, lse = fwd_mod.sparse_mla_fwd_interface(q, kv, indices, sm_scale=scale)
    o, lse = o.contiguous(), lse.contiguous()
    do = torch.randn_like(o)

    dq_ref, dkv_ref = sparse_mla_bwd_ref(q, kv, o, do, indices, lse, sm_scale=scale)
    dq_opt, dkv_opt = sparse_mla_bwd_fusion(q, kv, o, do, indices, lse, sm_scale=scale)

    assert dq_opt.shape == dq_ref.shape
    assert dkv_opt.shape == dkv_ref.shape
    dq_cos = torch.nn.functional.cosine_similarity(dq_opt.float().flatten(), dq_ref.float().flatten(), dim=0).item()
    dkv_cos = torch.nn.functional.cosine_similarity(dkv_opt.float().flatten(), dkv_ref.float().flatten(), dim=0).item()
    assert dq_cos > 0.999, f"dq parity too low: {dq_cos:.6f}"
    assert dkv_cos > 0.999, f"dkv parity too low: {dkv_cos:.6f}"

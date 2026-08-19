# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

# This file contains code adapted from flash-linear-attention v0.5.1:
# https://github.com/fla-org/flash-linear-attention/blob/v0.5.1/fla/ops/gated_delta_rule/fused_recurrent.py
# The original source is licensed under the MIT license.

"""Indexed, in-place Triton decode kernel for the Gated Delta Rule."""

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus


@triton.jit
def _indexed_recurrent_gated_delta_rule_kernel(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    state,
    state_indices,
    output,
    stride_q_batch,
    stride_q_head,
    stride_q_k,
    stride_k_batch,
    stride_k_head,
    stride_k_k,
    stride_v_batch,
    stride_v_head,
    stride_v_v,
    stride_g_batch,
    stride_g_head,
    stride_beta_batch,
    stride_beta_head,
    stride_state_batch,
    stride_state_head,
    stride_state_k,
    stride_state_v,
    stride_output_batch,
    stride_output_head,
    stride_output_v,
    scale,
    num_value_tiles,
    NUM_KEY_HEADS: tl.constexpr,
    NUM_VALUE_HEADS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_KEY_DIM: tl.constexpr,
    BLOCK_VALUE_DIM: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    """Update one cache row per valid request and skip negative slot indices."""
    program_id = tl.program_id(0)
    value_tile = program_id % num_value_tiles
    batch_head_id = program_id // num_value_tiles
    batch_id = batch_head_id // NUM_VALUE_HEADS
    value_head_id = batch_head_id % NUM_VALUE_HEADS
    key_head_id = value_head_id // (NUM_VALUE_HEADS // NUM_KEY_HEADS)

    key_offsets = tl.arange(0, BLOCK_KEY_DIM)
    value_offsets = value_tile * BLOCK_VALUE_DIM + tl.arange(0, BLOCK_VALUE_DIM)
    key_mask = key_offsets < KEY_DIM
    value_mask = value_offsets < VALUE_DIM

    output_ptrs = (
        output
        + batch_id * stride_output_batch
        + value_head_id * stride_output_head
        + value_offsets * stride_output_v
    )

    state_slot = tl.load(state_indices + batch_id)
    if state_slot < 0:
        tl.store(output_ptrs, 0.0, mask=value_mask)
        return

    query_ptrs = (
        q + batch_id * stride_q_batch + key_head_id * stride_q_head + key_offsets * stride_q_k
    )
    key_ptrs = (
        k + batch_id * stride_k_batch + key_head_id * stride_k_head + key_offsets * stride_k_k
    )
    value_ptrs = (
        v + batch_id * stride_v_batch + value_head_id * stride_v_head + value_offsets * stride_v_v
    )
    state_ptrs = (
        state
        + state_slot.to(tl.int64) * stride_state_batch
        + value_head_id * stride_state_head
        + key_offsets[:, None] * stride_state_k
        + value_offsets[None, :] * stride_state_v
    )

    state_mask = key_mask[:, None] & value_mask[None, :]
    hidden_state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    query = tl.load(query_ptrs, mask=key_mask, other=0.0).to(tl.float32)
    key = tl.load(key_ptrs, mask=key_mask, other=0.0).to(tl.float32)
    value = tl.load(value_ptrs, mask=value_mask, other=0.0).to(tl.float32)

    if USE_QK_L2NORM:
        query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
        key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
    query *= scale

    beta_value = tl.load(beta + batch_id * stride_beta_batch + value_head_id * stride_beta_head).to(
        tl.float32
    )
    beta_value = tl.sigmoid(beta_value)

    gate = tl.load(g + batch_id * stride_g_batch + value_head_id * stride_g_head).to(tl.float32)
    gate += tl.load(dt_bias + value_head_id).to(tl.float32)
    gate = -exp(tl.load(a_log + value_head_id).to(tl.float32)) * softplus(gate)
    hidden_state *= exp(gate)

    value = beta_value * (value - tl.sum(hidden_state * key[:, None], axis=0))
    hidden_state += key[:, None] * value[None, :]
    result = tl.sum(hidden_state * query[:, None], axis=0)

    tl.store(state_ptrs, hidden_state.to(state_ptrs.dtype.element_ty), mask=state_mask)
    tl.store(output_ptrs, result.to(output_ptrs.dtype.element_ty), mask=value_mask)


def indexed_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    use_qk_l2norm: bool,
    scale: float | None = None,
) -> torch.Tensor:
    """Run one GDN decode step using an indexed, in-place recurrent-state cache.

    Negative state indices denote CUDA-graph padding. Their output rows are
    zeroed without reading inputs or state, and the persistent cache is not
    modified.

    Args:
        q: Queries with shape ``[batch, 1, key_heads, key_dim]``.
        k: Keys with the same shape as ``q``.
        v: Values with shape ``[batch, 1, value_heads, value_dim]``.
        g: Raw decay logits with shape ``[batch, 1, value_heads]``.
        beta: Raw update logits with shape ``[batch, 1, value_heads]``.
        a_log: Per-value-head log decay parameters.
        dt_bias: Per-value-head decay biases.
        state: Cache with shape ``[slots, value_heads, key_dim, value_dim]``.
        state_indices: Cache slot per batch row; negative values mark padding.
            Nonnegative slots must be in bounds and unique within the batch.
        use_qk_l2norm: Apply FLA-compatible query/key L2 normalization.
        scale: Query scale. Defaults to ``key_dim**-0.5``.

    Returns:
        Output with the same shape and dtype as ``v``.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and state.is_cuda and state_indices.is_cuda
    assert q.ndim == 4 and k.shape == q.shape and q.shape[1] == 1
    assert v.ndim == 4 and v.shape[:2] == q.shape[:2]
    assert g.shape == beta.shape == v.shape[:3]

    batch, _, num_key_heads, key_dim = q.shape
    num_value_heads, value_dim = v.shape[2:]
    assert num_value_heads % num_key_heads == 0
    assert state.shape[1:] == (num_value_heads, key_dim, value_dim)
    assert state_indices.shape == (batch,)
    assert state_indices.dtype in (torch.int32, torch.int64)
    assert a_log.shape == dt_bias.shape == (num_value_heads,)

    if scale is None:
        scale = key_dim**-0.5

    output = torch.empty_like(v)
    block_key_dim = triton.next_power_of_2(key_dim)
    block_value_dim = min(8, triton.next_power_of_2(value_dim))
    num_value_tiles = triton.cdiv(value_dim, block_value_dim)
    grid = (batch * num_value_heads * num_value_tiles,)

    _indexed_recurrent_gated_delta_rule_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        state,
        state_indices,
        output,
        q.stride(0),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(2),
        v.stride(3),
        g.stride(0),
        g.stride(2),
        beta.stride(0),
        beta.stride(2),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        scale,
        num_value_tiles,
        NUM_KEY_HEADS=num_key_heads,
        NUM_VALUE_HEADS=num_value_heads,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        BLOCK_KEY_DIM=block_key_dim,
        BLOCK_VALUE_DIM=block_value_dim,
        USE_QK_L2NORM=use_qk_l2norm,
        num_warps=1,
        num_stages=3,
    )
    return output

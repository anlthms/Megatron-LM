# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

pytest.importorskip("fla")

from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

from megatron.core.ssm.ops.common.gated_delta_rule_triton import indexed_recurrent_gated_delta_rule


def _inputs(batch, key_heads, value_heads, key_dim, value_dim, dtype):
    return {
        "q": torch.randn(batch, 1, key_heads, key_dim, device="cuda", dtype=dtype),
        "k": torch.randn(batch, 1, key_heads, key_dim, device="cuda", dtype=dtype),
        "v": torch.randn(batch, 1, value_heads, value_dim, device="cuda", dtype=dtype),
        "g": torch.randn(batch, 1, value_heads, device="cuda", dtype=dtype),
        "beta": torch.randn(batch, 1, value_heads, device="cuda", dtype=dtype),
        "a_log": torch.randn(value_heads, device="cuda", dtype=torch.float32),
        "dt_bias": torch.randn(value_heads, device="cuda", dtype=torch.float32),
    }


@pytest.mark.internal
class TestIndexedRecurrentGatedDeltaRule:
    @pytest.mark.parametrize(
        ("dtype", "state_dtype"),
        [
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float32),
        ],
    )
    @pytest.mark.parametrize("use_qk_l2norm", [False, True])
    def test_matches_fla(self, dtype, state_dtype, use_qk_l2norm):
        torch.manual_seed(123)
        batch, slots = 3, 5
        key_heads, value_heads, key_dim, value_dim = 4, 8, 32, 64
        inputs = _inputs(batch, key_heads, value_heads, key_dim, value_dim, dtype)
        state = torch.randn(
            slots, value_heads, key_dim, value_dim, device="cuda", dtype=state_dtype
        )
        indices = torch.tensor([3, 1, 4], device="cuda", dtype=torch.int32)

        expected_state = state.clone()
        expected, final_state = fused_recurrent_gated_delta_rule(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            g=inputs["g"],
            beta=inputs["beta"],
            A_log=inputs["a_log"],
            dt_bias=inputs["dt_bias"],
            initial_state=expected_state[indices.long()].contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )
        expected_state[indices.long()] = final_state.to(state_dtype)

        actual_state = state.clone()
        actual = indexed_recurrent_gated_delta_rule(
            **inputs, state=actual_state, state_indices=indices, use_qk_l2norm=use_qk_l2norm
        )

        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual_state, expected_state, atol=2e-2, rtol=2e-2)

    def test_padding_skips_nan_state_and_inputs(self):
        torch.manual_seed(123)
        key_heads, value_heads, key_dim, value_dim = 4, 8, 32, 64
        inputs = _inputs(2, key_heads, value_heads, key_dim, value_dim, torch.bfloat16)
        state = torch.randn(4, value_heads, key_dim, value_dim, device="cuda", dtype=torch.bfloat16)
        state[0] = torch.nan
        for name in ("q", "k", "v", "g", "beta"):
            inputs[name][1] = torch.nan
        state_before = state.clone()
        indices = torch.tensor([2, -1], device="cuda", dtype=torch.int32)

        actual = indexed_recurrent_gated_delta_rule(
            **inputs, state=state, state_indices=indices, use_qk_l2norm=True
        )

        assert torch.isfinite(actual[0]).all()
        torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]), atol=0, rtol=0)
        assert torch.isnan(state[0]).all()
        torch.testing.assert_close(state[1], state_before[1], atol=0, rtol=0)
        torch.testing.assert_close(state[3], state_before[3], atol=0, rtol=0)

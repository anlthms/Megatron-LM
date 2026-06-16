# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for Gated Delta Net (GDN) dynamic-inference support.

These exercise the inference hooks added by the GDN dynamic-inference work:

  * ``ssm_state_shapes_per_request`` and the ``SSMInferenceStateConfig.from_model``
    wiring (no kernels -- always runnable on a GPU box).
  * ``_ssm_decode`` (single-token recurrent update) and ``_ssm_prefill`` (chunked
    varlen prefill) in isolation, including per-request state isolation.
  * The key correctness invariant: a chunked prefill followed by per-token decodes
    must reproduce a single full forward over the same sequence (state continuity).

The hook tests follow the pattern in ``tests/unit_tests/ssm/ops/test_ssm_kernel.py``:
build a real module, populate a real ``SSMMetadata``, and pass a lightweight
``SimpleNamespace`` context so we do not need a full ``DynamicInferenceContext``.
"""

import os
import types

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.inference.config import SSMInferenceStateConfig
from megatron.core.inference.contexts.attention_context.ssm_metadata import SSMMetadata
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import fla

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

# NVLS doesn't support one single GPU shared by multiple ranks; disable in tests.
os.environ.update({"NCCL_NVLS_ENABLE": "0"})


def _make_transformer_config(tp_size: int) -> TransformerConfig:
    """Small GDN config (Qwen-Next shaped, scaled down) for fast inference tests."""
    return TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,  # GVA: value heads != key heads (repeat_interleave path)
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        activation_func=F.silu,
        bf16=True,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=False,
        context_parallel_size=1,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )


def _build_gdn(config: TransformerConfig, dtype: torch.dtype = torch.bfloat16) -> GatedDeltaNet:
    pg_collection = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    submodules = get_experimental_attention_variant_module_spec(config=config).submodules
    gdn = GatedDeltaNet(
        config,
        submodules=submodules,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg_collection,
    )
    return gdn.cuda().to(dtype)


def _make_prefill_context(gdn, batch_indices, cu_seqlens, token_count):
    """Build a minimal SSMMetadata + context for ``_ssm_prefill``."""
    device = torch.cuda.current_device()
    metadata = SSMMetadata(
        max_requests=int(batch_indices.numel()),
        max_tokens=token_count,
        ssm_chunk_size=gdn.chunk_size,
        d_conv=gdn.conv_kernel_dim,
    )
    metadata.cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
    metadata.batch_indices_prefill = batch_indices.to(device=device, dtype=torch.int32)
    metadata.seq_idx = torch.zeros((1, token_count), dtype=torch.int32, device=device)
    metadata.real_prefill_token_count = int(cu_seqlens[-1].item())
    return types.SimpleNamespace(ssm_metadata=metadata, ssm_slot_allocator=None)


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNetInference:
    """GDN dynamic-inference hook tests at TP=1, CP=1 (single GPU)."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        self.config = _make_transformer_config(tp_size=1)
        self.gdn = _build_gdn(self.config)
        yield
        Utils.destroy_model_parallel()

    # ------------------------------------------------------------------
    # State-shape config wiring (no kernels).
    # ------------------------------------------------------------------
    def test_ssm_state_shapes_per_request(self):
        gdn = self.gdn
        conv_shape, recurrent_shape = gdn.ssm_state_shapes_per_request()
        assert conv_shape == (gdn.conv_dim_local_tp, gdn.conv_kernel_dim)
        assert recurrent_shape == (
            gdn.num_v_heads_local_tp,
            gdn.key_head_dim,
            gdn.value_head_dim,
        )

    def test_from_model_recognizes_gdn(self):
        """SSMInferenceStateConfig.from_model must detect a GDN hybrid model and read
        its per-request shapes + chunk size (mixer lives in the self_attention slot)."""
        gdn = self.gdn

        # Mirror what HybridStack exposes: layer_type_list + layers whose GDN entry is a
        # TransformerLayer with the mixer in `self_attention`.
        layer = types.SimpleNamespace(self_attention=gdn)
        decoder = types.SimpleNamespace(
            layer_type_list=[Symbols.GDN],
            layers=[layer],
            ssm_state_shapes_per_request=gdn.ssm_state_shapes_per_request,
        )
        model = types.SimpleNamespace(decoder=decoder, config=self.config)

        cfg = SSMInferenceStateConfig.from_model(model)
        assert cfg is not None, "from_model should return a config for a GDN hybrid model"
        assert cfg.conv_states_shape == (gdn.conv_dim_local_tp, gdn.conv_kernel_dim)
        assert cfg.recurrent_states_shape == (
            gdn.num_v_heads_local_tp,
            gdn.key_head_dim,
            gdn.value_head_dim,
        )
        assert cfg.ssm_chunk_size == gdn.chunk_size == 64

    # ------------------------------------------------------------------
    # Decode hook.
    # ------------------------------------------------------------------
    def test_ssm_decode_shapes_and_state_isolation(self):
        gdn = self.gdn
        device = torch.cuda.current_device()
        num_slots = 4
        active = 2  # first two slots are real; the rest are untouched padding

        proj = torch.randn(
            active, 1, gdn.in_proj_dim, device=device, dtype=torch.bfloat16
        )
        conv_state = torch.zeros(
            num_slots, gdn.conv_dim_local_tp, gdn.conv_kernel_dim, device=device, dtype=torch.float32
        )
        recurrent_state = torch.zeros(
            num_slots,
            gdn.num_v_heads_local_tp,
            gdn.key_head_dim,
            gdn.value_head_dim,
            device=device,
            dtype=torch.float32,
        )
        batch_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)

        y = gdn._ssm_decode(proj, conv_state, recurrent_state, batch_indices)

        assert y.shape == (active, 1, gdn.v_dim_local_tp)
        # Active slots advanced; padding slots untouched.
        assert conv_state[:active].abs().max() > 0
        assert recurrent_state[:active].abs().max() > 0
        assert torch.count_nonzero(conv_state[active:]) == 0
        assert torch.count_nonzero(recurrent_state[active:]) == 0

    def test_ssm_decode_rejects_speculative(self):
        gdn = self.gdn
        device = torch.cuda.current_device()
        proj = torch.randn(1, 2, gdn.in_proj_dim, device=device, dtype=torch.bfloat16)
        conv_state = torch.zeros(
            1, gdn.conv_dim_local_tp, gdn.conv_kernel_dim, device=device, dtype=torch.float32
        )
        recurrent_state = torch.zeros(
            1,
            gdn.num_v_heads_local_tp,
            gdn.key_head_dim,
            gdn.value_head_dim,
            device=device,
            dtype=torch.float32,
        )
        batch_indices = torch.tensor([0], dtype=torch.int32, device=device)
        with pytest.raises(AssertionError, match="speculative"):
            gdn._ssm_decode(proj, conv_state, recurrent_state, batch_indices)

    # ------------------------------------------------------------------
    # Prefill hook.
    # ------------------------------------------------------------------
    def test_ssm_prefill_padding_isolation(self):
        """_ssm_prefill must update only the active request's slot."""
        gdn = self.gdn
        device = torch.cuda.current_device()
        num_slots = 8
        seq_len = 6

        proj = torch.randn(seq_len, 1, gdn.in_proj_dim, device=device, dtype=torch.bfloat16)
        conv_state = torch.zeros(
            num_slots, gdn.conv_dim_local_tp, gdn.conv_kernel_dim, device=device, dtype=torch.float32
        )
        recurrent_state = torch.zeros(
            num_slots,
            gdn.num_v_heads_local_tp,
            gdn.key_head_dim,
            gdn.value_head_dim,
            device=device,
            dtype=torch.float32,
        )
        context = _make_prefill_context(
            gdn,
            batch_indices=torch.tensor([0]),
            cu_seqlens=torch.tensor([0, seq_len]),
            token_count=seq_len,
        )

        y = gdn._ssm_prefill(proj, conv_state, recurrent_state, context)

        assert y.shape == (seq_len, 1, gdn.v_dim_local_tp)
        assert conv_state[0].abs().max() > 0, "active conv_state should be written"
        assert recurrent_state[0].abs().max() > 0, "active recurrent_state should be written"
        assert torch.count_nonzero(conv_state[1:]) == 0, "padding conv_state must stay zero"
        assert torch.count_nonzero(recurrent_state[1:]) == 0, "padding recurrent_state must stay zero"

    # ------------------------------------------------------------------
    # The key invariant: prefill + per-token decode == single full forward.
    # ------------------------------------------------------------------
    def test_prefill_then_decode_matches_full_forward(self):
        """Chunked prefill of a prompt, followed by single-token decodes, should
        reproduce a full forward over the whole sequence (state continuity)."""
        gdn = self.gdn
        device = torch.cuda.current_device()
        # bf16 across chunk- vs fused-recurrent kernels: generous tolerances.
        atol, rtol = 2e-2, 2e-2

        total_len = 8
        prefill_len = 5
        hidden = torch.randn(total_len, 1, gdn.hidden_size, device=device, dtype=torch.bfloat16)

        # Reference: full training-path forward (initial state = 0), then out_proj.
        with torch.no_grad():
            y_ref, _ = gdn(hidden, attention_mask=None)  # [total_len, 1, hidden]

            # Shared projection for the inference hooks.
            proj, _ = gdn.in_proj(hidden)  # [total_len, 1, in_proj_dim]

            conv_state = torch.zeros(
                1,
                gdn.conv_dim_local_tp,
                gdn.conv_kernel_dim,
                device=device,
                dtype=torch.float32,
            )
            recurrent_state = torch.zeros(
                1,
                gdn.num_v_heads_local_tp,
                gdn.key_head_dim,
                gdn.value_head_dim,
                device=device,
                dtype=torch.float32,
            )

            # --- Prefill the first `prefill_len` tokens ---
            context = _make_prefill_context(
                gdn,
                batch_indices=torch.tensor([0]),
                cu_seqlens=torch.tensor([0, prefill_len]),
                token_count=prefill_len,
            )
            y_prefill = gdn._ssm_prefill(
                proj[:prefill_len], conv_state, recurrent_state, context
            )  # [prefill_len, 1, d_inner]
            out_prefill, _ = gdn.out_proj(y_prefill)  # [prefill_len, 1, hidden]

            torch.testing.assert_close(
                out_prefill, y_ref[:prefill_len], atol=atol, rtol=rtol
            )

            # --- Decode the remaining tokens one at a time ---
            batch_indices = torch.tensor([0], dtype=torch.int32, device=device)
            for pos in range(prefill_len, total_len):
                proj_step = proj[pos : pos + 1].squeeze(1).view(1, 1, -1)  # [N=1, S=1, d]
                y_dec = gdn._ssm_decode(proj_step, conv_state, recurrent_state, batch_indices)
                out_dec, _ = gdn.out_proj(y_dec.view(1, 1, -1))  # [1, 1, hidden]
                torch.testing.assert_close(
                    out_dec[0, 0], y_ref[pos, 0], atol=atol, rtol=rtol
                )


def _parity_config(n_kheads, n_vheads, head_k, head_v, hidden=128) -> TransformerConfig:
    """fp32 GDN config with configurable head dims for the FLA parity check."""
    return TransformerConfig(
        hidden_size=hidden,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=head_k,
        linear_value_head_dim=head_v,
        linear_num_key_heads=n_kheads,
        linear_num_value_heads=n_vheads,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        activation_func=F.silu,
        bf16=False,  # fp32 for a clean numerical comparison
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )


def _copy_fla_into_mcore(fla_layer, gdn):
    """Load FLA's separate projections/convs/params into mcore's fused layout.

    mcore fuses q/k/v/gate/beta/alpha into one ``in_proj`` and the three FLA
    convs into one depthwise ``conv1d``. mcore's ``in_proj`` also fuses an input
    RMSNorm (FLA's lives in the decoder block); it is neutralized by zeroing the
    norm weight (zero-centered gamma -> scale 1) and feeding unit-RMS input.
    """
    with torch.no_grad():
        # in_proj fused layout: [q | k | v | z(gate) | beta | alpha]
        gdn.in_proj.weight.copy_(
            torch.cat(
                [
                    fla_layer.q_proj.weight,
                    fla_layer.k_proj.weight,
                    fla_layer.v_proj.weight,
                    fla_layer.g_proj.weight,  # -> z / gate
                    fla_layer.b_proj.weight,  # -> beta
                    fla_layer.a_proj.weight,  # -> alpha
                ],
                dim=0,
            ).to(gdn.in_proj.weight.dtype)
        )
        gdn.in_proj.layer_norm_weight.zero_()  # neutralize fused input RMSNorm
        # single fused depthwise conv = channel-concat of FLA's q/k/v convs
        gdn.conv1d.weight.copy_(
            torch.cat(
                [fla_layer.q_conv1d.weight, fla_layer.k_conv1d.weight, fla_layer.v_conv1d.weight],
                dim=0,
            ).to(gdn.conv1d.weight.dtype)
        )
        gdn.A_log.copy_(fla_layer.A_log.to(gdn.A_log.dtype))
        gdn.dt_bias.copy_(fla_layer.dt_bias.to(gdn.dt_bias.dtype))
        # mcore uses zero-centered gamma (effective scale = 1 + weight); FLA's
        # o_norm scales by weight directly. Subtract 1 so effective scales match.
        gdn.out_norm.weight.copy_((fla_layer.o_norm.weight - 1.0).to(gdn.out_norm.weight.dtype))
        gdn.out_proj.weight.copy_(fla_layer.o_proj.weight.to(gdn.out_proj.weight.dtype))


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNetFLAParity:
    """mcore GDN must match FLA's reference ``GatedDeltaNet`` on identical weights.

    The hook tests above prove mcore is *self-consistent* (inference == its own
    forward). This is the *external* fidelity check: mcore's hand-written glue
    (fused in_proj split, single fused conv vs FLA's three, external g/beta and
    q/k L2-norm vs FLA's in-kernel versions, gated RMSNorm) must reproduce the
    canonical FLA layer -- including the grouped-value-attention repeat_interleave
    path and head_v != head_k. Both share FLA's kernels, so a mismatch isolates an
    mcore glue bug. Combined with the prefill+decode == forward invariant above,
    this validates mcore's dynamic inference against an independent reference.
    """

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "n_kheads,n_vheads,head_k,head_v",
        [
            (4, 4, 32, 32),  # no GVA, head_v == head_k
            (4, 8, 32, 32),  # GVA (repeat_interleave path)
            (4, 8, 32, 64),  # GVA + expand_v=2
            (2, 8, 64, 128),  # GVA x4 + expand_v=2
        ],
    )
    def test_matches_fla_reference(self, n_kheads, n_vheads, head_k, head_v):
        import fla.layers.gated_deltanet as fla_gdn

        torch.manual_seed(0)
        config = _parity_config(n_kheads, n_vheads, head_k, head_v)
        gdn = _build_gdn(config, dtype=torch.float32)

        fla_layer = (
            fla_gdn.GatedDeltaNet(
                hidden_size=config.hidden_size,
                expand_v=head_v / head_k,
                head_dim=head_k,
                num_heads=n_kheads,
                num_v_heads=n_vheads,
                mode="chunk",
                use_gate=True,
                use_short_conv=True,
                conv_size=config.linear_conv_kernel_dim,
                conv_bias=False,
                layer_idx=0,
            )
            .cuda()
            .float()
        )
        fla_layer.eval()
        _copy_fla_into_mcore(fla_layer, gdn)

        # Unit-RMS input per token so mcore's neutralized input RMSNorm is identity.
        # seq_len > 64 so FLA's layer uses the chunk kernel (matches prefill).
        hidden = torch.randn(1, 96, config.hidden_size, device="cuda", dtype=torch.float32)
        hidden = hidden / hidden.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()

        with torch.inference_mode():
            o_fla, _, _ = fla_layer(hidden.clone())  # [B, T, H]
            o_mc, _ = gdn(hidden.transpose(0, 1).clone(), attention_mask=None)  # [T, B, H]
        o_mc = o_mc.transpose(0, 1)

        torch.testing.assert_close(o_mc, o_fla, atol=1e-3, rtol=1e-3)

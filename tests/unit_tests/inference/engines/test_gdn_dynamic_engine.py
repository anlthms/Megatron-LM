# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tier-1 end-to-end engine correctness tests for GDN dynamic inference.

These drive a Gated Delta Net (GDN) hybrid model (layer pattern ``"G*-"``)
through the real ``DynamicInferenceEngine`` and check engine-integration
correctness with *relative* assertions (no golden token lists needed):

  * determinism            — same config twice -> identical tokens
  * cuda-graph parity      — graphs on vs off -> identical tokens
  * scheduling invariance  — different request interleaving -> identical tokens

This complements the hook-level continuity unit test
(``tests/unit_tests/ssm/test_gated_delta_net_inference.py``), which validates the
absolute math (prefill+decode == full forward). Together they cover both the
kernels and the engine plumbing (state-cache allocation via
``MambaInferenceStateConfig.from_model``, slot management, mixed batches, CUDA
graph capture/replay).

It reuses the machinery in ``test_dynamic_engine.py`` and only overrides the
model build for a GDN hybrid.
"""

import random
import types

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.inference.config import MambaInferenceStateConfig
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
    DynamicEngineTestEnv,
    DynamicInferenceEngineTestBase,
    set_rounder,
)
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

try:
    import fla  # noqa: F401

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


def _gdn_transformer_config(test_config, effective_cuda_graph_impl):
    """Small GDN hybrid config (Qwen3-Next shaped, scaled down)."""
    return TransformerConfig(
        params_dtype=torch.bfloat16,
        num_layers=3,  # pattern "G*-": GDN, attention, MLP
        hidden_size=256,
        num_attention_heads=16,
        # GDN (linear-attention) knobs:
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,  # GVA: value heads != key heads
        mamba_num_heads=16,  # unused (no 'M' in pattern) but keep config valid
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        activation_func=F.silu,
        use_cpu_initialization=True,
        cuda_graph_impl=effective_cuda_graph_impl,
        inference_rng_tracker=True,
        tensor_model_parallel_size=test_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        add_bias_linear=False,
        inference_sampling_seed=test_config.random_seed,
        inference_cuda_graph_scope=(
            test_config.inference_cuda_graph_scope
            if test_config.num_cuda_graphs is not None and test_config.force_build_cuda_graphs
            else InferenceCudaGraphScope.none
        ),
        transformer_impl="transformer_engine",
        is_hybrid_model=True,
    )


class _GDNDynamicEngineMixin:
    """Shared GDN-hybrid model build and helpers for the dynamic-engine tests.

    Not a test class (no ``Test`` prefix), so pytest does not collect it. The
    TP=1 and TP=2 test classes both mix it in and define only their own tests,
    so the (slow) TP=1 tests are not re-collected under the TP=2 class.
    """

    # When True, requests sample greedily (top_k=1 -> argmax). Under the default
    # random multinomial sampling the shared RNG is consumed in batch order, so
    # different request interleavings draw different numbers and produce
    # different tokens for *any* model -- an RNG-ordering artifact, not a
    # correctness property. Greedy sampling removes the RNG so a comparison is on
    # logits/state (used by the scheduling-invariance and TP=2 tests).
    _greedy_sampling = False

    @classmethod
    def _build_requests(cls, test_config):
        requests = super()._build_requests(test_config)
        if cls._greedy_sampling:
            for r in requests:
                r.sampling_params.top_k = 1
        return requests

    @classmethod
    @torch.inference_mode()
    def _build_test_env(cls, test_config):
        """Build a GDN hybrid model and wire it into the dynamic engine.

        Mirrors ``DynamicInferenceEngineTestBase._build_test_env`` but builds a
        ``"G*-"`` GDN hybrid instead of the Mamba/GPT models.
        """
        clear_nvte_env_vars()
        set_rounder(4)

        random.seed(test_config.random_seed)
        torch.manual_seed(test_config.random_seed)
        model_parallel_cuda_manual_seed(
            seed=test_config.random_seed,
            inference_rng_tracker=True,
            use_cudagraphable_rng=False,
            force_reset_rng=True,
        )

        requests = cls._build_requests(test_config)

        effective_cuda_graph_impl = test_config.cuda_graph_impl
        if effective_cuda_graph_impl is None:
            effective_cuda_graph_impl = (
                "local"
                if test_config.num_cuda_graphs is not None and test_config.force_build_cuda_graphs
                else "none"
            )

        transformer_config = _gdn_transformer_config(test_config, effective_cuda_graph_impl)

        model = HybridModel(
            config=transformer_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=test_config.vocab_size,
            max_sequence_length=test_config.max_sequence_length,
            parallel_output=True,
            hybrid_layer_pattern="G*-",
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()

        for param in model.parameters():
            param.data = param.data.to(transformer_config.params_dtype)
        model.eval()

        mamba_inference_state_config = MambaInferenceStateConfig.from_model(model)
        assert mamba_inference_state_config is not None, (
            "GDN hybrid should yield a Mamba inference state config"
        )

        inference_context = cls._build_inference_context(
            test_config=test_config,
            transformer_config=transformer_config,
            requests=requests,
            mamba_inference_state_config=mamba_inference_state_config,
        )

        inference_wrapped_model = GPTInferenceWrapper(model, inference_context)
        inference_wrapped_model.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )

        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=types.SimpleNamespace(
                vocab_size=test_config.vocab_size, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )

        delete_cuda_graphs()
        engine = DynamicInferenceEngine(text_generation_controller, inference_context)
        return DynamicEngineTestEnv(config=test_config, requests=requests, engine=engine)

    @staticmethod
    def _tokens(env):
        return [list(r.generated_tokens) for r in env.requests]


class TestGDNDynamicInferenceEngine(_GDNDynamicEngineMixin, DynamicInferenceEngineTestBase):
    """End-to-end dynamic-engine tests for a GDN hybrid model (TP=1)."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        set_rounder(64)
        Utils.destroy_model_parallel()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_smoke(self):
        """GDN hybrid runs end-to-end through the dynamic engine and generates tokens."""
        env = self._run_test(
            model_provider="gdn",
            num_tokens_to_generate=8,
            num_cuda_graphs=None,
            context_max_requests=128,
        )
        toks = self._tokens(env)
        assert all(len(t) > 0 for t in toks), f"some requests generated no tokens: {toks}"

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_determinism(self):
        """Two identical runs produce identical tokens."""
        a = self._tokens(self._run_test(model_provider="gdn", num_tokens_to_generate=8))
        b = self._tokens(self._run_test(model_provider="gdn", num_tokens_to_generate=8))
        assert a == b, "GDN dynamic inference is non-deterministic across identical runs"

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_scheduling_invariance(self):
        """Per-request output must not depend on how requests interleave (gap steps).

        Uses greedy sampling so the comparison is not confounded by RNG-draw
        ordering: random multinomial sampling consumes a single shared RNG in
        batch order, so different interleavings draw different numbers and yield
        different tokens for any model. Greedy (argmax) output depends only on
        the logits -- hence on per-request state -- so a mismatch here is a true
        state-isolation / batching bug.
        """
        type(self)._greedy_sampling = True
        try:
            dense = self._tokens(
                self._run_test(model_provider="gdn", num_tokens_to_generate=8, num_gap_steps=0)
            )
            staggered = self._tokens(
                self._run_test(model_provider="gdn", num_tokens_to_generate=8, num_gap_steps=3)
            )
        finally:
            type(self)._greedy_sampling = False
        assert dense == staggered, (
            "GDN per-request output changed with request interleaving -> "
            "state isolation / batching bug"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("inference_cuda_graph_scope", [InferenceCudaGraphScope.block])
    def test_cuda_graph_parity(self, inference_cuda_graph_scope):
        """Tokens generated with CUDA graphs must match those without."""
        eager = self._tokens(
            self._run_test(model_provider="gdn", num_tokens_to_generate=16, num_cuda_graphs=None)
        )
        graphed = self._tokens(
            self._run_test(
                model_provider="gdn",
                num_tokens_to_generate=16,
                num_cuda_graphs=4,
                inference_cuda_graph_scope=inference_cuda_graph_scope,
                force_build_cuda_graphs=True,
                context_max_requests=128,
                use_cuda_graphs_for_non_decode_steps=False,  # decode-only graphs; avoids OOM
            )
        )
        assert eager == graphed, "GDN CUDA-graph decode diverges from eager decode"


class TestGDNDynamicInferenceEngineTP2(_GDNDynamicEngineMixin, DynamicInferenceEngineTestBase):
    """Tensor-parallel (TP=2) coverage for GDN in the dynamic engine.

    Exercises the per-rank SSM state path under TP>1: the state cache allocated
    with *local* (sharded) head dims, the GDN kernels over sharded heads, the
    out_proj all-reduce, and slot management. Asserts the TP=2 run completes and
    is internally deterministic (same config twice -> identical tokens), which
    is reproducible without golden values.

    Launched with ``--nproc-per-node 2``; manages model-parallel state per run.
    """

    # Greedy decode so the determinism check is on logits, not RNG-draw order;
    # see TestGDNDynamicInferenceEngine._greedy_sampling.
    _greedy_sampling = True

    @classmethod
    def setup_class(cls):
        # Bring up the process group (but not a fixed TP layout) so the world
        # size is known for the skip guard; the model-parallel layout itself is
        # (re)initialized per run in _build_test_env.
        Utils.initialize_distributed()

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        set_rounder(64)
        if parallel_state.model_parallel_is_initialized():
            Utils.destroy_model_parallel()

    @classmethod
    @torch.inference_mode()
    def _build_test_env(cls, test_config):
        # Utils.initialize_model_parallel destroys any prior state first.
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=test_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=1,
        )
        delete_cuda_graphs()
        return super()._build_test_env(test_config)

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_tp2_runs_and_is_deterministic(self):
        """GDN runs end-to-end at TP=2 and generates deterministically."""
        if not torch.distributed.is_initialized():
            pytest.skip("Distributed not initialized")
        if torch.distributed.get_world_size() < 2:
            pytest.skip("TP=2 test requires at least 2 GPUs")

        a = self._tokens(
            self._run_test(
                model_provider="gdn", num_tokens_to_generate=8, tensor_model_parallel_size=2
            )
        )
        b = self._tokens(
            self._run_test(
                model_provider="gdn", num_tokens_to_generate=8, tensor_model_parallel_size=2
            )
        )
        assert all(len(t) > 0 for t in a), f"some requests generated no tokens at TP=2: {a}"
        assert a == b, "GDN TP=2 dynamic inference is non-deterministic across identical runs"

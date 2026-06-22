#!/bin/bash
# Tutorial: Train a small hybrid (Mamba + attention) model with Megatron-LM.
#
# This is the interactive version — run it after getting a shell on a compute
# node.  Obtain a node first (e.g. with start-interactive-node.sh), then:
#
#   bash examples/training_scripts/tutorial_hybrid_interactive.sh
#
# For the sbatch version (submit from a login node), see
# tutorial_hybrid_batch.sh.
#
# Model: 16-layer hybrid, ~300 M params.
# Layer pattern: MMM*MMM*MMM*MMM*  (12 Mamba + 4 attention layers)
#   M = Mamba (state-space) layer
#   * = multi-head attention layer
#
# Data modes (set USE_MOCK_DATA before running):
#   USE_MOCK_DATA=true  (default) — synthetic data, no setup.
#   USE_MOCK_DATA=false — Common Pile CI dataset: 12 M sequences / ~12.7 B
#       tokens, GPT-2 BPE tokenized, already preprocessed.  Same dataset
#       used by all Megatron-LM functional tests.  No extra setup needed.
#
# Runtime: ≈2 min on 1 H100.

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=0   # disable cuDNN fused attention

# ---------------------------------------------------------------------------
# Paths
#
# REPO_DIR: path to the megatron-lm checkout inside the container.
#   If you entered via start-interactive-node.sh the repo is mounted at
#   /opt/megatron-lm; adjust if your mount differs.
#
# OUT_DIR: where preprocessed data, checkpoints, and tensorboard events are
#   written.  Must be on a shared filesystem (not /tmp) to survive the session
#   and to allow data reuse across runs.
#   Override: OUT_DIR=/lustre/.../<yourname>/tutorial_hybrid bash ...
# ---------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-/opt/megatron-lm}"
OUT_DIR="${OUT_DIR:-/tmp/tutorial_hybrid_$(whoami)}"

CHECKPOINT_DIR="${OUT_DIR}/checkpoints"
DATACACHE_DIR="${OUT_DIR}/data_cache"
TENSORBOARD_DIR="${OUT_DIR}/tensorboard"

mkdir -p "${CHECKPOINT_DIR}" "${DATACACHE_DIR}" "${TENSORBOARD_DIR}"

# ---------------------------------------------------------------------------
# GPU count — defaults to all GPUs visible to this shell.
# Override: GPUS_PER_NODE=2 bash tutorial_hybrid_interactive.sh
# ---------------------------------------------------------------------------
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
USE_MOCK_DATA="${USE_MOCK_DATA:-true}"

# The Megatron-LM CI dataset: Common Pile v0.1, tokenized with GPT-2 BPE.
# 12 M sequences / ~12.7 B tokens — the same dataset all functional tests use.
# Already preprocessed; no extra setup needed.
# Canonical cluster path (mapped to /mnt/artifacts inside CI containers).
CI_DATA_ROOT="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci"
DATA_PATH="${CI_DATA_ROOT}/text/common_pile/v01_filtered_data/my-gpt3_00_text_document"
GPT2_VOCAB="${CI_DATA_ROOT}/text/common_pile/v01_filtered_data/bpe/vocab.json"
GPT2_MERGES="${CI_DATA_ROOT}/text/common_pile/v01_filtered_data/bpe/merges.txt"

# To learn how this dataset was built from raw text, see:
#   tools/common_pile_dataset/README.md  (download + preprocess_data.py pipeline)

if [ "${USE_MOCK_DATA}" = "true" ]; then
    data_options="--mock-data"
else
    data_options=" \
        --data-path ${DATA_PATH} \
        --split 949,50,1 \
        --data-cache-path ${DATACACHE_DIR} \
        --tokenizer-type GPT2BPETokenizer \
        --vocab-file ${GPT2_VOCAB} \
        --merge-file ${GPT2_MERGES} \
        --no-mmap-bin-files \
        --num-workers 2 \
        --no-create-attention-mask-in-dataloader "
fi

# ---------------------------------------------------------------------------
# W&B logging — uncomment and fill in to enable.
# Requires `wandb login` (or WANDB_API_KEY set) before launching.
# ---------------------------------------------------------------------------
# WANDB_PROJECT="megatron-tutorial"
# WANDB_EXP_NAME="tutorial_hybrid_$(date +%Y%m%d_%H%M%S)"
# wandb_options=" \
#     --wandb-project ${WANDB_PROJECT} \
#     --wandb-exp-name ${WANDB_EXP_NAME} \
#     --wandb-save-dir ${OUT_DIR}/wandb "
wandb_options=""

# ---------------------------------------------------------------------------
# Model and training options
# ---------------------------------------------------------------------------
# Hybrid layer pattern: repeat [3 Mamba, 1 attention] four times = 16 layers.
# Swap in G (GDN), D (DSA), or E (MoE) to experiment with other layer types.
HYBRID_PATTERN="MMM*MMM*MMM*MMM*"

# global-batch-size scales with GPU count (1 sample per GPU, data-parallel).
GLOBAL_BATCH_SIZE="${GPUS_PER_NODE}"

options=" \
    --use-mcore-models \
    --hybrid-layer-pattern ${HYBRID_PATTERN} \
    --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec \
    --hidden-size 1024 \
    --num-attention-heads 8 \
    --group-query-attention \
    --num-query-groups 4 \
    --ffn-hidden-size 4096 \
    --kv-channels 128 \
    --squared-relu \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.02 \
    --position-embedding-type none \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --normalization RMSNorm \
    \
    --bf16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --train-samples 100 \
    --lr-decay-style WSD \
    --lr-decay-samples 100 \
    --lr-warmup-samples 5 \
    --lr-wsd-decay-style minus_sqrt \
    --lr-wsd-decay-samples 20 \
    --micro-batch-size 1 \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 8e-4 \
    --min-lr 8e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --eval-interval 50 \
    --eval-iters 5 \
    \
    ${data_options} \
    \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --ddp-num-buckets 1 \
    --attention-backend flash \
    \
    --ckpt-format torch_dist \
    --load ${CHECKPOINT_DIR} \
    --save ${CHECKPOINT_DIR} \
    --save-interval 100 \
    \
    --log-interval 5 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-progress \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    ${wandb_options} \
    \
    --distributed-timeout-minutes 10 \
    --disable-gloo-process-groups "

# ---------------------------------------------------------------------------
# Phase 2: Train
# ---------------------------------------------------------------------------
echo "Launching ${GPUS_PER_NODE}-GPU training in ${REPO_DIR}"
echo "Output dir: ${OUT_DIR}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${GPUS_PER_NODE}" \
    "${REPO_DIR}/pretrain_hybrid.py" \
    ${options}

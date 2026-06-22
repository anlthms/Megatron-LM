#!/bin/bash
# Tutorial: Train a small hybrid (Mamba + attention) model with Megatron-LM.
#
# This is the sbatch version — submit it from a login node:
#
#   sbatch examples/training_scripts/tutorial_hybrid_batch.sh
#
# For the interactive-node version (no scheduler), see
# tutorial_hybrid_interactive.sh.
#
# Model: 16-layer hybrid, ~300 M params.
# Layer pattern: MMM*MMM*MMM*MMM*  (12 Mamba + 4 attention layers)
#   M = Mamba (state-space) layer
#   * = multi-head attention layer
#
# Data modes (set USE_MOCK_DATA before submitting):
#   USE_MOCK_DATA=true  (default) — synthetic data, no setup.
#   USE_MOCK_DATA=false — Common Pile CI dataset: 12 M sequences / ~12.7 B
#       tokens, GPT-2 BPE tokenized, already preprocessed.  Same dataset
#       used by all Megatron-LM functional tests.  No extra setup needed.
#
# Runtime: ≈5 min on 4 H100s.

#SBATCH -p interactive
#SBATCH --account=nemotron_sw_pre
#SBATCH --nodes=1
#SBATCH -t 0:30:00
#SBATCH --mem=0
# Adjust ntasks-per-node / gpus-per-node to match your node type.
# H100 nodes: 8 GPUs/node.  GB200/GB300 nodes: 4 GPUs/node.
# This script defaults to 4 GPUs so it runs on either type without change.
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --dependency=singleton
#SBATCH --job-name=tutorial_hybrid

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=0   # disable cuDNN fused attention

# ---------------------------------------------------------------------------
# Paths — set ROOT_DIR to a directory you have write access to.
# It should be the parent of megatron-lm/ and images/.
#
# Example: ROOT_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/$(whoami)"
# ---------------------------------------------------------------------------
ROOT_DIR=""
REPO_DIR="${ROOT_DIR}/megatron-lm"
NAME="tutorial_hybrid"
IMAGE_PATH="${ROOT_DIR}/images/mcore_ci_lts.sqsh"

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

RUN_DIR="${ROOT_DIR}/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATACACHE_DIR="${ROOT_DIR}/data_cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"

mkdir -p "${LOGS_DIR}" "${CHECKPOINT_DIR}" "${DATACACHE_DIR}" "${TENSORBOARD_DIR}"

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
# Requires `wandb login` (or WANDB_API_KEY set) inside the container.
# ---------------------------------------------------------------------------
# WANDB_PROJECT="megatron-tutorial"
# WANDB_EXP_NAME="${NAME}_$(date +%Y%m%d_%H%M%S)"
# wandb_options=" \
#     --wandb-project ${WANDB_PROJECT} \
#     --wandb-exp-name ${WANDB_EXP_NAME} \
#     --wandb-save-dir ${RUN_DIR}/wandb "
wandb_options=""

# ---------------------------------------------------------------------------
# Model and training options
# ---------------------------------------------------------------------------
# Hybrid layer pattern: repeat [3 Mamba, 1 attention] four times = 16 layers.
# Swap in G (GDN), D (DSA), or E (MoE) to experiment with other layer types.
HYBRID_PATTERN="MMM*MMM*MMM*MMM*"

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
    --train-samples 800 \
    --lr-decay-style WSD \
    --lr-decay-samples 800 \
    --lr-warmup-samples 20 \
    --lr-wsd-decay-style minus_sqrt \
    --lr-wsd-decay-samples 160 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --lr 8e-4 \
    --min-lr 8e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --eval-interval 100 \
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
    --save-interval 400 \
    \
    --log-interval 10 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-progress \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    ${wandb_options} \
    \
    --distributed-timeout-minutes 10 \
    --exit-duration-in-mins 25 \
    --disable-gloo-process-groups "

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
run_cmd="python -u ${REPO_DIR}/pretrain_hybrid.py ${options}"

srun -l \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"

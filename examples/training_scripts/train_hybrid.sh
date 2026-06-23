#!/bin/bash
# Tutorial: Train a small hybrid (Mamba + attention) model with Megatron-LM.
#
# Must be run inside a container on a compute node. Two ways to get there:
#
#   Batch (submit from login node):
#     sbatch examples/training_scripts/train_hybrid.sh
#
#   Interactive (get a shell first, then run the training script):
#     bash examples/training_scripts/start_interactive_node.sh
#     bash examples/training_scripts/train_hybrid.sh
#
# Override defaults:
#   ROOT_DIR=/my/path bash ...
#
# Model: 16-layer hybrid, ~300 M params.
# Layer pattern: MMM*MMM*MMM*MMM* (12 Mamba + 4 attention layers)
#   M = Mamba (state-space) layer
#   * = multi-head attention layer
#
# Data: Common Pile CI dataset (12 M sequences / ~12.7 B tokens, GPT-2 BPE).
#   Same dataset as Megatron-LM functional tests; no extra setup needed.
#
# Runtime: ≈5 min on 8 H100s (8000 samples, 1000 steps).

# ---------------------------------------------------------------------------
# SLURM batch directives — read by sbatch; treated as comments otherwise.
# ---------------------------------------------------------------------------
#SBATCH -p batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --nodes=1
#SBATCH -t 1:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --dependency=singleton
#SBATCH --job-name=train_hybrid

set -eu

# ---------------------------------------------------------------------------
# GPU count — must be non-zero; fail fast before any other setup.
# ---------------------------------------------------------------------------
GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "${GPUS_PER_NODE}" -eq 0 ]; then
    echo "Error: no GPUs detected. Run this script inside a container on a compute node."
    echo "  Batch:       sbatch examples/training_scripts/train_hybrid.sh"
    echo "  Interactive: bash examples/training_scripts/start_interactive_node.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=0

# ---------------------------------------------------------------------------
# Paths
#
# ROOT_DIR must be the parent of megatron-lm/ and images/ on the shared Lustre
# filesystem. It is accessible from login nodes, compute nodes, and inside
# containers (/lustre is mounted everywhere).
# ---------------------------------------------------------------------------
# whoami returns 'root' inside containers, so we rely on ROOT_DIR being set
# explicitly (forwarded by start_interactive_node.sh) or on SLURM_JOB_USER
# (set by SLURM to the submitting account for batch jobs).
if [ -z "${ROOT_DIR:-}" ]; then
    if [ -z "${SLURM_JOB_USER:-}" ]; then
        echo "Error: set ROOT_DIR to the parent of megatron-lm/ on Lustre." >&2
        exit 1
    fi
    ROOT_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/${SLURM_JOB_USER}"
fi
REPO_DIR="${ROOT_DIR}/megatron-lm"
IMAGE_PATH="${ROOT_DIR}/images/mcore_ci_lts.sqsh"

NAME="train_hybrid"
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
RUN_DIR="${ROOT_DIR}/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATACACHE_DIR="${ROOT_DIR}/data_cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"

mkdir -p "${LOGS_DIR}" "${CHECKPOINT_DIR}" "${DATACACHE_DIR}" "${TENSORBOARD_DIR}"

# ---------------------------------------------------------------------------
# Data — Common Pile v0.1, GPT-2 BPE tokenized.
# 12 M sequences / ~12.7 B tokens; already preprocessed.
# Canonical path on this cluster (mapped to /mnt/artifacts inside CI containers).
# ---------------------------------------------------------------------------
CI_DATA_ROOT="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_mcore/mcore_ci"
DATA_PATH="${CI_DATA_ROOT}/text/common_pile/v01_filtered_data/my-gpt3_00_text_document"
GPT2_VOCAB="${CI_DATA_ROOT}/text/common_pile/v01_filtered_data/bpe/vocab.json"
GPT2_MERGES="${CI_DATA_ROOT}/text/common_pile/v01_filtered_data/bpe/merges.txt"

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

# ---------------------------------------------------------------------------
# W&B logging
# Requires wandb login (or WANDB_API_KEY set) inside the container.
# ---------------------------------------------------------------------------
WANDB_PROJECT="megatron-$(basename "${ROOT_DIR}")"
WANDB_EXP_NAME="${NAME}_${DATETIME}"
wandb_options=" \
     --wandb-project ${WANDB_PROJECT} \
     --wandb-exp-name ${WANDB_EXP_NAME} \
     --wandb-save-dir ${RUN_DIR}/wandb "

# ---------------------------------------------------------------------------
# Model and training options
# ---------------------------------------------------------------------------
# Layer pattern: [3 Mamba + 1 attention] × 4 = 16 layers.
# Replace M with G (GDN), D (DSA), or E (MoE) to try other layer types.
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
    --train-samples 8000 \
    --lr-decay-style WSD \
    --lr-decay-samples 8000 \
    --lr-warmup-samples 400 \
    --lr-wsd-decay-style minus_sqrt \
    --lr-wsd-decay-samples 1600 \
    --micro-batch-size 1 \
    --global-batch-size ${GPUS_PER_NODE} \
    --lr 8e-4 \
    --min-lr 8e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --eval-interval 400 \
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
    --save-interval 1000 \
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
    --exit-duration-in-mins 55 \
    --disable-gloo-process-groups "

# ---------------------------------------------------------------------------
# Launch
#
# If INSIDE_CONTAINER is not set, spin up the container via srun and re-invoke
# this script inside it.  start_interactive_node.sh sets INSIDE_CONTAINER=1
# so interactive sessions skip this step.
# ---------------------------------------------------------------------------
SCRIPT_ABS="${REPO_DIR}/examples/training_scripts/train_hybrid.sh"

if [ -z "${INSIDE_CONTAINER:-}" ]; then
    exec srun -l \
        --ntasks=1 \
        --container-image "${IMAGE_PATH}" \
        --container-mounts "/lustre:/lustre,${HOME}:${HOME}" \
        --no-container-mount-home \
        --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
        sh -c "INSIDE_CONTAINER=1 ROOT_DIR='${ROOT_DIR}' HOME='${HOME}' bash ${SCRIPT_ABS}"
fi

echo "Launching ${GPUS_PER_NODE}-GPU training."
echo "Checkpoints: ${CHECKPOINT_DIR}"

uv --project "${REPO_DIR}" run python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${GPUS_PER_NODE}" \
    "${REPO_DIR}/pretrain_hybrid.py" \
    ${options}

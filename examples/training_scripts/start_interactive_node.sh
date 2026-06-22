#!/bin/bash
# Get an interactive GPU node and open a shell inside the Megatron-LM container.
#
# Usage:
#   bash examples/training_scripts/start_interactive_node.sh
#
# Inside the container:
#   /opt/megatron-lm  ->  your megatron-lm checkout on Lustre
#   /lustre           ->  full cluster Lustre filesystem
#   $HOME             ->  your home directory (claude config, credentials, etc.)
#
# Then run the training tutorial (for example):
#   bash examples/training_scripts/train_hybrid.sh
#
# Override defaults:
#   ROOT_DIR=...  GPUS_PER_NODE=8  TIME=4:00:00  bash start_interactive_node.sh

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/$(whoami)}"
TIME="${TIME:-2:00:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
PARTITION="${PARTITION:-interactive}"
ACCOUNT="${ACCOUNT:-nemotron_sw_pre}"
IMAGE="${IMAGE:-${ROOT_DIR}/images/mcore_ci_lts.sqsh}"

MOUNTS="/lustre:/lustre"
MOUNTS+=",${ROOT_DIR}/megatron-lm:/opt/megatron-lm"
MOUNTS+=",${HOME}:${HOME}"

# Export HOME so tools that look up ~/.config, ~/.claude, etc. find the right path.
INIT="export HOME='${HOME}'; export PATH=\"\$HOME/.local/bin:\$PATH\"; cd /opt/megatron-lm; exec /bin/bash"

exec srun \
    --nodes=1 \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --container-image="${IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-workdir=/opt/megatron-lm \
    --time="${TIME}" \
    --pty /bin/bash --login -c "${INIT}"

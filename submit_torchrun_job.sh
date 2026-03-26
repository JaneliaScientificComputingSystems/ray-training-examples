#!/bin/bash
#===============================================================================
# Universal Torchrun Job Submission Script
# Janelia HPC — auto-configures NCCL for IB (H100/H200) or Ethernet (L4/A100)
#
# Handles single-node and multi-node torchrun launches via LSF/blaunch.
# Sources nccl_ib.sh automatically for InfiniBand queues.
#
# Usage:
#   ./submit_torchrun_job.sh <num_nodes> --script=SCRIPT [options] [-- script_args...]
#
# Examples:
#   # Multi-node on parallel queues (whole nodes, all GPUs):
#   ./submit_torchrun_job.sh 2 --script=train.py --venv=~/myenv
#   ./submit_torchrun_job.sh 4 --queue=gpu_h200_parallel --script=train.py \
#       -- --epochs=50 --batch-size=128
#
#   # Single-node on non-parallel queues (request specific GPU count):
#   ./submit_torchrun_job.sh 1 --queue=gpu_h100 --num-gpus=4 --script=train.py
#   ./submit_torchrun_job.sh 1 --queue=gpu_h200 --num-gpus=8 --script=train.py \
#       -- --lr=0.001
#
#   # With conda/mamba instead of venv:
#   ./submit_torchrun_job.sh 2 --queue=gpu_h200_parallel \
#       --conda=RNAnix --modules=cuda/12.8,gcc/12.3 \
#       --workdir=~/RNAnix --script=~/RNAnix/runner/train_rna.py \
#       -- --dtype bf16 --max_steps 50000
#
# Parallel queues (whole nodes, multi-node):
#   gpu_h200_parallel  -> 96 CPUs, 8 GPUs, InfiniBand NCCL
#   gpu_h100_parallel  -> 96 CPUs, 8 GPUs, InfiniBand NCCL  [default]
#   gpu_l4_parallel    -> 64 CPUs, 8 GPUs, Ethernet NCCL
#   gpu_a100_parallel  -> 48 CPUs, 4 GPUs, Ethernet NCCL
#
# Non-parallel queues (single node, flexible GPU count):
#   gpu_h200           -> use --num-gpus=N (1-8), InfiniBand NCCL
#   gpu_h100           -> use --num-gpus=N (1-8), InfiniBand NCCL
#   gpu_l4             -> use --num-gpus=N (1-8), Ethernet NCCL
#   gpu_a100           -> use --num-gpus=N (1-4), Ethernet NCCL
#===============================================================================

usage() {
    echo "Usage: $0 <num_nodes> --script=SCRIPT [options] [-- script_args...]"
    echo ""
    echo "Required:"
    echo "  <num_nodes>          Number of nodes"
    echo "  --script=FILE        Python training script to run"
    echo ""
    echo "Optional:"
    echo "  --queue=QUEUE        LSF queue (default: gpu_h100_parallel)"
    echo "  --num-gpus=N         GPUs to request (non-parallel queues, default: 1)"
    echo "  --num-cpus=N         CPUs to request (non-parallel queues, default: 12 per GPU)"
    echo "  --job-name=NAME      Job name (default: torchrun_job)"
    echo "  --walltime=TIME      Walltime H:MM (default: 24:00 non-parallel, auto parallel)"
    echo "  --venv=PATH          Python venv to activate"
    echo "  --conda=NAME         Conda/mamba environment to activate"
    echo "  --modules=MOD1,MOD2  Modules to load (comma-separated, e.g. cuda/12.8,gcc/12.3)"
    echo "  --workdir=PATH       Working directory (default: script's directory)"
    echo "  --output-dir=PATH    Output log directory (default: ../output)"
    echo ""
    echo "Script arguments: use '--' separator"
    echo "  Example: $0 2 --script=train.py -- --epochs=50 --batch-size=128"
    exit 1
}

if [ $# -lt 2 ]; then usage; fi

NUM_NODES=$1
shift

QUEUE_NAME="gpu_h100_parallel"
PYTHON_SCRIPT=""
JOB_NAME="torchrun_job"
WALLTIME=""
VENV_PATH=""
CONDA_ENV=""
MODULES=""
WORK_DIR=""
OUTPUT_DIR=""
SCRIPT_ARGS=""
USER_NUM_GPUS=""
USER_NUM_CPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --queue=*)      QUEUE_NAME="${1#*=}"; shift ;;
        --script=*)     PYTHON_SCRIPT="${1#*=}"; shift ;;
        --num-gpus=*)   USER_NUM_GPUS="${1#*=}"; shift ;;
        --num-cpus=*)   USER_NUM_CPUS="${1#*=}"; shift ;;
        --job-name=*)   JOB_NAME="${1#*=}"; shift ;;
        --walltime=*)   WALLTIME="${1#*=}"; shift ;;
        --venv=*)       VENV_PATH="${1#*=}"; shift ;;
        --conda=*)      CONDA_ENV="${1#*=}"; shift ;;
        --modules=*)    MODULES="${1#*=}"; shift ;;
        --workdir=*)    WORK_DIR="${1#*=}"; shift ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --) shift; SCRIPT_ARGS="$*"; break ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$PYTHON_SCRIPT" ]; then echo "ERROR: --script is required"; usage; fi
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "ERROR: Script not found: $PYTHON_SCRIPT"; exit 1; fi

# Resolve absolute path of the script
PYTHON_SCRIPT=$(cd "$(dirname "$PYTHON_SCRIPT")" && pwd)/$(basename "$PYTHON_SCRIPT")

# Default working directory: where the script lives
if [ -z "$WORK_DIR" ]; then
    WORK_DIR=$(dirname "$PYTHON_SCRIPT")
fi
WORK_DIR=$(cd "$WORK_DIR" && pwd)

# Default output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${WORK_DIR}/../output"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

# Auto-configure resources and network backend per queue
PARALLEL_QUEUE=true
case $QUEUE_NAME in
    gpu_h200_parallel|gpu_h100_parallel)
        CPUS_PER_NODE=96
        GPUS_PER_NODE=8
        APP_PROFILE="parallel-96"
        NETWORK_BACKEND="IB"
        ;;
    gpu_l4_parallel)
        CPUS_PER_NODE=64
        GPUS_PER_NODE=8
        APP_PROFILE="parallel-64"
        NETWORK_BACKEND="ETH"
        ;;
    gpu_a100_parallel)
        CPUS_PER_NODE=48
        GPUS_PER_NODE=4
        APP_PROFILE="parallel-48"
        NETWORK_BACKEND="ETH"
        ;;
    gpu_h200|gpu_h100)
        PARALLEL_QUEUE=false
        GPUS_PER_NODE=${USER_NUM_GPUS:-1}
        CPUS_PER_NODE=${USER_NUM_CPUS:-$((GPUS_PER_NODE * 12))}
        APP_PROFILE=""
        NETWORK_BACKEND="IB"
        ;;
    gpu_l4|gpu_l4_large|gpu_l4_16)
        PARALLEL_QUEUE=false
        GPUS_PER_NODE=${USER_NUM_GPUS:-1}
        CPUS_PER_NODE=${USER_NUM_CPUS:-$((GPUS_PER_NODE * 8))}
        APP_PROFILE=""
        NETWORK_BACKEND="ETH"
        ;;
    gpu_a100)
        PARALLEL_QUEUE=false
        GPUS_PER_NODE=${USER_NUM_GPUS:-1}
        CPUS_PER_NODE=${USER_NUM_CPUS:-$((GPUS_PER_NODE * 12))}
        APP_PROFILE=""
        NETWORK_BACKEND="ETH"
        ;;
    *)
        echo "ERROR: Unsupported queue: $QUEUE_NAME"
        echo "Parallel:     gpu_h200_parallel, gpu_h100_parallel, gpu_l4_parallel, gpu_a100_parallel"
        echo "Non-parallel: gpu_h200, gpu_h100, gpu_l4, gpu_a100 (use --num-gpus=N)"
        exit 1
        ;;
esac

if [ "$PARALLEL_QUEUE" = false ] && [ $NUM_NODES -gt 1 ]; then
    echo "ERROR: Non-parallel queues only support 1 node. Use a parallel queue for multi-node."
    exit 1
fi

TOTAL_CPUS=$((NUM_NODES * CPUS_PER_NODE))
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# Default walltime
if [ -z "$WALLTIME" ]; then
    if [ "$PARALLEL_QUEUE" = true ]; then
        [ $NUM_NODES -le 2 ] && WALLTIME="4:00" || WALLTIME="8:00"
    else
        WALLTIME="24:00"
    fi
fi

# NCCL env file path (for IB queues)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NCCL_ENV_FILE="${SCRIPT_DIR}/nccl_ib.sh"

echo "================================================================"
echo "Torchrun Job Submission"
echo "================================================================"
echo "Script:          $PYTHON_SCRIPT"
echo "Script args:     $SCRIPT_ARGS"
echo "Working dir:     $WORK_DIR"
echo "Nodes:           $NUM_NODES"
echo "Queue:           $QUEUE_NAME"
echo "Network backend: $NETWORK_BACKEND"
echo "GPUs per node:   $GPUS_PER_NODE"
echo "Total GPUs:      $TOTAL_GPUS"
echo "Total CPUs:      $TOTAL_CPUS"
echo "Walltime:        $WALLTIME"
if [ -n "$CONDA_ENV" ]; then echo "Conda env:       $CONDA_ENV"; fi
if [ -n "$VENV_PATH" ]; then echo "Python venv:     $VENV_PATH"; fi
if [ -n "$MODULES" ]; then echo "Modules:         $MODULES"; fi
echo "================================================================"

#-----------------------------------------------------------------------
# Write worker launch script to shared NFS (accessible from all nodes)
# Written at submission time to avoid heredoc quoting nightmares.
# Dynamic values (MASTER_ADDR, MASTER_PORT, NODE_RANK) passed as args.
#-----------------------------------------------------------------------
WORKER_SCRIPT="${OUTPUT_DIR}/.torchrun_worker_$$.sh"

cat > "$WORKER_SCRIPT" << 'WORKER_EOF'
#!/bin/bash
# Per-node torchrun worker — launched by blaunch on each node
# Usage: worker.sh NODE_RANK NUM_NODES GPUS_PER_NODE MASTER_ADDR MASTER_PORT [script] [args...]
NODE_RANK=$1; NUM_NODES=$2; GPUS_PER_NODE=$3; MASTER_ADDR=$4; MASTER_PORT=$5
shift 5

echo "================================================================"
echo "Worker node $NODE_RANK of $NUM_NODES — $(hostname) — $(date)"
echo "================================================================"

# __ENV_SETUP__ (replaced below)

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "$@"

EXIT_CODE=$?
echo "Worker $NODE_RANK finished — exit code: $EXIT_CODE — $(date)"
exit $EXIT_CODE
WORKER_EOF

# Now inject the environment setup block into the worker script.
# We build it as a temp file and splice it in to avoid sed escaping issues.
ENV_BLOCK=""

# Module loads
if [ -n "$MODULES" ]; then
    IFS=',' read -ra MOD_ARRAY <<< "$MODULES"
    for mod in "${MOD_ARRAY[@]}"; do
        ENV_BLOCK="${ENV_BLOCK}module load ${mod}"$'\n'
    done
fi

# Python environment activation
if [ -n "$VENV_PATH" ]; then
    ENV_BLOCK="${ENV_BLOCK}source ${VENV_PATH}/bin/activate"$'\n'
elif [ -n "$CONDA_ENV" ]; then
    ENV_BLOCK="${ENV_BLOCK}"'eval "$(mamba shell hook --shell bash 2>/dev/null || conda shell.bash hook)"'$'\n'
    ENV_BLOCK="${ENV_BLOCK}mamba activate ${CONDA_ENV} 2>/dev/null || conda activate ${CONDA_ENV}"$'\n'
fi

# Working directory and PYTHONPATH
ENV_BLOCK="${ENV_BLOCK}cd ${WORK_DIR}"$'\n'
ENV_BLOCK="${ENV_BLOCK}"'export PYTHONPATH="${PYTHONPATH}:'"${WORK_DIR}"'"'$'\n'

# NCCL configuration
if [ "$NETWORK_BACKEND" = "IB" ] && [ -f "$NCCL_ENV_FILE" ]; then
    ENV_BLOCK="${ENV_BLOCK}source ${NCCL_ENV_FILE}"$'\n'
else
    ENV_BLOCK="${ENV_BLOCK}export NCCL_IB_DISABLE=1"$'\n'
    ENV_BLOCK="${ENV_BLOCK}export NCCL_NET_GDR_LEVEL=0"$'\n'
    ENV_BLOCK="${ENV_BLOCK}export NCCL_P2P_DISABLE=0"$'\n'
    ENV_BLOCK="${ENV_BLOCK}export NCCL_SHM_DISABLE=0"$'\n'
    ENV_BLOCK="${ENV_BLOCK}export NCCL_BUFFSIZE=8388608"$'\n'
    ENV_BLOCK="${ENV_BLOCK}export NCCL_DEBUG=INFO"$'\n'
    ENV_BLOCK="${ENV_BLOCK}export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"$'\n'
fi

# Splice ENV_BLOCK into worker script (replace the marker line)
# Use python to avoid sed escaping issues with complex strings
python3 -c "
import sys
marker = '# __ENV_SETUP__ (replaced below)'
with open(sys.argv[1], 'r') as f:
    content = f.read()
with open(sys.argv[1], 'w') as f:
    f.write(content.replace(marker, sys.argv[2]))
" "$WORKER_SCRIPT" "$ENV_BLOCK"

chmod +x "$WORKER_SCRIPT"

echo "Worker script: $WORKER_SCRIPT"

#-----------------------------------------------------------------------
# Build BSUB directives
#-----------------------------------------------------------------------
APP_LINE=""
SPAN_LINE=""
if [ -n "$APP_PROFILE" ]; then
    APP_LINE="#BSUB -app ${APP_PROFILE}"
fi
if [ "$PARALLEL_QUEUE" = true ]; then
    SPAN_LINE="#BSUB -R \"span[ptile=${CPUS_PER_NODE}]\""
fi

cat << EOF | bsub
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -n ${TOTAL_CPUS}
${APP_LINE}
#BSUB -q ${QUEUE_NAME}
#BSUB -gpu "num=${GPUS_PER_NODE}:mode=exclusive_process"
${SPAN_LINE}
#BSUB -o ${OUTPUT_DIR}/${JOB_NAME}_%J.out
#BSUB -e ${OUTPUT_DIR}/${JOB_NAME}_%J.err
#BSUB -W ${WALLTIME}

echo "================================================================"
echo "Torchrun Distributed Training"
echo "================================================================"
echo "Job ID:       \$LSB_JOBID"
echo "Date:         \$(date)"
echo "Working dir:  ${WORK_DIR}"
echo "================================================================"

#-----------------------------------------------------------------------
# Parse LSF host allocation
#-----------------------------------------------------------------------
hosts=()
for host in \$(cat \$LSB_DJOB_HOSTFILE | uniq); do
    echo "Adding host: \$host"
    hosts+=(\$host)
done

NUM_NODES=\${#hosts[@]}
GPUS_PER_NODE=${GPUS_PER_NODE}
TOTAL_GPUS=\$((NUM_NODES * GPUS_PER_NODE))

# Master address — resolve to IP for cross-node communication
MASTER_HOST=\${hosts[0]}
MASTER_ADDR=\$(getent ahostsv4 \${MASTER_HOST} | awk 'NR==1{print \$1}')

# Find a free port
MASTER_PORT=\$(python -c 'import socket; s=socket.socket(); s.bind(("0.0.0.0", 0)); print(s.getsockname()[1]); s.close()')

echo "Host list:    \${hosts[@]}"
echo "Nodes:        \$NUM_NODES | GPUs: \$TOTAL_GPUS | Network: ${NETWORK_BACKEND}"
echo "Master:       \$MASTER_HOST (\$MASTER_ADDR:\$MASTER_PORT)"

# IB pre-flight check
if [ "${NETWORK_BACKEND}" = "IB" ]; then
    echo ""
    echo "Running IB pre-flight check..."
    blaunch -z \${hosts[0]} "ibv_devinfo | grep hca_id" || {
        echo "ERROR: ibv_devinfo failed — IB drivers may not be loaded on this node"
        exit 1
    }
    active_ports=\$(blaunch -z \${hosts[0]} "ibstat 2>/dev/null | grep -c 'State: Active'" 2>/dev/null || echo 0)
    echo "Active IB ports on head node: \$active_ports (expected 8)"
    if [ "\$active_ports" -lt 8 ]; then
        echo "WARNING: fewer than 8 IB ports active — some rails may be down"
    fi
fi

echo ""
echo "================================================================"
echo "Launching torchrun..."
echo "================================================================"

#-----------------------------------------------------------------------
# Launch torchrun on each node via blaunch
#-----------------------------------------------------------------------
if [ \$NUM_NODES -eq 1 ]; then
    # Single-node: run worker script directly (node_rank=0)
    ${WORKER_SCRIPT} 0 1 \$GPUS_PER_NODE 127.0.0.1 \$MASTER_PORT \
        ${PYTHON_SCRIPT} ${SCRIPT_ARGS}
    EXIT_CODE=\$?
else
    # Multi-node: blaunch worker script on each node
    PIDS=()
    for i in "\${!hosts[@]}"; do
        host=\${hosts[\$i]}
        echo "  Node \$i: \$host"

        blaunch -z \$host ${WORKER_SCRIPT} \$i \$NUM_NODES \$GPUS_PER_NODE \$MASTER_ADDR \$MASTER_PORT \
            ${PYTHON_SCRIPT} ${SCRIPT_ARGS} &
        PIDS+=(\$!)
    done

    # Wait for all nodes — capture first non-zero exit code
    EXIT_CODE=0
    for pid in "\${PIDS[@]}"; do
        wait \$pid
        rc=\$?
        if [ \$rc -ne 0 ] && [ \$EXIT_CODE -eq 0 ]; then
            EXIT_CODE=\$rc
        fi
    done
fi

# Cleanup worker script
rm -f ${WORKER_SCRIPT}

echo ""
echo "================================================================"
if [ \$EXIT_CODE -ne 0 ]; then
    echo "JOB FAILED — exit code: \$EXIT_CODE"
else
    echo "JOB COMPLETED SUCCESSFULLY"
fi
echo "End time: \$(date)"
echo "================================================================"
if [ "${NETWORK_BACKEND}" = "IB" ]; then
    echo "Tip: grep 'NET/IB' in the .out file to confirm NCCL selected InfiniBand"
    echo "     If you see NET/Socket, the HCA names may have changed — run ibv_devinfo"
fi

exit \$EXIT_CODE
EOF

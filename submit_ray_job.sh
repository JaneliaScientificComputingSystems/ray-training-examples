#!/bin/bash
#===============================================================================
# Universal Ray Cluster Job Submission Script
# Janelia HPC — auto-configures NCCL for IB (H100/H200) or Ethernet (L4/A100)
#
# Usage:
#   ./submit_ray_job.sh <num_nodes> --script=SCRIPT [options] [-- script_args...]
#
# Examples:
#   # Multi-node on parallel queues (whole nodes, all GPUs):
#   ./submit_ray_job.sh 2 --script=test_ray_cluster.py
#   ./submit_ray_job.sh 8 --queue=gpu_h200_parallel --venv=~/ray_env \
#       --script=train_ray.py -- --epochs=50
#
#   # Single-node on non-parallel queues (request specific GPU count):
#   ./submit_ray_job.sh 1 --queue=gpu_h100 --num-gpus=2 --venv=~/ray_env \
#       --script=train_ray.py -- --epochs=20
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
    echo "  --script=FILE        Python script to run"
    echo ""
    echo "Optional:"
    echo "  --queue=QUEUE        LSF queue (default: gpu_h100_parallel)"
    echo "  --num-gpus=N         GPUs to request (non-parallel queues only, default: all)"
    echo "  --job-name=NAME      Job name (default: ray_job)"
    echo "  --walltime=TIME      Walltime in hours (default: auto)"
    echo "  --venv=PATH          Python venv path"
    echo ""
    echo "Script arguments: use '--' separator"
    echo "  Example: $0 2 --script=train.py -- --epochs=50 --batch-size=128"
    echo "  Example: $0 1 --queue=gpu_h100 --num-gpus=2 --script=train.py -- --epochs=10"
    exit 1
}

if [ $# -lt 2 ]; then usage; fi

NUM_NODES=$1
shift

QUEUE_NAME="gpu_h100_parallel"
PYTHON_SCRIPT=""
JOB_NAME="ray_job"
WALLTIME=""
VENV_PATH=""
SCRIPT_ARGS=""
USER_NUM_GPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --queue=*)    QUEUE_NAME="${1#*=}"; shift ;;
        --script=*)   PYTHON_SCRIPT="${1#*=}"; shift ;;
        --num-gpus=*) USER_NUM_GPUS="${1#*=}"; shift ;;
        --job-name=*) JOB_NAME="${1#*=}"; shift ;;
        --walltime=*) WALLTIME="${1#*=}"; shift ;;
        --venv=*)     VENV_PATH="${1#*=}"; shift ;;
        --) shift; SCRIPT_ARGS=$(printf '%q ' "$@"); break ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$PYTHON_SCRIPT" ]; then echo "ERROR: --script is required"; usage; fi
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "ERROR: Script not found: $PYTHON_SCRIPT"; exit 1; fi

# Auto-configure resources and network backend per queue
PARALLEL_QUEUE=true
case $QUEUE_NAME in
    # Parallel queues — whole nodes, multi-node capable
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
    # Non-parallel queues — single node, flexible GPU count
    gpu_h200|gpu_h100)
        PARALLEL_QUEUE=false
        GPUS_PER_NODE=${USER_NUM_GPUS:-8}
        CPUS_PER_NODE=$((GPUS_PER_NODE * 12))
        APP_PROFILE=""
        NETWORK_BACKEND="IB"
        ;;
    gpu_l4|gpu_l4_large|gpu_l4_16)
        PARALLEL_QUEUE=false
        GPUS_PER_NODE=${USER_NUM_GPUS:-8}
        CPUS_PER_NODE=$((GPUS_PER_NODE * 8))
        APP_PROFILE=""
        NETWORK_BACKEND="ETH"
        ;;
    gpu_a100)
        PARALLEL_QUEUE=false
        GPUS_PER_NODE=${USER_NUM_GPUS:-4}
        CPUS_PER_NODE=$((GPUS_PER_NODE * 12))
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
[ -z "$WALLTIME" ] && { [ $NUM_NODES -le 2 ] && WALLTIME="4:00" || WALLTIME="8:00"; }

echo "================================================================"
echo "Ray Cluster Job Submission"
echo "================================================================"
echo "Script:          $PYTHON_SCRIPT"
echo "Script args:     $SCRIPT_ARGS"
echo "Nodes:           $NUM_NODES"
echo "Queue:           $QUEUE_NAME"
echo "Network backend: $NETWORK_BACKEND"
echo "Total GPUs:      $TOTAL_GPUS"
echo "Total CPUs:      $TOTAL_CPUS"
echo "Walltime:        $WALLTIME"
echo "================================================================"

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
#BSUB -J ${JOB_NAME}_${NUM_NODES}nodes
#BSUB -n ${TOTAL_CPUS}
${APP_LINE}
#BSUB -q ${QUEUE_NAME}
#BSUB -gpu "num=${GPUS_PER_NODE}:mode=exclusive_process"
${SPAN_LINE}
#BSUB -o ../output/${JOB_NAME}_%J.out
#BSUB -e ../output/${JOB_NAME}_%J.err
#BSUB -W ${WALLTIME}

if [ -n "${VENV_PATH}" ]; then source ${VENV_PATH}/bin/activate; fi

#-----------------------------------------------------------------------
# NCCL configuration — sourced from nccl_ib.sh for IB queues
# All NCCL env vars live in one place to avoid duplication
#-----------------------------------------------------------------------
if [ "${NETWORK_BACKEND}" = "IB" ]; then
    echo "NCCL backend: InfiniBand"
    source "$(cd "$(dirname "$0")" && pwd)/nccl_ib.sh"
    echo "  HCAs: \$NCCL_IB_HCA"
else
    echo "NCCL backend: Ethernet"
    export NCCL_IB_DISABLE=1
    export NCCL_NET_GDR_LEVEL=0
    export NCCL_P2P_DISABLE=0
    export NCCL_SHM_DISABLE=0
    export NCCL_BUFFSIZE=8388608
    export NCCL_DEBUG=INFO
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

export RAY_TMPDIR="/scratch/\$(whoami)/ray_\$(whoami)"
mkdir -p \$RAY_TMPDIR

function getfreeport() {
    CHECK="do while"
    while [[ ! -z \$CHECK ]]; do
        port=\$(( ( RANDOM % 40000 ) + 20000 ))
        CHECK=\$(ss -tln | grep ":\$port ")
    done
    echo \$port
}

hosts=()
for host in \$(cat \$LSB_DJOB_HOSTFILE | uniq); do
    echo "Adding host: \$host"
    hosts+=(\$host)
done
echo "Host list: \${hosts[@]}"
echo "Nodes: ${NUM_NODES} | GPUs: ${TOTAL_GPUS} | Network: ${NETWORK_BACKEND}"

for host in "\${hosts[@]}"; do
    blaunch -z \$host "mkdir -p /scratch/\$(whoami)/ray_\$(whoami)" &
done
wait

# IB pre-flight check (IB queues only)
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

port=\$(getfreeport)
head_node=\${hosts[0]}
echo ""
echo "Head node: \$head_node  Port: \$port"

# Path to NCCL env file — sourced by ray start shells so the daemon
# (and all workers it spawns) inherit the vars.
if [ "${NETWORK_BACKEND}" = "IB" ]; then
    NCCL_ENV_FILE="$(cd "$(dirname "$0")" && pwd)/nccl_ib.sh"
else
    NCCL_ENV_FILE=""
fi

echo ""
echo "================================================================"
echo "Starting Ray cluster..."
echo "================================================================"

blaunch -z \$head_node "
    if [ -n '${VENV_PATH}' ]; then source ${VENV_PATH}/bin/activate; fi
    if [ -n '\$NCCL_ENV_FILE' ] && [ -f '\$NCCL_ENV_FILE' ]; then source \$NCCL_ENV_FILE; fi
    ray start --head --port \$port \
        --num-cpus=${CPUS_PER_NODE} \
        --num-gpus=${GPUS_PER_NODE} \
        --temp-dir=/scratch/\$(whoami)/ray_\$(whoami) \
        --dashboard-agent-grpc-port=20100 \
        --dashboard-agent-listen-port=20101
" &

sleep 20
while ! ray status --address \$head_node:\$port 2>/dev/null; do
    echo "Waiting for Ray head node..."
    sleep 3
done
echo "Ray head node ready"

if [ ${NUM_NODES} -gt 1 ]; then
    workers=("\${hosts[@]:1}")
    for host in "\${workers[@]}"; do
        blaunch -z \$host "
            if [ -n '${VENV_PATH}' ]; then source ${VENV_PATH}/bin/activate; fi
            if [ -n '\$NCCL_ENV_FILE' ] && [ -f '\$NCCL_ENV_FILE' ]; then source \$NCCL_ENV_FILE; fi
            ray start --address \$head_node:\$port \
                --num-cpus=${CPUS_PER_NODE} \
                --num-gpus=${GPUS_PER_NODE} \
                --temp-dir=/scratch/\$(whoami)/ray_\$(whoami) \
                --dashboard-agent-grpc-port=20100 \
                --dashboard-agent-listen-port=20101
        " &
        sleep 10
        while ! blaunch -z \$host ray status --address \$head_node:\$port 2>/dev/null; do
            echo "Waiting for worker \$host..."
            sleep 3
        done
        echo "Worker \$host connected"
    done
fi

echo ""
echo "================================================================"
echo "Ray cluster ready"
ray status --address \$head_node:\$port
echo "================================================================"
if [ "${NETWORK_BACKEND}" = "IB" ]; then
    echo "Tip: grep 'NET/IB' in the .out file to confirm NCCL selected InfiniBand"
    echo "     If you see NET/Socket, the HCA names may have changed — run ibv_devinfo"
fi
echo ""

python ${PYTHON_SCRIPT} ${SCRIPT_ARGS}
script_exit_code=\$?
echo "Script exited: \$script_exit_code"

for host in "\${hosts[@]}"; do
    blaunch -z \$host "if [ -n '${VENV_PATH}' ]; then source ${VENV_PATH}/bin/activate; fi; ray stop" &
done
wait
echo "Ray cluster stopped"

if [ \$script_exit_code -ne 0 ]; then
    echo "================================================================"
    echo "JOB FAILED — exit code: \$script_exit_code"
    echo "================================================================"
    exit \$script_exit_code
else
    echo "================================================================"
    echo "JOB COMPLETED SUCCESSFULLY"
    echo "================================================================"
fi
EOF

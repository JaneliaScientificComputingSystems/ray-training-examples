#!/bin/bash
#===============================================================================
# CPU-Only Ray Cluster Job Submission Script
# Janelia HPC — launches a Ray cluster on CPU-only LSF queues
#
# Usage:
#   ./submit_ray_cpu_job.sh --num-cpus=N --script=SCRIPT [options] [-- script_args...]
#
# Examples:
#   # 8 CPUs (LSF picks the node):
#   ./submit_ray_cpu_job.sh --num-cpus=8 --script=my_pipeline.py
#
#   # 300 CPUs spread across nodes (LSF decides layout):
#   ./submit_ray_cpu_job.sh --num-cpus=300 --script=distributed_etl.py -- --input /data
#
#   # Quick test on short queue (1 hour max):
#   ./submit_ray_cpu_job.sh --queue=short --num-cpus=4 --script=test.py
#
# Queue:
#   cpu_parallel -> multi-node, app profile parallel-128   [default]
#===============================================================================

usage() {
    echo "Usage: $0 --num-cpus=N --script=SCRIPT [options] [-- script_args...]"
    echo ""
    echo "Required:"
    echo "  --num-cpus=N         Total CPUs to request"
    echo "  --script=FILE        Python script to run"
    echo ""
    echo "Optional:"
    echo "  --queue=QUEUE        LSF queue (default: cpu_parallel)"
    echo "  --job-name=NAME      Job name (default: ray_cpu_job)"
    echo "  --walltime=TIME      Walltime HH:MM (default: 24:00 local, 1:00 short)"
    echo "  --venv=PATH          Python venv path"
    echo ""
    echo "Script arguments: use '--' separator"
    echo "  Example: $0 --num-cpus=300 --script=etl.py -- --input /data"
    exit 1
}

if [ $# -lt 1 ]; then usage; fi

QUEUE_NAME="cpu_parallel"
PYTHON_SCRIPT=""
JOB_NAME="ray_cpu_job"
WALLTIME=""
VENV_PATH=""
SCRIPT_ARGS=""
TOTAL_CPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --queue=*)    QUEUE_NAME="${1#*=}"; shift ;;
        --script=*)   PYTHON_SCRIPT="${1#*=}"; shift ;;
        --num-cpus=*) TOTAL_CPUS="${1#*=}"; shift ;;
        --job-name=*) JOB_NAME="${1#*=}"; shift ;;
        --walltime=*) WALLTIME="${1#*=}"; shift ;;
        --venv=*)     VENV_PATH="${1#*=}"; shift ;;
        --) shift; SCRIPT_ARGS=$(printf '%q ' "$@"); break ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$TOTAL_CPUS" ]; then echo "ERROR: --num-cpus is required"; usage; fi
if [ -z "$PYTHON_SCRIPT" ]; then echo "ERROR: --script is required"; usage; fi
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "ERROR: Script not found: $PYTHON_SCRIPT"; exit 1; fi

if [ -z "$WALLTIME" ]; then WALLTIME="24:00"; fi

echo "================================================================"
echo "Ray CPU Cluster Job Submission"
echo "================================================================"
echo "Script:          $PYTHON_SCRIPT"
echo "Script args:     $SCRIPT_ARGS"
echo "Queue:           $QUEUE_NAME"
echo "Total CPUs:      $TOTAL_CPUS"
echo "Walltime:        $WALLTIME"
echo "================================================================"

cat << EOF | bsub
#!/bin/bash
#BSUB -J ${JOB_NAME}
#BSUB -n ${TOTAL_CPUS}
#BSUB -q ${QUEUE_NAME}
#BSUB -app parallel-128
#BSUB -o ../output/${JOB_NAME}_%J.out
#BSUB -e ../output/${JOB_NAME}_%J.err
#BSUB -W ${WALLTIME}

if [ -n "${VENV_PATH}" ]; then source ${VENV_PATH}/bin/activate; fi

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

# Discover hosts and CPUs per host from LSF allocation
declare -A host_cpus
for host in \$(cat \$LSB_DJOB_HOSTFILE); do
    host_cpus[\$host]=\$(( \${host_cpus[\$host]:-0} + 1 ))
done
hosts=(\$(echo "\${!host_cpus[@]}" | tr ' ' '\n' | sort))
num_nodes=\${#hosts[@]}

echo "Allocated nodes: \$num_nodes"
for host in "\${hosts[@]}"; do
    echo "  \$host: \${host_cpus[\$host]} CPUs"
done

for host in "\${hosts[@]}"; do
    blaunch -z \$host "mkdir -p /scratch/\$(whoami)/ray_\$(whoami)" &
done
wait

port=\$(getfreeport)
head_node=\${hosts[0]}
echo ""
echo "Head node: \$head_node  Port: \$port"

echo ""
echo "================================================================"
echo "Starting Ray cluster..."
echo "================================================================"

blaunch -z \$head_node "
    if [ -n '${VENV_PATH}' ]; then source ${VENV_PATH}/bin/activate; fi
    ray start --head --port \$port \
        --num-cpus=\${host_cpus[\$head_node]} \
        --num-gpus=0 \
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

if [ \$num_nodes -gt 1 ]; then
    workers=("\${hosts[@]:1}")
    for host in "\${workers[@]}"; do
        blaunch -z \$host "
            if [ -n '${VENV_PATH}' ]; then source ${VENV_PATH}/bin/activate; fi
            ray start --address \$head_node:\$port \
                --num-cpus=\${host_cpus[\$host]} \
                --num-gpus=0 \
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
echo ""

RAY_ADDRESS=\$head_node:\$port python ${PYTHON_SCRIPT} ${SCRIPT_ARGS}
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

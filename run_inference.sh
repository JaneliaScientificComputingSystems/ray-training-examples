#!/bin/bash
#===============================================================================
# Run inference scripts as LSF jobs (single GPU)
#
# Usage:
#   ./run_inference.sh --script=image_classifier.py -- --model ../models/cifar10_resnet18_best.pth --test
#   ./run_inference.sh --script=gpt2_generate.py -- --model ../models/gpt2_small_ddp_best.pth --interactive
#   ./run_inference.sh --script=gpt2_eval.py -- --model ../models/gpt2_small_ddp_best.pth --num-batches 200
#   ./run_inference.sh --script=imagenet_classifier.py -- --model ../models/resnet50_imagenet_best.pth --test
#
# Options:
#   --queue=QUEUE    GPU queue (default: gpu_l4)
#   --venv=PATH      Python venv path
#   --script=FILE    Inference script to run
#===============================================================================

QUEUE_NAME="gpu_l4"
PYTHON_SCRIPT=""
VENV_PATH=""
SCRIPT_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --queue=*)   QUEUE_NAME="${1#*=}"; shift ;;
        --script=*)  PYTHON_SCRIPT="${1#*=}"; shift ;;
        --venv=*)    VENV_PATH="${1#*=}"; shift ;;
        --) shift; SCRIPT_ARGS=$(printf '%q ' "$@"); break ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$PYTHON_SCRIPT" ]; then
    echo "Usage: $0 --script=SCRIPT [--queue=QUEUE] [--venv=PATH] -- [script_args...]"
    exit 1
fi
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Script not found: $PYTHON_SCRIPT"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_NAME=$(basename "$PYTHON_SCRIPT" .py)

# Set app profile based on queue
case $QUEUE_NAME in
    gpu_h200*|gpu_h100*) APP_PROFILE="parallel-96" ;;
    gpu_l4_parallel)     APP_PROFILE="parallel-64" ;;
    gpu_a100_parallel)   APP_PROFILE="parallel-48" ;;
    *)                   APP_PROFILE="" ;;
esac

echo "Submitting: $PYTHON_SCRIPT $SCRIPT_ARGS"
echo "Queue: $QUEUE_NAME (1 GPU)"

APP_LINE=""
if [ -n "$APP_PROFILE" ]; then
    APP_LINE="#BSUB -app ${APP_PROFILE}"
fi

cat << EOF | bsub
#!/bin/bash
#BSUB -J ${SCRIPT_NAME}
#BSUB -n 8
#BSUB -q ${QUEUE_NAME}
${APP_LINE}
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o ../output/${SCRIPT_NAME}_%J.out
#BSUB -e ../output/${SCRIPT_NAME}_%J.err
#BSUB -W 1:00

if [ -n "${VENV_PATH}" ]; then source ${VENV_PATH}/bin/activate; fi

cd ${SCRIPT_DIR}
python ${PYTHON_SCRIPT} ${SCRIPT_ARGS}
EOF

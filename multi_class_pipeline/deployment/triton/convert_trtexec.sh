#!/usr/bin/env bash
# ==============================================================================
# Step 2: Convert ONNX to TensorRT Plan (model.plan)
# Must be executed on an environment matching the Triton server's GPU architecture
# and TensorRT version (e.g. inside the Triton container or GPU VM).
# ==============================================================================
set -euo pipefail

ONNX_MODEL="${1:-rfdetr_model.onnx}"
PLAN_OUTPUT="${2:-model.plan}"
RESOLUTION="${3:-560}"
MAX_BATCH="${4:-4}"

echo "=============================================================================="
echo " Converting RF-DETR ONNX to TensorRT Engine (.plan)"
echo " • Input ONNX:       ${ONNX_MODEL}"
echo " • Output Plan:      ${PLAN_OUTPUT}"
echo " • Resolution:       ${RESOLUTION}x${RESOLUTION}"
echo " • Max Batch Size:   ${MAX_BATCH}"
echo "=============================================================================="

if ! command -v trtexec &> /dev/null; then
    echo "ERROR: 'trtexec' not found in PATH."
    echo "Please run this command inside an NVIDIA TensorRT container, e.g.:"
    echo "  docker run --gpus all -it --rm -v \$(pwd):/workspace nvcr.io/nvidia/tensorrt:23.10-py3"
    exit 1
fi

trtexec \
    --onnx="${ONNX_MODEL}" \
    --saveEngine="${PLAN_OUTPUT}" \
    --fp16 \
    --minShapes=images:1x3x"${RESOLUTION}"x"${RESOLUTION}" \
    --optShapes=images:1x3x"${RESOLUTION}"x"${RESOLUTION}" \
    --maxShapes=images:"${MAX_BATCH}"x3x"${RESOLUTION}"x"${RESOLUTION}" \
    --workspace=4096 \
    --verbose

echo ""
echo "✓ TensorRT plan generated successfully: ${PLAN_OUTPUT}"
ls -lh "${PLAN_OUTPUT}"

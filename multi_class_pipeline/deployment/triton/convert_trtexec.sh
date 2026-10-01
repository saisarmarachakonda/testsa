#!/usr/bin/env bash
# ==============================================================================
# Step 2: Convert ONNX to TensorRT Plan (model.plan)
# Must be executed on an environment matching the Triton server's GPU architecture
# and TensorRT version (e.g. inside the Triton container or T4 GPU node).
#
# Target Triton config.pbtxt:
#   name: "tagdet_rt"
#   platform: "tensorrt_plan"
#   max_batch_size: 1
#   input: images [ 3, 640, 480 ]
#   output: output [ 18900, 7 ]
#
# Usage:
#   ./convert_trtexec.sh [onnx_path] [output_plan] [height] [width]
#   ./convert_trtexec.sh rfdetr_model.onnx model.plan 640 480
# ==============================================================================
set -euo pipefail

ONNX_MODEL="${1:-rfdetr_model.onnx}"
PLAN_OUTPUT="${2:-model.plan}"
INPUT_HEIGHT="${3:-640}"
INPUT_WIDTH="${4:-480}"

echo "=============================================================================="
echo " Converting RF-DETR ONNX to TensorRT Engine (.plan) for tagdet_rt"
echo " • Input ONNX:       ${ONNX_MODEL}"
echo " • Output Plan:      ${PLAN_OUTPUT}"
echo " • Input Shape:      images:1x3x${INPUT_HEIGHT}x${INPUT_WIDTH}"
echo " • Output Shape:     output:1x18900x7"
echo " • Precision:        FP16"
echo "=============================================================================="

if ! command -v trtexec &> /dev/null; then
    echo "ERROR: 'trtexec' not found in PATH."
    echo "Please run this command inside an NVIDIA TensorRT container matching the Triton pod:"
    echo "  docker run --gpus all -it --rm -v \$(pwd):/workspace nvcr.io/nvidia/tensorrt:23.10-py3"
    exit 1
fi

trtexec \
    --onnx="${ONNX_MODEL}" \
    --saveEngine="${PLAN_OUTPUT}" \
    --fp16 \
    --minShapes=images:1x3x"${INPUT_HEIGHT}"x"${INPUT_WIDTH}" \
    --optShapes=images:1x3x"${INPUT_HEIGHT}"x"${INPUT_WIDTH}" \
    --maxShapes=images:1x3x"${INPUT_HEIGHT}"x"${INPUT_WIDTH}" \
    --workspace=4096 \
    --verbose

echo ""
echo "✓ TensorRT plan generated successfully: ${PLAN_OUTPUT}"
ls -lh "${PLAN_OUTPUT}"

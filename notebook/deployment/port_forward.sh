#!/usr/bin/env bash
# ==============================================================================
# port_forward.sh: Secure TLS Port-Forwarding Tunnel for AKS Microservices
#
# Azure & Kubernetes Context (Configurable via ENV or arguments):
#   AZURE_SUBSCRIPTION="${AZURE_SUBSCRIPTION:-<YOUR_AZURE_SUBSCRIPTION>}"
#   AZURE_RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-<YOUR_RESOURCE_GROUP>}"
#   AKS_CLUSTER_NAME="${AKS_CLUSTER_NAME:-<YOUR_AKS_CLUSTER_NAME>}"
#   K8S_NAMESPACE="${K8S_NAMESPACE:-<YOUR_NAMESPACE>}"
#
# Port-Forwarded Services:
#   1. Detection Pod (TensorRT):   tensor-rt-server-v100-*      -> Local 8001:8001
#   2. OCR Pod (PaddleOCR / TRT):  tensor-rt-server-v100-ocr-*  -> Local 8002:8001
#
# Gateway Microservice (New Terminal):
#   cd multi_class_pipeline/deployment
#   python3 app.py
# ==============================================================================

set -euo pipefail

# Azure & Kubernetes Parameters (Defaults read from environment)
SUBSCRIPTION="${AZURE_SUBSCRIPTION:-}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-}"
CLUSTER_NAME="${AKS_CLUSTER_NAME:-}"
NAMESPACE="${K8S_NAMESPACE:-default}"

DETECTION_LOCAL_PORT="${DETECTION_PORT:-8001}"
DETECTION_POD_PORT="8001"

OCR_LOCAL_PORT="${OCR_PORT:-8002}"
OCR_POD_PORT="8001"

echo "=============================================================================="
echo " Azure AKS Port Forwarding Configuration"
echo "=============================================================================="

# Prompt for parameters if not set in environment
if [ -z "${SUBSCRIPTION}" ]; then
    read -p "Enter Azure Subscription Name / ID [or press Enter to skip]: " user_sub || true
    SUBSCRIPTION="${user_sub:-}"
fi

if [ -z "${RESOURCE_GROUP}" ]; then
    read -p "Enter Azure Resource Group [or press Enter to skip]: " user_rg || true
    RESOURCE_GROUP="${user_rg:-}"
fi

if [ -z "${CLUSTER_NAME}" ]; then
    read -p "Enter AKS Cluster Name [or press Enter to skip]: " user_cluster || true
    CLUSTER_NAME="${user_cluster:-}"
fi

if [ -z "${NAMESPACE}" ] || [ "${NAMESPACE}" = "default" ]; then
    read -p "Enter Kubernetes Namespace [default]: " user_ns || true
    NAMESPACE="${user_ns:-default}"
fi

echo ""
echo " Active Configuration:"
echo " - Subscription:    ${SUBSCRIPTION:-[Using Current Active Subscription]}"
echo " - Resource Group:  ${RESOURCE_GROUP:-[Using Existing Kubeconfig Context]}"
echo " - AKS Cluster:     ${CLUSTER_NAME:-[Using Existing Kubeconfig Context]}"
echo " - Namespace:       ${NAMESPACE}"
echo "=============================================================================="

# 1. Configure Azure & Kubectl Context
if command -v az &>/dev/null; then
    if [ -n "${SUBSCRIPTION}" ]; then
        echo "1. Setting Azure Subscription: ${SUBSCRIPTION}..."
        az account set --subscription "${SUBSCRIPTION}" || echo "Warning: Subscription set encountered notice."
    fi
    
    if [ -n "${RESOURCE_GROUP}" ] && [ -n "${CLUSTER_NAME}" ]; then
        echo "2. Fetching AKS Cluster Credentials for ${CLUSTER_NAME}..."
        az aks get-credentials --resource-group "${RESOURCE_GROUP}" --name "${CLUSTER_NAME}" --overwrite-existing || echo "Warning: AKS credentials fetch notice."
    fi
fi

if command -v kubectl &>/dev/null; then
    echo "3. Configuring Kubernetes Namespace Context: ${NAMESPACE}..."
    kubectl config set-context --current --namespace="${NAMESPACE}" 2>/dev/null || true
    
    echo "4. Current Running Pods in '${NAMESPACE}':"
    kubectl get pods -n "${NAMESPACE}" || true
    echo "------------------------------------------------------------------------------"
else
    echo "ERROR: 'kubectl' command not found. Please install kubectl."
    exit 1
fi

# 2. Discover Pod Names
OCR_POD=$(kubectl get pods -n "${NAMESPACE}" --no-headers -o custom-columns=":metadata.name" 2>/dev/null | grep -i "tensor-rt-server-v100-ocr" | head -n 1 || true)
DETECTION_POD=$(kubectl get pods -n "${NAMESPACE}" --no-headers -o custom-columns=":metadata.name" 2>/dev/null | grep -i "tensor-rt-server-v100" | grep -v -i "ocr" | head -n 1 || true)

if [ -z "${OCR_POD}" ]; then
    OCR_POD="tensor-rt-server-v100-ocr-xxxxxxxxxxx"
    echo "Notice: Could not automatically detect running OCR pod. Using placeholder: ${OCR_POD}"
else
    echo "✓ Detected Running OCR Pod:       ${OCR_POD}"
fi

if [ -z "${DETECTION_POD}" ]; then
    DETECTION_POD="tensor-rt-server-v100-xxxxxxxxxxx"
    echo "Notice: Could not automatically detect running Detection pod. Using placeholder: ${DETECTION_POD}"
else
    echo "✓ Detected Running Detection Pod: ${DETECTION_POD}"
fi

echo "=============================================================================="
echo " MANUAL TERMINAL COMMANDS (For running across separate terminal tabs):"
echo "=============================================================================="
echo " Terminal 1 (OCR Port Forward):"
echo "   kubectl port-forward ${OCR_POD} ${OCR_LOCAL_PORT}:${OCR_POD_PORT} -n ${NAMESPACE}"
echo ""
echo " Terminal 2 (Detection Port Forward):"
echo "   kubectl port-forward ${DETECTION_POD} ${DETECTION_LOCAL_PORT}:${DETECTION_POD_PORT} -n ${NAMESPACE}"
echo ""
echo " Terminal 3 (Unified Gateway Server):"
echo "   cd multi_class_pipeline/deployment"
echo "   python3 app.py"
echo "=============================================================================="

# 3. Clean up any existing stale port-forwards on local ports
if lsof -ti:"${OCR_LOCAL_PORT}" &>/dev/null; then
    echo "Freeing local port ${OCR_LOCAL_PORT}..."
    kill -9 $(lsof -ti:"${OCR_LOCAL_PORT}") 2>/dev/null || true
fi
if lsof -ti:"${DETECTION_LOCAL_PORT}" &>/dev/null; then
    echo "Freeing local port ${DETECTION_LOCAL_PORT}..."
    kill -9 $(lsof -ti:"${DETECTION_LOCAL_PORT}") 2>/dev/null || true
fi

echo ""
read -p "Would you like to start both tunnels in background now? [Y/n]: " choice || choice="Y"
if [[ "${choice}" =~ ^[Yy]$ ]] || [ -z "${choice}" ]; then
    echo "Starting OCR port-forward (${OCR_POD} -> ${OCR_LOCAL_PORT}:${OCR_POD_PORT})..."
    kubectl port-forward "${OCR_POD}" "${OCR_LOCAL_PORT}:${OCR_POD_PORT}" -n "${NAMESPACE}" &
    PID_OCR=$!

    echo "Starting Detection port-forward (${DETECTION_POD} -> ${DETECTION_LOCAL_PORT}:${DETECTION_POD_PORT})..."
    kubectl port-forward "${DETECTION_POD}" "${DETECTION_LOCAL_PORT}:${DETECTION_POD_PORT}" -n "${NAMESPACE}" &
    PID_DET=$!

    cleanup() {
        echo ""
        echo "Terminating port-forwarding background processes..."
        kill "${PID_OCR}" 2>/dev/null || true
        kill "${PID_DET}" 2>/dev/null || true
        exit 0
    }
    trap cleanup SIGINT SIGTERM EXIT

    echo "=============================================================================="
    echo " ✓ Tunnels Active!"
    echo "   - OCR Service:       http://127.0.0.1:${OCR_LOCAL_PORT}"
    echo "   - Detection Service: http://127.0.0.1:${DETECTION_LOCAL_PORT}"
    echo "=============================================================================="
    echo " Next Step: Open a NEW terminal and run the Unified Gateway:"
    echo "   cd $(pwd)"
    echo "   python3 app.py"
    echo "=============================================================================="
    echo " Press Ctrl+C in this terminal to terminate both tunnels when finished."
    
    wait "${PID_OCR}" "${PID_DET}"
fi

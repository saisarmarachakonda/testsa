#!/usr/bin/env bash
# ==============================================================================
# deploy.sh: Automated deployment script for RF-DETR on Azure Kubernetes Service (AKS)
# Features:
#  - Azure Resource Group & Azure Container Registry (ACR) provisioning
#  - Cloud Docker image build using `az acr build` (No local Docker needed)
#  - AKS cluster provisioning with attached ACR
#  - Deployment of internal Kubernetes Service (Zero Public IP exposure)
# ==============================================================================

set -euo pipefail

# ----------------- Configuration Defaults -----------------
RESOURCE_GROUP="${RESOURCE_GROUP:-rg-rfdetr-inference}"
LOCATION="${LOCATION:-eastus}"
AKS_CLUSTER_NAME="${AKS_CLUSTER_NAME:-aks-rfdetr-cluster}"
IMAGE_NAME="${IMAGE_NAME:-rfdetr-inference}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
NODE_COUNT="${NODE_COUNT:-2}"
NODE_VM_SIZE="${NODE_VM_SIZE:-Standard_D4s_v5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="${SCRIPT_DIR}/server"
K8S_DIR="${SCRIPT_DIR}/k8s"
MODELS_DIR="${SCRIPT_DIR}/models"

echo "=============================================================================="
echo " RF-DETR AKS Deployment (Private Internal LoadBalancer - No Public IP)"
echo "=============================================================================="
echo " - Resource Group: ${RESOURCE_GROUP}"
echo " - Location:       ${LOCATION}"
echo " - AKS Cluster:    ${AKS_CLUSTER_NAME} (VM: ${NODE_VM_SIZE}, Nodes: ${NODE_COUNT})"
echo " - Server Path:    ${SERVER_DIR}"
echo "=============================================================================="

# 1. Pre-flight Checks
if ! command -v az &>/dev/null; then
    echo "ERROR: Azure CLI ('az') is not installed or not in PATH."
    exit 1
fi
if ! command -v kubectl &>/dev/null; then
    echo "ERROR: 'kubectl' is not installed or not in PATH."
    exit 1
fi

# Ensure ONNX model exists in server directory
if [ ! -f "${SERVER_DIR}/rfdetr_model.onnx" ]; then
    if [ -f "${MODELS_DIR}/rfdetr_model.onnx" ]; then
        echo "Copying ${MODELS_DIR}/rfdetr_model.onnx -> ${SERVER_DIR}/"
        cp "${MODELS_DIR}/rfdetr_model.onnx" "${SERVER_DIR}/"
    elif [ -f "${SCRIPT_DIR}/../deployment_artifacts/rfdetr_model.onnx" ]; then
        echo "Copying from deployment_artifacts -> ${SERVER_DIR}/"
        cp "${SCRIPT_DIR}/../deployment_artifacts/rfdetr_model.onnx" "${SERVER_DIR}/"
    else
        echo "WARNING: ${SERVER_DIR}/rfdetr_model.onnx not found."
        echo "Please run convert_rfdetr_to_onnx.ipynb first to generate the ONNX model."
        exit 1
    fi
fi

# Check Azure Login Status
if ! az account show &>/dev/null; then
    echo "You are not logged in to Azure CLI. Running 'az login'..."
    az login
fi

ACTIVE_SUB=$(az account show --query "name" -o tsv)
echo "✓ Authenticated to Azure Subscription: ${ACTIVE_SUB}"

# 2. Resource Group
echo "Ensuring Resource Group '${RESOURCE_GROUP}' in '${LOCATION}'..."
az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" -o table

# 3. Azure Container Registry (ACR)
RAND_SUFFIX=$(date +%s | tail -c 6)
ACR_NAME="${ACR_NAME:-acrrfdetr${RAND_SUFFIX}}"
echo "Creating/checking Azure Container Registry '${ACR_NAME}'..."
az acr create --resource-group "${RESOURCE_GROUP}" --name "${ACR_NAME}" --sku Standard --location "${LOCATION}" -o table || true

ACR_LOGIN_SERVER=$(az acr show --name "${ACR_NAME}" --query "loginServer" -o tsv)
echo "✓ ACR Login Server: ${ACR_LOGIN_SERVER}"

# 4. Build Container Image via ACR Cloud Build (Server-side, no local Docker required)
FULL_IMAGE="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "Building container image '${FULL_IMAGE}' in ACR..."
az acr build --registry "${ACR_NAME}" --image "${IMAGE_NAME}:${IMAGE_TAG}" "${SERVER_DIR}"

# 5. Provision AKS Cluster (with ACR attached for native image pulling)
echo "Ensuring AKS Cluster '${AKS_CLUSTER_NAME}'..."
if ! az aks show --resource-group "${RESOURCE_GROUP}" --name "${AKS_CLUSTER_NAME}" &>/dev/null; then
    echo "Creating AKS cluster (this may take 4-7 minutes)..."
    az aks create \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${AKS_CLUSTER_NAME}" \
        --node-count "${NODE_COUNT}" \
        --node-vm-size "${NODE_VM_SIZE}" \
        --attach-acr "${ACR_NAME}" \
        --enable-managed-identity \
        --generate-ssh-keys \
        -o table
else
    echo "✓ AKS cluster '${AKS_CLUSTER_NAME}' already exists. Ensuring ACR is attached..."
    az aks update --resource-group "${RESOURCE_GROUP}" --name "${AKS_CLUSTER_NAME}" --attach-acr "${ACR_NAME}" -o table || true
fi

# 6. Configure kubectl credentials
echo "Configuring kubectl context..."
az aks get-credentials --resource-group "${RESOURCE_GROUP}" --name "${AKS_CLUSTER_NAME}" --overwrite-existing

# 7. Render and Apply Kubernetes Manifests
echo "Rendering deployment manifest with ACR image..."
RENDERED_DEPLOYMENT="${K8S_DIR}/deployment_rendered.yaml"
sed "s|{{ACR_LOGIN_SERVER}}|${ACR_LOGIN_SERVER}|g; s|{{IMAGE_TAG}}|${IMAGE_TAG}|g" "${K8S_DIR}/deployment.yaml" > "${RENDERED_DEPLOYMENT}"

echo "Applying Kubernetes manifests (Internal Service, Zero Public IP)..."
kubectl apply -f "${RENDERED_DEPLOYMENT}"
kubectl apply -f "${K8S_DIR}/service.yaml"
kubectl apply -f "${K8S_DIR}/hpa.yaml"

# 8. Monitor Rollout
echo "Waiting for pod rollout to complete..."
kubectl rollout status deployment/rfdetr-inference-deployment --timeout=300s

echo "✓ Deployment rollout successful!"

# 9. Retrieve Private Internal IP
echo "Querying Internal LoadBalancer Private VNet IP..."
PRIVATE_IP=""
for i in {1..24}; do
    IP=$(kubectl get service rfdetr-inference-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
    if [ -n "$IP" ] && [ "$IP" != "<pending>" ]; then
        PRIVATE_IP="$IP"
        break
    fi
    echo "Waiting for Internal VNet IP... (attempt $i/24)"
    sleep 5
done

echo "=============================================================================="
echo " DEPLOYMENT COMPLETE (NO PUBLIC IP)"
echo "=============================================================================="
if [ -n "$PRIVATE_IP" ]; then
    echo " Private VNet IP:  http://${PRIVATE_IP}"
    echo " Public Exposure:  NONE (Internal LoadBalancer only)"
else
    echo " Private VNet IP:  Pending allocation in VNet subnet"
fi
echo ""
echo " To securely access and test the service from your workstation without VPN:"
echo " Run:"
echo "   ./port_forward.sh"
echo " Then query: http://127.0.0.1:8000/docs or run python test_client.py"
echo "=============================================================================="

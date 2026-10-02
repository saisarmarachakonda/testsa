# RF-DETR Private Deployment on Azure Kubernetes Service (AKS)

This directory contains the complete toolkit to:
1. Convert your trained **RF-DETR** model to optimized **ONNX** format.
2. Deploy a high-throughput **Object Detection microservice** (**RF-DETR**) on **Azure Kubernetes Service (AKS)** behind an **Internal Load Balancer** (**Zero Public IP Exposure**).
3. Connect securely and run end-to-end tag detection via **`kubectl port-forward`**.

---

## Architecture: Multi-Class Detection Pipeline

```
[ Input Image ]
       │
       ▼ (Detection)
[ RF-DETR ONNX / TensorRT ] ──> Detects tags & bounding boxes (blue_aisle, blue_bay, location_tag)
       │
       ▼ (Recalibration & Fusion)
[ Post-Processing ] ─────────> Weighted Box Averaging & Boundary Shrinkage
       │
       ▼
[ Combined Response ] ───────> JSON with bounding box coordinates, class labels & confidence scores
```

---

## Directory Structure

```
deployment/
├── convert_rfdetr_to_onnx.ipynb    # Jupyter Notebook to convert & benchmark ONNX model
├── deploy.sh                        # Bash script for automated Azure ACR + AKS deployment
├── deploy.py                        # Cross-platform Python deployment automation script
├── port_forward.sh                  # Secure port-forwarding tunnel script (port 8001:8001)
├── test_client.py                   # Python CLI client to test inference via port-forwarding
├── server/
│   ├── app.py                      # Production FastAPI service (RF-DETR ONNX)
│   ├── Dockerfile                  # Container definition with ONNX Runtime
│   └── requirements.txt            # Runtime dependencies (FastAPI, ONNX Runtime, etc.)
├── k8s/
│   ├── deployment.yaml             # Kubernetes Deployment template with resource limits & probes
│   ├── service.yaml                # Internal LoadBalancer Service (no public IP)
│   └── hpa.yaml                    # Horizontal Pod Autoscaler (HPA)
├── models/                         # Output folder for converted ONNX model
└── README.md                       # This guide
```

---

## Step-by-Step Workflow

### Step 1: Convert RF-DETR Model to ONNX
Open and run [`convert_rfdetr_to_onnx.ipynb`](convert_rfdetr_to_onnx.ipynb).
- Discovers your trained checkpoint (`.pth`) automatically.
- Exports the model to `deployment/models/rfdetr_model.onnx` with dynamic batch axes and Sigmoid scores.
- Validates the ONNX graph with `onnx.checker` and runs a latency benchmark with ONNX Runtime.
- Copies the model artifact to `deployment/server/rfdetr_model.onnx`.

---

### Step 2: Deploy to Azure Kubernetes Service (AKS)
Ensure Azure CLI and `kubectl` are available, then run:

```bash
# Option A: Bash
./deploy.sh

# Option B: Python (Cross-platform)
python deploy.py
```

**What this script does:**
1. Verifies/performs `az login`.
2. Creates an Azure Resource Group (`rg-rfdetr-inference`) and Azure Container Registry (`acrrfdetr<id>`).
3. Executes `az acr build` to package `server/` into a Docker image directly in Azure (no local Docker needed).
4. Provisions an AKS cluster (`aks-rfdetr-cluster`) with `--attach-acr` enabled.
5. Deploys the rendered Kubernetes manifests.
6. The service is created with `service.beta.kubernetes.io/azure-load-balancer-internal: "true"`, assigning a **private VNet IP** and ensuring **ZERO public IP exposure**.

---

### Step 3: Establish Secure Port-Forwarding

Connect to the enterprise AKS cluster and establish port-forward tunnels:

```bash
# 1. Set Subscription & Cluster Credentials
export AZURE_SUBSCRIPTION="<YOUR_AZURE_SUBSCRIPTION>"
export AZURE_RESOURCE_GROUP="<YOUR_RESOURCE_GROUP>"
export AKS_CLUSTER_NAME="<YOUR_AKS_CLUSTER_NAME>"
export K8S_NAMESPACE="<YOUR_NAMESPACE>"

az account set --subscription "$AZURE_SUBSCRIPTION"
az aks get-credentials --resource-group "$AZURE_RESOURCE_GROUP" --name "$AKS_CLUSTER_NAME"
kubectl config set-context --current --namespace="$K8S_NAMESPACE"

# 2. Inspect Running Pods
kubectl get pods -n "$K8S_NAMESPACE"

# 3. Port Forward Detection Pod
kubectl port-forward <DETECTION_POD_NAME> 8001:8001 -n "$K8S_NAMESPACE"
```

Or execute the all-in-one script which auto-discovers pod IDs and manages background tunnels:
```bash
./port_forward.sh
```

---

### Step 4: Launch Unified Gateway Server

On a new terminal:
```bash
cd multi_class_pipeline/deployment
python3 app.py
```

The gateway server:
- Runs locally on **port 8000** (`http://127.0.0.1:8000`)
- Routes detection queries to **`http://127.0.0.1:8001`** (TensorRT V100 Detection Pod)
- Performs **Weighted Box Fusion (Averaging)** and **Boundary Box Shrinkage (4% insetting)** to tighten query boundaries.

Interactive endpoints:
- **Swagger Docs**: `http://127.0.0.1:8000/docs`
- **Health Check**: `http://127.0.0.1:8000/healthz`
- **Predict**: `POST http://127.0.0.1:8000/predict`

---

### Step 5: Run Predictions via Test Client
In another terminal, run:

```bash
# Test with default image / synthetic image (runs Detection)
python test_client.py

# Test with a specific image
python test_client.py --image path/to/sample.jpg --conf 0.20
```

The client will:
- Check `/healthz` and verify RF-DETR is active.
- Send the image to `POST /predict`.
- Print detections, confidence scores, and latency.
- Save a visualization image with bounding boxes and labels to `prediction_result.jpg`.

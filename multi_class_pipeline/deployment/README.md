# RF-DETR + PaddleOCR Private Deployment on Azure Kubernetes Service (AKS)

This directory contains the complete toolkit to:
1. Convert your trained **RF-DETR** model to optimized **ONNX** format.
2. Deploy a **two-stage Object Detection & OCR microservice** (**RF-DETR + PaddleOCR**) on **Azure Kubernetes Service (AKS)** behind an **Internal Load Balancer** (**Zero Public IP Exposure**).
3. Connect securely and run end-to-end tag detection and text recognition via **`kubectl port-forward`**.

---

## Architecture: Detection + Text Recognition Pipeline

```
[ Input Image ]
       │
       ▼ (Stage 1: Detection)
[ RF-DETR ONNX ] ──> Detects tags & bounding boxes (e.g. location_tag, blue_aisle, blue_bay)
       │
       ▼ (Stage 2: Text Recognition)
[ PaddleOCR ] ─────> Crops detected bounding boxes & extracts alphanumeric text (e.g. "LOC-A-12-04")
       │
       ▼
[ Combined Response ] ──> JSON with bounding box coordinates, class labels & OCR text
```

---

## Directory Structure

```
deployment/
├── convert_rfdetr_to_onnx.ipynb    # Jupyter Notebook to convert & benchmark ONNX model
├── deploy.sh                        # Bash script for automated Azure ACR + AKS deployment
├── deploy.py                        # Cross-platform Python deployment automation script
├── port_forward.sh                  # Secure port-forwarding tunnel script (port 8000 -> 80)
├── test_client.py                   # Python CLI client to test inference & OCR via port-forwarding
├── server/
│   ├── app.py                      # Production FastAPI service (RF-DETR ONNX + PaddleOCR)
│   ├── Dockerfile                  # Container definition with pre-downloaded PaddleOCR models
│   └── requirements.txt            # Runtime dependencies (FastAPI, ONNX Runtime, PaddleOCR, etc.)
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
3. Executes `az acr build` to package `server/` into a Docker image directly in Azure (no local Docker needed). PaddleOCR models are cached into the image layer during build time.
4. Provisions an AKS cluster (`aks-rfdetr-cluster`) with `--attach-acr` enabled.
5. Deploys the rendered Kubernetes manifests.
6. The service is created with `service.beta.kubernetes.io/azure-load-balancer-internal: "true"`, assigning a **private VNet IP** and ensuring **ZERO public IP exposure**.

---

### Step 3: Establish Secure Port-Forwarding
To interact with the private AKS service from your local workstation:

```bash
./port_forward.sh
```

This establishes an encrypted tunnel over the authenticated Kubernetes control plane:
- **Local Address**: `http://127.0.0.1:8000`
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **Health Check**: `http://127.0.0.1:8000/healthz`
- **Predict (Detection + OCR)**: `POST http://127.0.0.1:8000/predict?run_ocr=true`
- **Direct OCR**: `POST http://127.0.0.1:8000/ocr`

---

### Step 4: Run Predictions via Test Client
In another terminal, run:

```bash
# Test with default image / synthetic image (runs Detection + PaddleOCR)
python test_client.py

# Test with a specific image
python test_client.py --image path/to/sample.jpg --conf 0.20

# Test detection only (disable OCR)
python test_client.py --image path/to/sample.jpg --no-ocr
```

The client will:
- Check `/healthz` and verify both RF-DETR and PaddleOCR are active.
- Send the image to `POST /predict`.
- Print detections, OCR recognized strings (e.g. `LOC-A-12-04`), confidence scores, and latency.
- Save a visualization image with bounding boxes and recognized text labels to `prediction_result.jpg`.

# RF-DETR Multi-Class Detection Pipeline (`multi_class_pipeline`)

This folder contains the complete, unified multi-class machine learning lifecycle for RF-DETR: **Exploratory Data Analysis**, **Anomaly Auditing**, **80/10/10 Stratification**, **Metric Precision-Driven Training**, **Post-Training Prediction with PaddleOCR**, **Clean Test Evaluation**, **Anomaly Robustness Audit**, and **Azure Kubernetes Service (AKS) Deployment**.

All notebooks are organized in sequential order (01 through 09) within this single folder.

---

## Notebooks (Sequential Execution)

| Step | Notebook | Description | Key Outputs / Artifacts |
|---|---|---|---|
| **00** | [`00_download_coco_files.ipynb`](00_download_coco_files.ipynb) | Multi-source dataset ingestion (Azure Blob, HTTP/Zip, Roboflow API, or local discovery) with cache validation and zero-copy image pooling | `coco_files/`, raw COCO annotations & images |
| **01** | [`01_eda_annotations.ipynb`](01_eda_annotations.ipynb) | Exploratory Data Analysis of bounding boxes, aspect ratios, and category distributions | Statistical profiles & class distributions |
| **02** | [`02_detect_and_audit_anomalies.ipynb`](02_detect_and_audit_anomalies.ipynb) | 8-rule automated anomaly detector with **4-panel diagnostic graphs** & **visual sample overlays** (red anomaly boxes vs green clean tags) | `anomalies_manifest.json`, `cleaned_dataset/` |
| **03** | [`03_stratified_train_val_test_split.ipynb`](03_stratified_train_val_test_split.ipynb) | Iterative multi-label greedy stratification into **80% Train**, **10% Val**, and **10% Test** (strictly excluding anomalies) | `dataset_stratified/` |
| **04** | [`04_train_rfdetr_clean_data.ipynb`](04_train_rfdetr_clean_data.ipynb) | Precision-driven training on clean data with **robust training resumption**, auto-epoch extension, Step A loss breakdown, and Step C frozen backbone | `model/rfdetr_best_precision.pth`, `checkpoints/` |
| **05** | [`05_postprocessing_techniques.ipynb`](05_postprocessing_techniques.ipynb) | Dedicated postprocessing suite: **Boundary-box shrinkage (4%)**, **Weighted Box Fusion (WBF)**, per-class confidence tuning & PaddleOCR | Recalibrated bounding boxes & OCR tags |
| **06** | [`06_predict_rfdetr_images.ipynb`](06_predict_rfdetr_images.ipynb) | Post-training batch inference with **sample visual predictions gallery** (color-coded boxes, confidence scores, OCR text badges) | `predictions/` |
| **07** | [`07_evaluate_rfdetr_clean_data.ipynb`](07_evaluate_rfdetr_clean_data.ipynb) | Benchmark evaluation on held-out clean test set: COCO standard mAP@50:95, multi-threshold sweeps, confusion matrix, and 100% FP/FN case inspector | `evaluation_results/` |
| **08** | [`08_evaluate_anomalies_audit.ipynb`](08_evaluate_anomalies_audit.ipynb) | Robustness audit specifically on flagged anomaly images to evaluate model behavior against labeler noise | Anomaly audit reports & case visualizer |
| **09** | [`09_convert_rfdetr_to_onnx.ipynb`](09_convert_rfdetr_to_onnx.ipynb) | Standalone conversion of fine-tuned RF-DETR model to ONNX runtime format + verification with PaddleOCR | `deployment/models/rfdetr_model.onnx` |
| **10** | [`10_evaluate_compare_rfdetr_multiclass.ipynb`](10_evaluate_compare_rfdetr_multiclass.ipynb) | 3-way evaluation benchmark: **Ground Truth vs. Old Model Endpoint vs. New RF-DETR Model** with 3-panel visualizer and metrics table | Head-to-head comparison metrics & showcase |
| **11** | [`11_convert_rfdetr_to_onnx_and_deploy_aks.ipynb`](11_convert_rfdetr_to_onnx_and_deploy_aks.ipynb) | Interactive end-to-end cloud deployment notebook: ONNX export + ACR build + AKS deployment (Internal LoadBalancer, Zero Public IP) | Live AKS Deployment Manifests & Tests |

---

## Directory Organization

```
multi_class_pipeline/
├── 00_download_coco_files.ipynb                   # [Step 00 Notebook] Dataset download & ingestion
├── 01_eda_annotations.ipynb                       # [Step 01 Notebook] Exploratory Data Analysis
├── 02_detect_and_audit_anomalies.ipynb            # [Step 02 Notebook] 8-rule anomaly detector & graphs
├── 03_stratified_train_val_test_split.ipynb       # [Step 03 Notebook] Multi-label 80/10/10 stratified split
├── 04_train_rfdetr_clean_data.ipynb               # [Step 04 Notebook] Metric precision training & resumption
├── 05_postprocessing_techniques.ipynb             # [Step 05 Notebook] Boundary shrinkage, WBF & OCR
├── 06_predict_rfdetr_images.ipynb                 # [Step 06 Notebook] Inference + Sample Predictions Gallery
├── 07_evaluate_rfdetr_clean_data.ipynb            # [Step 07 Notebook] Benchmark evaluation on clean test set
├── 08_evaluate_anomalies_audit.ipynb              # [Step 08 Notebook] Anomaly robustness & labeler error audit
├── 09_convert_rfdetr_to_onnx.ipynb                # [Step 09 Notebook] Standalone ONNX model export & test
├── 10_evaluate_compare_rfdetr_multiclass.ipynb    # [Step 10 Notebook] Ground Truth vs Old vs New Model
├── 11_convert_rfdetr_to_onnx_and_deploy_aks.ipynb # [Step 11 Notebook] Interactive AKS deployment notebook
├── README.md                                      # This guide
└── deployment/                                    # Production AKS microservice & deployment scripts
    ├── server/                                    # FastAPI + PaddleOCR + Dockerfile
    ├── k8s/                                       # AKS manifests (Internal LoadBalancer, Zero Public IP)
    ├── deploy.sh / deploy.py                      # Automated deployment scripts
    ├── port_forward.sh                            # Secure port forwarding (8000 -> 80)
    ├── test_client.py                             # CLI test client
    └── models/                                    # ONNX runtime model files
```

---

---

## Engineering Features

### 1. Model Versioning & Central Registry
- **Semantic Versioning (`MODEL_VERSION`)**: Configure training versions via environment variable `export MODEL_VERSION="v1.2.0"` or directly in [`04_train_rfdetr_clean_data.ipynb`](04_train_rfdetr_clean_data.ipynb) (e.g. `v1.1.0-precision-clean`).
- **Version-Isolated Checkpoints**: Intermediate checkpoints are safely isolated in `checkpoints/<MODEL_VERSION>/` so subsequent runs never overwrite prior checkpoint history. A dynamic `checkpoints/latest` pointer is automatically maintained.
- **Version-Tagged Exports**: Best weights are saved as `model/rfdetr_best_<VERSION>.pth` and `model/rfdetr_best_precision_<VERSION>.pth`, alongside canonical aliases (`best_model.pth`, `best_model_precision.pth`).
- **Central Model Registry (`model_registry.json`)**: Automatically records all trained model versions, timestamps, hyperparameters, best epochs, and evaluation metrics (Precision, Recall, F1, mAP50, mAP50-95).
- **Downstream Auto-Discovery**: Prediction ([`05`](05_predict_rfdetr_images.ipynb)), Evaluation ([`06`](06_evaluate_rfdetr_clean_data.ipynb)), and ONNX Export ([`08`](08_convert_rfdetr_to_onnx.ipynb) / [`09`](09_convert_rfdetr_to_onnx_and_deploy_aks.ipynb)) automatically discover the active version from `model_registry.json` or honor `MODEL_VERSION`.

### 2. Skip Download If Images Exist (Cache Validation)
- All image acquisition logic performs an instant pre-flight check:
  ```python
  if dest_path.exists() and dest_path.stat().st_size > 0:
      return dest_path  # Skips download immediately
  ```
- Re-running notebooks or resuming training completely bypasses network downloads for any image already on disk.

### 3. Zero-Copy Architecture (Save Disk Space)
- **Single Shared Master Image Storage**: Images reside in a single central master repository (`coco_files/images` or `dataset_stratified/images`).
- **Zero Redundant Copying**: Dataset splits (`train/`, `val/`, `test/`) store only lightweight COCO JSON metadata and soft symbolic links (`train/images -> ../images`).
- **Multi-Path DataLoader**: `COCODetectionDataset` dynamically resolves images directly from the shared pool, eliminating duplicate image copies across training and test splits.

---

## Runtime Artifacts Generated During Execution

When you run the notebooks in sequence, they will generate clean output folders directly in `multi_class_pipeline/`:
- `coco_files/` (from Step 00) — Ingested COCO JSON files & centralized raw image storage
- `anomalies_manifest.json` & `cleaned_dataset/` (from Step 02)
- `dataset_stratified/` (from Step 03) — Zero-copy architecture with master image repository
- `checkpoints/<MODEL_VERSION>/` & `model/model_registry.json` (from Step 04) — Version-isolated models & checkpoints
- `predictions/` (from Step 05)
- `evaluation_results/` (from Step 06)
- `deployment/models/rfdetr_model.onnx` (from Step 08)


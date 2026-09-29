# RF-DETR Multi-Class Detection Pipeline (`multi_class_pipeline`)

This folder contains the complete, unified multi-class machine learning lifecycle for RF-DETR: **Exploratory Data Analysis**, **Anomaly Auditing**, **80/10/10 Stratification**, **Metric Precision-Driven Training**, **Post-Training Prediction with PaddleOCR**, **Clean Test Evaluation**, **Anomaly Robustness Audit**, and **Azure Kubernetes Service (AKS) Deployment**.

All notebooks are organized in sequential order (01 through 09) within this single folder.

---

## Notebooks (Sequential Execution)

| Step | Notebook | Description | Key Outputs / Artifacts |
|---|---|---|---|
| **01** | [`01_eda_annotations.ipynb`](01_eda_annotations.ipynb) | Exploratory Data Analysis of bounding boxes, aspect ratios, and category distributions | Statistical profiles & class distributions |
| **02** | [`02_detect_and_audit_anomalies.ipynb`](02_detect_and_audit_anomalies.ipynb) | 8-rule automated anomaly detector (degenerate boxes, micro-clicks, giant boxes, IoU duplicates, etc.) | `anomalies_manifest.json`, `cleaned_dataset/` |
| **03** | [`03_stratified_train_val_test_split.ipynb`](03_stratified_train_val_test_split.ipynb) | Iterative multi-label greedy stratification into **80% Train**, **10% Val**, and **10% Test** (strictly excluding anomalies) | `dataset_stratified/` |
| **04** | [`04_train_rfdetr_clean_data.ipynb`](04_train_rfdetr_clean_data.ipynb) | Precision-driven training on clean data with Step A loss breakdown, Step B label verification, Step C frozen DINOv2 backbone, and boundary recalibration | `model/rfdetr_best_precision.pth`, `checkpoints/` |
| **05** | [`05_predict_rfdetr_images.ipynb`](05_predict_rfdetr_images.ipynb) | Post-training inference: Bounding-box averaging + Boundary-box shrinkage + **PaddleOCR** tag recognition | `predictions/` |
| **06** | [`06_evaluate_rfdetr_clean_data.ipynb`](06_evaluate_rfdetr_clean_data.ipynb) | Benchmark evaluation on held-out clean test set: COCO standard mAP@50:95, multi-threshold sweeps, confusion matrix, 6-panel dashboard, and 100% FP/FN case inspector | `evaluation_results/` |
| **07** | [`07_evaluate_anomalies_audit.ipynb`](07_evaluate_anomalies_audit.ipynb) | Robustness audit specifically on flagged anomaly images to evaluate model behavior against labeler noise | Anomaly audit reports & case visualizer |
| **08** | [`08_convert_rfdetr_to_onnx.ipynb`](08_convert_rfdetr_to_onnx.ipynb) | Standalone conversion of fine-tuned RF-DETR model to ONNX runtime format + verification with PaddleOCR | `deployment/models/rfdetr_model.onnx` |
| **09** | [`09_convert_rfdetr_to_onnx_and_deploy_aks.ipynb`](09_convert_rfdetr_to_onnx_and_deploy_aks.ipynb) | Interactive end-to-end cloud deployment notebook: ONNX export + ACR build + AKS deployment (Internal LoadBalancer, Zero Public IP) | Live AKS Deployment Manifests & Tests |

---

## Directory Organization

```
multi_class_pipeline/
├── 01_eda_annotations.ipynb                       # [Step 1 Notebook] Exploratory Data Analysis
├── 02_detect_and_audit_anomalies.ipynb            # [Step 2 Notebook] 8-rule anomaly detector & audit
├── 03_stratified_train_val_test_split.ipynb       # [Step 3 Notebook] Multi-label 80/10/10 stratified split
├── 04_train_rfdetr_clean_data.ipynb               # [Step 4 Notebook] Metric precision training (clean data)
├── 05_predict_rfdetr_images.ipynb                 # [Step 5 Notebook] Inference + Recalibration + PaddleOCR
├── 06_evaluate_rfdetr_clean_data.ipynb            # [Step 6 Notebook] Benchmark evaluation on clean test set
├── 07_evaluate_anomalies_audit.ipynb              # [Step 7 Notebook] Anomaly robustness & labeler error audit
├── 08_convert_rfdetr_to_onnx.ipynb                # [Step 8 Notebook] Standalone ONNX model export & test
├── 09_convert_rfdetr_to_onnx_and_deploy_aks.ipynb # [Step 9 Notebook] Interactive AKS deployment notebook
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

## Runtime Artifacts Generated During Execution

When you run the notebooks in sequence, they will generate clean output folders directly in `multi_class_pipeline/`:
- `anomalies_manifest.json` & `cleaned_dataset/` (from Step 02)
- `dataset_stratified/` (from Step 03)
- `checkpoints/` & `model/` (from Step 04)
- `predictions/` (from Step 05)
- `evaluation_results/` (from Step 06)
- `deployment/models/rfdetr_model.onnx` (from Step 08)

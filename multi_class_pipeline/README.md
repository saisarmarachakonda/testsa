# RF-DETR Multi-Class Detection Pipeline (`multi_class_pipeline`)

This folder contains the complete, end-to-end multi-class machine learning lifecycle for RF-DETR: **Exploratory Data Analysis**, **Anomaly Auditing**, **80/10/10 Stratification**, **Metric Precision-Driven Training**, **Post-Training Prediction with PaddleOCR**, **Clean Test Evaluation**, **Anomaly Robustness Audit**, and **Azure Kubernetes Service (AKS) Deployment**.

---

## Notebooks (Sequential Execution)

All pipeline notebooks are directly accessible in this folder and can be run sequentially:

| Step | Notebook | Description | Key Outputs / Artifacts |
|---|---|---|---|
| **01** | [`01_eda_annotations.ipynb`](01_eda_annotations.ipynb) | Exploratory Data Analysis of bounding boxes, aspect ratios, and category distributions | Statistical profiles & distributions |
| **02** | [`02_detect_and_audit_anomalies.ipynb`](02_detect_and_audit_anomalies.ipynb) | 8-rule automated anomaly detector (degenerate boxes, micro-clicks, giant boxes, IoU duplicates, etc.) | `02_anomalies/anomalies_manifest.json` |
| **03** | [`03_stratified_train_val_test_split.ipynb`](03_stratified_train_val_test_split.ipynb) | Iterative multi-label greedy stratification into **80% Train**, **10% Val**, and **10% Test** (strictly excluding anomalies) | `03_dataset_split/dataset_stratified/` |
| **04** | [`04_train_rfdetr_clean_data.ipynb`](04_train_rfdetr_clean_data.ipynb) | Precision-driven training on clean data with Step A loss breakdown, Step B label verification, Step C frozen DINOv2 backbone, and boundary recalibration | `04_training/model/rfdetr_best_precision.pth` |
| **05** | [`05_predict_rfdetr_images.ipynb`](05_predict_rfdetr_images.ipynb) | Post-training inference: Bounding-box averaging + Boundary-box shrinkage + **PaddleOCR** tag recognition | `05_prediction/predictions/` |
| **06** | [`06_evaluate_rfdetr_clean_data.ipynb`](06_evaluate_rfdetr_clean_data.ipynb) | Benchmark evaluation on held-out clean test set: COCO standard mAP@50:95, multi-threshold sweeps, confusion matrix, 6-panel dashboard, and 100% FP/FN case inspector | `06_evaluation/evaluation_results/` |
| **07** | [`07_evaluate_anomalies_audit.ipynb`](07_evaluate_anomalies_audit.ipynb) | Robustness audit specifically on flagged anomaly images to evaluate model behavior against labeler noise | `07_evaluation_anomalies/` audit logs |
| **08** | [`08_convert_rfdetr_to_onnx.ipynb`](08_convert_rfdetr_to_onnx.ipynb) | Standalone conversion of fine-tuned RF-DETR model to ONNX runtime format + verification with PaddleOCR | `08_deployment/models/rfdetr_model.onnx` |
| **09** | [`09_convert_rfdetr_to_onnx_and_deploy_aks.ipynb`](09_convert_rfdetr_to_onnx_and_deploy_aks.ipynb) | Interactive end-to-end cloud deployment notebook: ONNX export + ACR build + AKS deployment (Internal LoadBalancer, Zero Public IP) | AKS Deployment Manifests & Tests |

---

## Directory Organization

The underlying stage folders house runtime data, models, logs, and deployment assets:

```
multi_class_pipeline/
├── 01_eda_annotations.ipynb                       # [Step 1 Notebook]
├── 02_detect_and_audit_anomalies.ipynb            # [Step 2 Notebook]
├── 03_stratified_train_val_test_split.ipynb       # [Step 3 Notebook]
├── 04_train_rfdetr_clean_data.ipynb               # [Step 4 Notebook]
├── 05_predict_rfdetr_images.ipynb                 # [Step 5 Notebook]
├── 06_evaluate_rfdetr_clean_data.ipynb            # [Step 6 Notebook]
├── 07_evaluate_anomalies_audit.ipynb              # [Step 7 Notebook]
├── 08_convert_rfdetr_to_onnx.ipynb                # [Step 8 Notebook]
├── 09_convert_rfdetr_to_onnx_and_deploy_aks.ipynb # [Step 9 Notebook]
├── 01_eda/                                        # EDA scripts and profiles
├── 02_anomalies/                                  # Anomaly detection results & manifest
├── 03_dataset_split/                              # Stratified 80/10/10 dataset splits
├── 04_training/                                   # Checkpoints, logs, and exported models
├── 05_prediction/                                 # Prediction outputs & annotated image crops
├── 06_evaluation/                                 # Evaluation charts, CSVs, and FP/FN plots
├── 07_evaluation_anomalies/                       # Anomaly audit logs and comparisons
├── 08_deployment/ (alias: deployment/)            # Production deployment files
│   ├── server/                                    # FastAPI + PaddleOCR + Dockerfile
│   ├── k8s/                                       # AKS manifests (Internal LB, no public IP)
│   ├── deploy.sh / deploy.py                      # Automated deployment scripts
│   ├── port_forward.sh                            # Secure port forwarding (8000 -> 80)
│   ├── test_client.py                             # CLI test client
│   └── models/                                    # ONNX runtime model files
└── README.md                                      # This guide
```

---

## Key Features

1. **Zero-Anomaly Policy**: Anomalies audited in `02_anomalies/` are strictly excluded from dataset splitting, clean training, and clean evaluation.
2. **Metric Precision-Driven Training**: Tracks and trains for operational **Precision** ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) at each validation epoch, saving `rfdetr_best_precision.pth`.
3. **Bounding-Box Recalibration**:
   - **Boundary-Box Shrinkage**: Contracts loose query margins by 4% (`shrink_factor=0.96`), tightening crops for PaddleOCR and eliminating borderline near-misses.
   - **Bounding-Box Averaging**: Fuses overlapping multi-query detections ($\text{IoU} \ge 0.50$) via confidence-weighted coordinate averaging.
4. **Two-Stage Detection + PaddleOCR**: Detects tags and recognizes alphanumeric text inside each bounding box.
5. **Private AKS Deployment**: Internal LoadBalancer with **Zero Public IP**, port forwarding via `port_forward.sh`, and sub-millisecond ONNX Runtime CPU/GPU inference.

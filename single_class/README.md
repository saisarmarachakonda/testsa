# Single Class RF-DETR Detection Project

This project trains an **RF-DETR Base** model configured for **1 class only** (`TARGET_CLASS = 'object'`), training on all annotated images.

## Features
- **Early Visual EDA**: Inspects ground-truth bounding boxes colored by their original multi-categories before training.
- **1-Class Remapping**: Unifies all categories across all images into class `0`.
- **Optimizations**:
  - Resolution: `560`
  - Optimizer: `Adam` (Smaller LR: `5e-5`, Weight Decay: `1e-4`)
  - LR Scheduler: `Cosine decay`
  - PyTorch CUDA: `cudnn.benchmark = True`, `set_float32_matmul_precision('high')`
  - Early Stopping: `early_stopping = True`, `patience = 5`, `min_delta = 0.001`
  - Inference: `model.optimize_for_inference()`, Default confidence threshold `0.50`, NMS post-processing (`0.50`)
  - Logging: File and console logger (`pipeline.log`)

## Notebooks
1. `RF_DETR_Single_Class_Sample_1000.ipynb`: Pre-configured for a **1,000-image sample** to test and verify the entire pipeline flow quickly.
2. `RF_DETR_Single_Class_Full_Data.ipynb`: Pre-configured for **Full Data** production training across all images.

# Multi-Class RF-DETR Detection Project

This project trains an **RF-DETR Base** model configured for **multi-category detection** (`location_tag`, `blue_aisle`, etc.).

## Features
- **Auto-Discovery**: Automatically extracts and maps all categories to 0-indexed IDs (`0, 1, ...`).
- **Category Proportions Stratification**: Guarantees equal proportions of every category across `train`, `val`, and `test` splits.
- **Optimizations**:
  - Resolution: `560`
  - Optimizer: `Adam` (Smaller LR: `5e-5`, Weight Decay: `1e-4`)
  - LR Scheduler: `Cosine decay`
  - Safe T4 Batch Size: `2`
  - Dataloader Workers: `2`
  - Inference: `model.optimize_for_inference()`, Default confidence threshold `0.50`, NMS post-processing (`0.50`)

## Notebooks
1. `RF_DETR_Multi_Class_Sample_1000.ipynb`: Pre-configured for a **1,000-image stratified sample** (preserving category ratios) to verify pipeline flow quickly.
2. `RF_DETR_Multi_Class_Full_Data.ipynb`: Pre-configured for **Full Data** production training across all images.

# multi_class_train_rfdetr

This folder contains the dedicated multi-class RF-DETR pipeline.

## Notebook: `multi_class_train_rfdetr.ipynb`
- **Model**: `RFDETRBase` (RF-DETR Base)
- **Multi-Class Discovery**: Automatically discovers categories from `coco_files/*.json`
- **Category Stratification**: Equal category proportions maintained across Train (80%), Val (10%), and Test (10%) splits
- **Resolution**: `560`
- **Optimizer & Schedule**: `Adam` (lr=5e-5, weight_decay=1e-4) with `cosine` decay
- **Dual Logging**: Informative `print()` output in cells + `output_*/pipeline.log`
- **Mode Toggle (Cell 2)**: `SAMPLE_SIZE = 1000` (test flow) vs `SAMPLE_SIZE = None` (full data)
- **Dynamic Folders**: Automatically generates `dataset_*`, `output_*`, `inference_*`, and `model/` inside this directory

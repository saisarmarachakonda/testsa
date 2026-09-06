# location_tag_single_class_train_rfdetr

This folder contains the dedicated single-class RF-DETR pipeline focusing on `location_tag`.

## Notebook: `location_tag_single_class_train_rfdetr.ipynb`
- **Model**: `RFDETRBase` (RF-DETR Base)
- **Target Category**: `location_tag` (All annotations mapped to class 0)
- **Resolution**: `560`
- **Optimizer & Schedule**: `Adam` (lr=5e-5, weight_decay=1e-4) with `cosine` decay
- **Dual Logging**: Informative `print()` output in cells + `output_*/pipeline.log`
- **Mode Toggle (Cell 2)**: `SAMPLE_SIZE = 1000` (test flow) vs `SAMPLE_SIZE = None` (full data)
- **Dynamic Folders**: Automatically generates `dataset_*`, `output_*`, `inference_*`, and `model/` inside this directory

# location_tag_single_class_train_rfdetr/

Output directory for the single-class `location_tag` pipeline.
Running `location_tag_single_class_train_rfdetr.ipynb` from the repository root generates:
- `dataset_sample_1000/` or `dataset_full_data/`: Train, Val, and Test split images and `_annotations.coco.json`.
- `output_sample_1000/` or `output_full_data/`: Training checkpoints and `pipeline.log`.
- `inference_sample_1000/` or `inference_full_data/`: Test prediction JSON files.
- `model/`: Exported best model checkpoint (`best_model_<mode>.pth`).

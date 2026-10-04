# multi_class_rfdetr_v2

Unified Pipeline Version 2 Output Directory for RF-DETR Multi-Class Training.

## Structure
- `images/`: Shared pool of downloaded / verified dataset images
- `dataset/`:
  - `train/`: Training split (`_annotations.coco.json` and zero-copy images link)
  - `val/`: Validation split (`_annotations.coco.json` and zero-copy images link)
  - `test/`: Test split (`_annotations.coco.json` and zero-copy images link)
- `cleaned_dataset/`: Sanitized COCO annotations from Step 02 anomaly audit
- `checkpoints/`: Model checkpoints (`latest_checkpoint.pth`, `best_model.pth`)
- `model/`: Exported production models (`best_model.pth`, `best_model_precision.pth`, `rfdetr_multiclass.onnx`)
- `runs/logs/`: Pipeline logs and TensorBoard telemetry
- `anomalies_audit/`: Anomaly audit manifests and sample logs
- `inference_results/`: Inference visual predictions and previews

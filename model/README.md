# RF-DETR Model Checkpoints & Architecture

## Architecture Overview
- **Backbone**: DINOv2 Vision Transformer
- **Detector**: Real-time Detection Transformer (RF-DETR Base)
- **Input Resolution**: `560 x 560`
- **Pretrained Weights**: `/home/jupyter/rf-detr-base-coco.pth` or official Roboflow weights.

## Checkpoint Selection
During training, PyTorch Lightning and RF-DETR output checkpoints to `./rfdetr_output/`:
- `checkpoint_best_regular.pth`: Best regular validation checkpoint.
- `checkpoint_best_total.pth`: Best total metric checkpoint.
- `last.ckpt` / `checkpoint.pth`: Latest checkpoint for resuming.

## Inference Optimization
After loading checkpoints for deployment, call:
```python
model = RFDETRBase(pretrain_weights="path/to/best_checkpoint.pth", resolution=560, num_classes=NUM_CLASSES)
model.optimize_for_inference()
detections = model.predict(image, threshold=0.50)
```

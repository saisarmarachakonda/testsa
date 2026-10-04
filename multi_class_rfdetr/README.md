# multi_class_rfdetr

Unified Pipeline Directory for RF-DETR Multi-Class Detection.

## Architecture

```
multi_class_rfdetr/
├── images/                      <-- Shared image pool (train, val, test all draw from here)
├── dataset/                     <-- Fixed Train/Val/Test Split (consistent across model versions)
│   ├── train/                   <-- _annotations.coco.json
│   ├── val/                     <-- _annotations.coco.json
│   └── test/                    <-- _annotations.coco.json
└── v2/                          <-- Version-Specific Artifacts (v2, v3, etc.)
    ├── checkpoints/             <-- Training checkpoints (latest, best)
    ├── model/                   <-- Final exported weights & ONNX
    ├── runs/logs/               <-- Training logs & TensorBoard telemetry
    ├── anomalies_audit/         <-- Version-specific anomaly audit manifests & reports
    ├── postprocessing/          <-- Postprocessing configurations & evaluations
    └── inference_results/       <-- Prediction previews & visual evaluations
```

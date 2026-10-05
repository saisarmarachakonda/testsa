#!/usr/bin/env python3
"""
================================================================================
RF-DETR Staging Deployment & Inference Verification Test Script
================================================================================
Test against staging environment supporting:
  1. Direct Triton gRPC Endpoint (AKS / local port-forward)
  2. Local ONNX Runtime model (deployment/models/rfdetr_model.onnx)
  3. Local TensorRT Plan model (deployment/models/model.plan)

Features:
  • Tests health / liveness and model readiness.
  • Runs end-to-end inference with 0.75 confidence threshold and multi-stage post-processing:
      - WBF (Weighted Box Fusion) query consolidation
      - Placard Span Fragment Merging (e.g. 'D' + '17' -> 'D17')
      - Placard Context Margin Recalibration ('C1' text-only crop expansion)
      - Geometric Hallucination & Edge Crop guards (removes 'EXIT' signs & specks)
  • Computes latency benchmarks (P50, P95, P99).
  • Saves annotated visual bounding box outputs.

Usage Examples:
  # Test against Staging Triton Pod via port-forward:
  python test_staging_inference.py --backend triton --triton-url localhost:8001 --model-name tagdet_rt --model-version 2

  # Test local ONNX model:
  python test_staging_inference.py --backend onnx --model-path deployment/models/rfdetr_model.onnx --image path/to/sample.jpg

  # Test local TensorRT Engine plan:
  python test_staging_inference.py --backend plan --model-path deployment/models/model.plan --image path/to/sample.jpg
================================================================================
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure current and parent deployment modules are discoverable
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from rfdetr_inference_pipeline import (
        load_rfdetr_detector,
        preprocess_image,
        postprocess_output,
        CLASS_NAMES,
        INPUT_WIDTH,
        INPUT_HEIGHT,
        TOTAL_OUTPUT_ROWS,
        OUTPUT_COLS
    )
except ImportError:
    # Look in deployment folder
    sys.path.append(str(Path(__file__).resolve().parent / "notebook" / "deployment"))
    from rfdetr_inference_pipeline import (
        load_rfdetr_detector,
        preprocess_image,
        postprocess_output,
        CLASS_NAMES,
        INPUT_WIDTH,
        INPUT_HEIGHT,
        TOTAL_OUTPUT_ROWS,
        OUTPUT_COLS
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StagingVerification")


def parse_args():
    parser = argparse.ArgumentParser(description="RF-DETR Staging Verification Test Script")
    parser.add_argument("--backend", default="triton", choices=["triton", "onnx", "plan"],
                        help="Backend to test: 'triton' (staging server), 'onnx', or 'plan'")
    parser.add_argument("--triton-url", default="localhost:8001",
                        help="Triton gRPC endpoint (default: localhost:8001)")
    parser.add_argument("--model-name", default="tagdet_rt",
                        help="Triton model name (default: tagdet_rt)")
    parser.add_argument("--model-version", default="2",
                        help="Model version to test (default: 2)")
    parser.add_argument("--model-path", default="deployment/models/rfdetr_model.onnx",
                        help="Path to local .onnx or .plan file if testing locally")
    parser.add_argument("--image", default=None,
                        help="Path to input test image or folder of images")
    parser.add_argument("--conf", type=float, default=0.75,
                        help="Confidence threshold (default: 0.75)")
    parser.add_argument("--output-dir", default="staging_test_results",
                        help="Directory to save visual test outputs")
    parser.add_argument("--warmup-runs", type=int, default=3,
                        help="Number of warmup inference passes")
    parser.add_argument("--benchmark-runs", type=int, default=10,
                        help="Number of latency benchmark iterations")
    return parser.parse_args()


def draw_bounding_boxes(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    output_path: Path
):
    """Draws color-coded bounding boxes and detection summaries on image."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # High-contrast color mapping per category
    color_map = {
        "location_tag": (255, 61, 0),    # Bright Red-Orange
        "blue_aisle":   (0, 229, 255),   # Bright Cyan
        "blue_bay":     (0, 230, 118)    # Bright Green
    }

    for det in detections:
        box = det["bbox"] # [ymin, xmin, ymax, xmax]
        score = det["score"]
        label = det["description"]
        color = color_map.get(label, (255, 255, 0))

        ymin, xmin, ymax, xmax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        caption = f"{label} {score:.2f}"
        # Text background badge
        text_bbox = draw.textbbox((xmin + 4, max(0, ymin - 16)), caption)
        draw.rectangle(text_bbox, fill=color)
        draw.text((xmin + 4, max(0, ymin - 16)), caption, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_draw.save(str(output_path), quality=95)
    logger.info(f"✓ Saved visual detection artifact: {output_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" RF-DETR STAGING DEPLOYMENT & INFERENCE VERIFICATION")
    print("=" * 85)
    print(f" • Mode / Backend:         {args.backend.upper()}")
    print(f" • Target Confidence:      {args.conf} (Production Floor)")
    print(f" • Post-Processing Suite:  WBF + Span Fragment Merge + Margin Recalibration")
    if args.backend == "triton":
        print(f" • Staging Triton Server:  {args.triton_url}")
        print(f" • Model Name & Version:   {args.model_name} (v{args.model_version})")
    else:
        print(f" • Local Model Path:       {args.model_path}")
    print("=" * 85)

    # 1. Initialize Detector Engine
    try:
        if args.backend == "triton":
            detector = load_rfdetr_detector(
                backend="triton",
                triton_url=args.triton_url,
                model_version=args.model_version
            )
            logger.info("✓ Staging Triton Server is LIVE and responding!")
        elif args.backend == "onnx":
            detector = load_rfdetr_detector(backend="onnx", model_path=args.model_path)
            logger.info("✓ Loaded ONNX Runtime engine.")
        else:
            detector = load_rfdetr_detector(backend="plan", model_path=args.model_path)
            logger.info("✓ Loaded TensorRT Engine plan.")
    except Exception as e:
        logger.error(f"Failed to connect to / load {args.backend} engine: {e}")
        if args.backend == "triton":
            print("\nTroubleshooting Triton Port-Forwarding:")
            print("  1. Find your running Triton pod in Kubernetes:")
            print("     kubectl get pods -n <namespace> -l app=triton")
            print("  2. Forward gRPC port 8001:")
            print("     kubectl port-forward <pod-name> 8001:8001 -n <namespace>")
            print("  3. Re-run this test script.")
        sys.exit(1)

    # 2. Collect Test Images
    test_images: List[Path] = []
    if args.image:
        p = Path(args.image)
        if p.is_dir():
            test_images = sorted([f for f in p.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        elif p.exists():
            test_images = [p]
        else:
            logger.warning(f"Specified image '{args.image}' not found.")

    # Search for available test sample images in repo if none provided
    if not test_images:
        candidate_dirs = [
            Path("notebook/deployment/images"),
            Path("multi_class_rfdetr/images"),
            Path("coco_files/test"),
            Path("test")
        ]
        for cd in candidate_dirs:
            if cd.exists():
                found = sorted([f for f in cd.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
                if found:
                    test_images = found[:5]
                    break

    # If still none found, generate a synthetic warehouse placard image
    if not test_images:
        synth_path = output_dir / "synthetic_placard_test.jpg"
        print(f"Generating synthetic warehouse placard test image -> {synth_path}...")
        img = Image.new("RGB", (1920, 1080), color=(220, 220, 220))
        d = ImageDraw.Draw(img)
        # Draw mock blue bay placard 'D17'
        d.rectangle([600, 250, 1100, 480], fill=(20, 60, 160), outline=(10, 30, 100), width=4)
        d.text((650, 290), "D17", fill=(255, 255, 255))
        # Draw mock location shelf tag 'LOC-01-A'
        d.rectangle([700, 650, 1000, 750], fill=(250, 250, 250), outline=(0, 0, 0), width=2)
        d.text((720, 680), "LOC-01-A-12", fill=(0, 0, 0))
        img.save(str(synth_path))
        test_images = [synth_path]

    logger.info(f"Found {len(test_images)} test image(s) for verification.")

    # 3. Warmup Passes
    logger.info(f"Running {args.warmup_runs} warmup iteration(s)...")
    sample_img = Image.open(str(test_images[0])).convert("RGB")
    for _ in range(args.warmup_runs):
        _ = detector.predict_image(sample_img, conf_threshold=args.conf)

    # 4. Latency Benchmark
    logger.info(f"Executing latency benchmark ({args.benchmark_runs} runs)...")
    latencies = []
    for _ in range(args.benchmark_runs):
        t0 = time.perf_counter()
        _ = detector.predict_image(sample_img, conf_threshold=args.conf)
        latencies.append((time.perf_counter() - t0) * 1000)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    fps = 1000.0 / p50

    print("\n" + "=" * 85)
    print(" LATENCY BENCHMARK RESULTS")
    print("=" * 85)
    print(f" • Median Latency (P50):  {p50:.2f} ms")
    print(f" • 95th Percentile (P95): {p95:.2f} ms")
    print(f" • 99th Percentile (P99): {p99:.2f} ms")
    print(f" • Throughput (FPS):      {fps:.1f} inferences/sec")
    print("=" * 85 + "\n")

    # 5. End-to-End Inference Verification on Test Images
    total_detections = 0
    results_summary = []

    for idx, img_path in enumerate(test_images, 1):
        raw_pil = Image.open(str(img_path)).convert("RGB")
        t0 = time.perf_counter()

        detections = detector.predict_image(
            raw_pil,
            conf_threshold=args.conf,
            enable_wbf=True,
            enable_fragment_merge=True,
            enable_margin_recalibration=True,
            enable_shrinkage=True,
            shrink_factor=0.96
        )
        duration_ms = (time.perf_counter() - t0) * 1000
        total_detections += len(detections)

        logger.info(f"[{idx}/{len(test_images)}] {img_path.name}: {len(detections)} detection(s) ({duration_ms:.1f} ms)")
        for d_i, d in enumerate(detections, 1):
            logger.info(f"   {d_i}. [{d['description']}] Score: {d['score']:.3f} | BBox (ymin, xmin, ymax, xmax): {d['bbox']}")

        # Save annotated preview
        out_vis = output_dir / f"pred_{img_path.stem}.jpg"
        draw_bounding_boxes(raw_pil, detections, out_vis)

        results_summary.append({
            "image": img_path.name,
            "num_detections": len(detections),
            "latency_ms": round(duration_ms, 2),
            "detections": detections
        })

    print("\n" + "=" * 85)
    print(" STAGING VERIFICATION COMPLETE")
    print("=" * 85)
    print(f" • Total Images Evaluated:  {len(test_images)}")
    print(f" • Total Detections:        {total_detections}")
    print(f" • Visual Artifacts Saved:  {output_dir.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()

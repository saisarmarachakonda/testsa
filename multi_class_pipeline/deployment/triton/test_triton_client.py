#!/usr/bin/env python3
"""
test_triton_client.py: End-to-end verification script for RF-DETR via Triton.

Tests:
1. Triton server liveliness and model readiness.
2. Full image preprocessing, gRPC inference, and postprocessing.
3. Verification of {"description", "score", "bbox"} output schema.

Usage:
    # In Terminal 1: Forward Triton pod:
    kubectl port-forward <triton-pod-name> 8001:8001 -n <namespace>

    # In Terminal 2: Run test client:
    python test_triton_client.py --triton-url localhost:8001 --model-version 2
"""

import sys
import io
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

# Add local path
sys.path.append(str(Path(__file__).resolve().parent / "samscv_triton_clients"))
sys.path.append(str(Path(__file__).resolve().parent))

from models.rfdetr import RFDetrTritonClient
from inference.iras.floor.detection import RfDetrLocationTagDetection

def parse_args():
    parser = argparse.ArgumentParser(description="Test RF-DETR Triton Client via port-forward.")
    parser.add_argument("--triton-url", default="localhost:8001", help="Triton gRPC endpoint (default: localhost:8001)")
    parser.add_argument("--model-name", default="rfdetr", help="Triton model name (default: rfdetr)")
    parser.add_argument("--model-version", default="2", help="Model version (default: 2)")
    parser.add_argument("--image", default=None, help="Path to input test image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", default="triton_prediction_result.jpg", help="Output visualization path")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 75)
    print(" RF-DETR Triton Client Verification")
    print(f" • Triton Server URL: {args.triton_url}")
    print(f" • Model Name:        {args.model_name}")
    print(f" • Model Version:     {args.model_version}")
    print(f" • Confidence Thresh: {args.conf}")
    print("=" * 75)

    # 1. Initialize Client
    try:
        detector = RfDetrLocationTagDetection(
            triton_url=args.triton_url,
            model_name=args.model_name,
            model_version=args.model_version,
            conf_threshold=args.conf
        )
        print("✓ Initialized RfDetrLocationTagDetection client.")
    except Exception as e:
        print(f"ERROR: Could not initialize client: {e}")
        sys.exit(1)

    # 2. Check model readiness
    print(f"Checking if {args.model_name} (v{args.model_version}) is ready on {args.triton_url}...")
    if not detector.client.is_model_ready():
        print(f"WARNING: Model {args.model_name} v{args.model_version} is not reporting READY yet.")
        print("Ensure port-forwarding is running:")
        print("  kubectl port-forward <triton-pod-name> 8001:8001 -n <namespace>")

    # 3. Load or generate image
    if args.image and Path(args.image).exists():
        img = Image.open(args.image).convert("RGB")
        print(f"Using image from: {args.image} ({img.size[0]}x{img.size[1]})")
    else:
        print("Creating synthetic test image with sample warehouse location tag...")
        img = Image.new("RGB", (1000, 750), color=(230, 230, 230))
        d = ImageDraw.Draw(img)
        d.rectangle([200, 200, 500, 400], fill=(245, 245, 245), outline=(0, 0, 0), width=3)
        d.text((250, 280), "LOC-01-A-12", fill=(0, 0, 0))

    # 4. Perform Inference
    print("\nExecuting Triton gRPC inference...")
    try:
        detections = detector.detect(img)
        print(f"✓ Received {len(detections)} detections:")
        for idx, det in enumerate(detections, 1):
            desc = det.get("description")
            score = det.get("score")
            bbox = det.get("bbox")
            print(f"  {idx}. [{desc}] Score: {score:.3f} | BBox (ymin, xmin, ymax, xmax): {bbox}")

        # 5. Draw visualization
        draw = ImageDraw.Draw(img)
        for det in detections:
            ymin, xmin, ymax, xmax = det["bbox"]
            label = f"{det['description']} ({det['score']:.2f})"
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 59, 48), width=3)
            draw.text((xmin + 4, max(0, ymin - 12)), label, fill=(255, 59, 48))

        img.save(args.output)
        print(f"\n✓ Saved prediction visualization to: {args.output}")
        print("==============================================================================")
        print(" Triton Verification SUCCESSFUL!")
        print("==============================================================================")
    except Exception as e:
        print(f"\nInference failed: {e}")
        print("Make sure the Triton pod is running and port-forwarded.")

if __name__ == "__main__":
    main()

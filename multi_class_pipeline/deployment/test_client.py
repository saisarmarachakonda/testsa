#!/usr/bin/env python3
"""
test_client.py: Test client for RF-DETR private AKS microservice via port-forwarding.

Usage:
    python test_client.py
    python test_client.py --image path/to/image.jpg --endpoint http://127.0.0.1:8000
"""

import sys
import io
import argparse
from pathlib import Path
import requests
from PIL import Image, ImageDraw

DEFAULT_ENDPOINT = "http://127.0.0.1:8000"

def parse_args():
    parser = argparse.ArgumentParser(description="Test private RF-DETR inference service.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Service URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--image", default=None, help="Path to input test image")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (default: 0.20)")
    parser.add_argument("--output", default="prediction_result.jpg", help="Path to save output visualization")
    return parser.parse_args()

def main():
    args = parse_args()
    endpoint = args.endpoint.rstrip("/")

    print("==============================================================================")
    print(" Testing RF-DETR Private Service via Port-Forwarding")
    print(f" Target Endpoint: {endpoint}")
    print("==============================================================================")

    # 1. Test Health Endpoint
    try:
        health_resp = requests.get(f"{endpoint}/healthz", timeout=5)
        health_data = health_resp.json()
        print(f"✓ Health Check: {health_resp.status_code}")
        print(f"   - RF-DETR Loaded: {health_data.get('rfdetr_loaded')}")
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Could not connect to {endpoint}.")
        print("Make sure you have started port forwarding in another terminal:")
        print("   ./port_forward.sh")
        print("or run:")
        print("   kubectl port-forward service/rfdetr-inference-service 8000:80")
        sys.exit(1)

    # 2. Test Metadata Endpoint
    meta_resp = requests.get(f"{endpoint}/metadata", timeout=5)
    if meta_resp.status_code == 200:
        meta = meta_resp.json()
        print(f"✓ Metadata: Resolution={meta.get('resolution')} | Classes={meta.get('classes')}")

    # 3. Locate or Generate Test Image
    img_path = None
    if args.image and Path(args.image).exists():
        img_path = Path(args.image)
    else:
        for p in [Path("../test_1.jpg"), Path("../test.jpg"), Path("test.jpg")]:
            if p.exists():
                img_path = p
                break
        if not img_path:
            cand = list(Path("..").glob("*.jpg")) + list(Path("..").glob("*.png"))
            if cand:
                img_path = cand[0]

    if img_path and img_path.exists():
        print(f"Using image: {img_path}")
        with open(img_path, "rb") as f:
            img_bytes = f.read()
    else:
        print("No image file specified; creating synthetic test image...")
        synthetic = Image.new("RGB", (800, 600), color=(220, 220, 220))
        d = ImageDraw.Draw(synthetic)
        d.rectangle([150, 150, 450, 350], fill=(240, 240, 240), outline=(0, 0, 0), width=4)
        buf = io.BytesIO()
        synthetic.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

    # 4. Invoke Prediction (Detection)
    print(f"\nSending POST to {endpoint}/predict (conf={args.conf})...")
    resp = requests.post(
        f"{endpoint}/predict",
        params={"conf_threshold": args.conf},
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        timeout=45
    )

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    count = data.get("detections_count", 0)
    inf_time = data.get("inference_time_ms", 0)
    print(f"\n✓ Received {count} detections (Detection: {inf_time} ms):")
    for idx, det in enumerate(data.get("detections", []), 1):
        cname = det['class_name']
        score = det['confidence']
        box = det['box_xyxy']
        print(f"  {idx}. [{cname}] (conf: {score:.2f}) | box: {box}")

    # 5. Draw and Save Visualization
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    colors = [(255, 59, 48), (52, 199, 89), (0, 122, 255), (88, 86, 214), (255, 149, 0)]

    for det in data.get("detections", []):
        x1, y1, x2, y2 = det["box_xyxy"]
        cid = det["class_id"]
        c = colors[cid % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        label = f"{det['class_name']} ({det['confidence']:.2f})"
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill=c)

    pil_img.save(args.output)
    print(f"\n✓ Visualization saved to: {args.output}")

if __name__ == "__main__":
    main()

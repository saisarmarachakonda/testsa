#!/usr/bin/env python3
"""
test_client.py: Test client for RF-DETR + PaddleOCR private AKS microservice via port-forwarding.

Usage:
    python test_client.py
    python test_client.py --image path/to/image.jpg --endpoint http://127.0.0.1:8000
    python test_client.py --image path/to/image.jpg --no-ocr
"""

import sys
import io
import argparse
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFont

DEFAULT_ENDPOINT = "http://127.0.0.1:8000"

def parse_args():
    parser = argparse.ArgumentParser(description="Test private RF-DETR + PaddleOCR inference service.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Service URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--image", default=None, help="Path to input test image")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (default: 0.20)")
    parser.add_argument("--no-ocr", action="store_true", help="Disable PaddleOCR text recognition on tags")
    parser.add_argument("--output", default="prediction_result.jpg", help="Path to save output visualization")
    return parser.parse_args()

def main():
    args = parse_args()
    endpoint = args.endpoint.rstrip("/")

    print("==============================================================================")
    print(" Testing RF-DETR + PaddleOCR Private Service via Port-Forwarding")
    print(f" Target Endpoint: {endpoint}")
    print(f" Run PaddleOCR:   {not args.no_ocr}")
    print("==============================================================================")

    # 1. Test Health Endpoint
    try:
        health_resp = requests.get(f"{endpoint}/healthz", timeout=5)
        health_data = health_resp.json()
        print(f"✓ Health Check: {health_resp.status_code}")
        print(f"   - RF-DETR Loaded:   {health_data.get('rfdetr_loaded')}")
        print(f"   - PaddleOCR Loaded: {health_data.get('paddleocr_loaded')}")
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
        print("No image file specified; creating synthetic test image with text...")
        synthetic = Image.new("RGB", (800, 600), color=(220, 220, 220))
        d = ImageDraw.Draw(synthetic)
        # Draw tag box with text
        d.rectangle([150, 150, 450, 350], fill=(240, 240, 240), outline=(0, 0, 0), width=4)
        d.text((180, 220), "LOC-A-12-04", fill=(0, 0, 0))
        buf = io.BytesIO()
        synthetic.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

    # 4. Invoke Prediction (Detection + PaddleOCR)
    run_ocr_flag = not args.no_ocr
    print(f"\nSending POST to {endpoint}/predict (conf={args.conf}, run_ocr={run_ocr_flag})...")
    resp = requests.post(
        f"{endpoint}/predict",
        params={"conf_threshold": args.conf, "run_ocr": run_ocr_flag},
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        timeout=45
    )

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    count = data.get("detections_count", 0)
    inf_time = data.get("inference_time_ms", 0)
    ocr_time = data.get("ocr_time_ms")
    print(f"\n✓ Received {count} detections (Detection: {inf_time} ms | PaddleOCR: {ocr_time or 0} ms):")
    for idx, det in enumerate(data.get("detections", []), 1):
        cname = det['class_name']
        score = det['confidence']
        box = det['box_xyxy']
        ocr_txt = det.get('ocr_text')
        ocr_conf = det.get('ocr_confidence')
        if ocr_txt:
            print(f"  {idx}. [{cname}] (conf: {score:.2f}) -> OCR Text: '{ocr_txt}' (ocr_conf: {ocr_conf:.2f}) | box: {box}")
        else:
            print(f"  {idx}. [{cname}] (conf: {score:.2f}) -> No text recognized | box: {box}")

    # 5. Draw and Save Visualization
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    colors = [(255, 59, 48), (52, 199, 89), (0, 122, 255), (88, 86, 214), (255, 149, 0)]

    for det in data.get("detections", []):
        x1, y1, x2, y2 = det["box_xyxy"]
        cid = det["class_id"]
        c = colors[cid % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        
        # Display detection label + OCR text
        ocr_txt = det.get("ocr_text")
        if ocr_txt:
            label = f"{det['class_name']} ({det['confidence']:.2f}) | '{ocr_txt}'"
        else:
            label = f"{det['class_name']} ({det['confidence']:.2f})"
            
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill=c)

    pil_img.save(args.output)
    print(f"\n✓ Visualization with PaddleOCR text saved to: {args.output}")

if __name__ == "__main__":
    main()

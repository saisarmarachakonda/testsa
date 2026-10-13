#!/usr/bin/env python3
"""
================================================================================
RF-DETR ONNX Inference & Post-Processing Client via Port-Forwarding
================================================================================
Supports inference against:
  1. Triton Server (v2 HTTP or gRPC) port-forwarded from Kubernetes (e.g. localhost:8000 / 8001)
  2. RF-DETR FastAPI / REST Microservice port-forwarded from Kubernetes (e.g. localhost:8000 / 8001)
  3. Local ONNX Runtime engine (if onnxruntime is installed and model file is provided)

Key Features:
  • Default confidence threshold: 0.71 (strictly configurable via CLI)
  • Full multi-stage production post-processing:
      - Sigmoid normalization of classification logits
      - Minimum & maximum area ratio guard (suppresses microscopic noise and full-frame boxes)
      - Aspect ratio guard (rejects abnormal vertical/horizontal slivers)
      - Edge crop filter (discards objects truncated at image boundaries)
      - Weighted Box Fusion (WBF) query consolidation
      - Alphanumeric placard fragment merging (e.g., 'D' + '17' -> 'D17')
      - Placard context margin recalibration (expands text crop to outer physical plate)
      - Boundary box shrinkage (insets box slightly to suppress shelf edge noise)
  • Automatic response parsing across Triton v2 JSON, Triton gRPC, and REST schemas
  • Output formatting in both normalized and pixel coordinates ([ymin, xmin, ymax, xmax])
  • Visual bounding box rendering with Pillow

Usage:
  # 1. Connect to port-forwarded Triton HTTP service:
  # (Run in background: kubectl port-forward pod/triton-server 8000:8000)
  python onnx_portforward_inference.py --endpoint http://localhost:8000 --service-type triton_http --image sample.jpg --conf 0.71

  # 2. Connect to port-forwarded Triton gRPC service:
  # (Run in background: kubectl port-forward pod/triton-server 8001:8001)
  python onnx_portforward_inference.py --endpoint localhost:8001 --service-type triton_grpc --image sample.jpg --conf 0.71

  # 3. Connect to port-forwarded FastAPI Gateway microservice:
  # (Run in background: kubectl port-forward service/rfdetr-service 8000:8000)
  python onnx_portforward_inference.py --endpoint http://localhost:8000 --service-type rest_api --image sample.jpg --conf 0.71

  # 4. Run local ONNX file directly:
  python onnx_portforward_inference.py --service-type local_onnx --local-model notebook/deployment/models/rfdetr_model.onnx --image sample.jpg --conf 0.71
================================================================================
"""

import os
import sys
import time
import json
import argparse
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

# Third-party dependencies
try:
    import numpy as np
except ImportError:
    print("Error: 'numpy' is required. Install via: pip install numpy")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: 'Pillow' is required. Install via: pip install pillow")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RFDetrPortForwardInference")

# ==============================================================================
# Model & Pipeline Specifications
# ==============================================================================
CLASS_NAMES = ["blue_aisle", "blue_bay", "location_tag"]
NUM_CLASSES = len(CLASS_NAMES)

DEFAULT_CONF_THRESHOLD = 0.71
INPUT_HEIGHT = 640
INPUT_WIDTH = 480
DEFAULT_MODEL_NAME = "tagdet_rt"
DEFAULT_INPUT_NAME = "images"
DEFAULT_OUTPUT_NAME = "output"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==============================================================================
# 1. Image Preprocessing (Strictly matches model input: [1, 3, 640, 480] FP32)
# ==============================================================================
def preprocess_image(
    image: Union[Image.Image, np.ndarray, str, Path],
    target_width: int = INPUT_WIDTH,
    target_height: int = INPUT_HEIGHT
) -> Tuple[np.ndarray, int, int]:
    """
    Standard RF-DETR Preprocessing:
    1. Loads / converts image to RGB PIL.
    2. Resizes image to (target_width, target_height) using bilinear interpolation.
    3. Normalizes pixel values using ImageNet mean & standard deviation.
    4. Formats to shape [1, 3, target_height, target_width] FP32 (NCHW).
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(str(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 3:
            pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            pil_img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    orig_w, orig_h = pil_img.size

    # Bilinear resize
    resized = pil_img.resize((target_width, target_height), Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0

    # Channel normalization
    img_norm = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    # Transpose HWC -> CHW -> NCHW
    input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    return input_tensor, orig_w, orig_h


# ==============================================================================
# 2. Post-Processing Techniques
# ==============================================================================

def apply_weighted_box_fusion(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_thresh: float = 0.55
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Box Fusion (WBF):
    Merges overlapping queries of the same class, calculating consensus bounding box
    coordinates weighted by each candidate's confidence score.
    """
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=int)

    fused_boxes, fused_scores, fused_labels = [], [], []

    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        lbl_boxes = boxes[idx]
        lbl_scores = scores[idx]

        order = np.argsort(lbl_scores)[::-1]
        lbl_boxes = lbl_boxes[order]
        lbl_scores = lbl_scores[order]

        used = np.zeros(len(lbl_boxes), dtype=bool)
        for i in range(len(lbl_boxes)):
            if used[i]:
                continue
            cluster = [i]
            for j in range(i + 1, len(lbl_boxes)):
                if used[j]:
                    continue
                bA, bB = lbl_boxes[i], lbl_boxes[j]
                xA, yA = max(bA[0], bB[0]), max(bA[1], bB[1])
                xB, yB = min(bA[2], bB[2]), min(bA[3], bB[3])
                inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                areaA = (bA[2] - bA[0]) * (bA[3] - bA[1])
                areaB = (bB[2] - bB[0]) * (bB[3] - bB[1])
                denom = areaA + areaB - inter
                if denom > 0 and (inter / denom) >= iou_thresh:
                    cluster.append(j)
                    used[j] = True
            used[i] = True

            c_boxes = lbl_boxes[cluster]
            c_scores = lbl_scores[cluster]
            total_s = np.sum(c_scores)
            fused_box = np.sum(c_boxes * c_scores[:, None], axis=0) / total_s
            fused_boxes.append(fused_box)
            fused_scores.append(float(np.max(c_scores)))
            fused_labels.append(lbl)

    return np.array(fused_boxes, dtype=np.float32), np.array(fused_scores, dtype=np.float32), np.array(fused_labels, dtype=int)


def merge_adjacent_placard_fragments(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    y_overlap_thresh: float = 0.65,
    max_horizontal_gap_ratio: float = 0.45
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Placard Fragment Merging:
    Merges split alphanumeric fragments (e.g. 'D' and '17' on a 'D17' warehouse placard)
    into a unified physical bounding box.
    """
    if len(boxes) <= 1:
        return boxes, scores, labels

    fused_boxes, fused_scores, fused_labels = [], [], []

    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        lbl_boxes = boxes[idx]
        lbl_scores = scores[idx]

        used = np.zeros(len(lbl_boxes), dtype=bool)
        for i in range(len(lbl_boxes)):
            if used[i]:
                continue
            group = [i]
            bA = lbl_boxes[i]
            hA = bA[3] - bA[1]

            for j in range(i + 1, len(lbl_boxes)):
                if used[j]:
                    continue
                bB = lbl_boxes[j]
                hB = bB[3] - bB[1]

                # Check vertical alignment (same beam line)
                y_inter = max(0.0, min(bA[3], bB[3]) - max(bA[1], bB[1]))
                min_h = min(hA, hB)
                if min_h > 0 and (y_inter / min_h) >= y_overlap_thresh:
                    # Check horizontal proximity
                    x_gap = max(0.0, max(bA[0], bB[0]) - min(bA[2], bB[2]))
                    if x_gap <= max_horizontal_gap_ratio * min_h:
                        group.append(j)
                        used[j] = True

            used[i] = True
            grp_boxes = lbl_boxes[group]
            grp_scores = lbl_scores[group]

            union_box = [
                float(np.min(grp_boxes[:, 0])),
                float(np.min(grp_boxes[:, 1])),
                float(np.max(grp_boxes[:, 2])),
                float(np.max(grp_boxes[:, 3]))
            ]
            fused_boxes.append(union_box)
            fused_scores.append(float(np.max(grp_scores)))
            fused_labels.append(lbl)

    return np.array(fused_boxes, dtype=np.float32), np.array(fused_scores, dtype=np.float32), np.array(fused_labels, dtype=int)


def recalibrate_placard_margins(
    box_xyxy: Union[List[float], np.ndarray],
    expand_x_ratio: float = 0.10,
    expand_y_ratio: float = 0.05
) -> List[float]:
    """
    Margin Recalibration:
    Expands inner text crops out to the physical placard boundary plate.
    """
    x1, y1, x2, y2 = box_xyxy
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    new_w = w * (1.0 + expand_x_ratio)
    new_h = h * (1.0 + expand_y_ratio)

    return [
        max(0.0, float(cx - new_w / 2.0)),
        max(0.0, float(cy - new_h / 2.0)),
        min(1.0, float(cx + new_w / 2.0)),
        min(1.0, float(cy + new_h / 2.0))
    ]


def postprocess_rfdetr_output(
    raw_output: np.ndarray,
    orig_w: int,
    orig_h: int,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    classes: Optional[List[str]] = None,
    pixel_coordinates: bool = True,
    bbox_format: str = "ymin_xmin_ymax_xmax",
    filter_edge_crops: bool = True,
    edge_crop_margin: float = 0.005,
    enable_shrinkage: bool = True,
    shrink_factor: float = 0.96,
    enable_wbf: bool = True,
    wbf_iou_thresh: float = 0.55,
    enable_fragment_merge: bool = True,
    enable_margin_recalibration: bool = True,
    min_area_ratio: float = 0.0001,
    max_area_ratio: float = 0.75,
    min_aspect_ratio: float = 0.35,
    max_aspect_ratio: float = 8.0
) -> List[Dict[str, Any]]:
    """
    Comprehensive Post-Processing Pipeline:
      1. Slices raw output: coords [cx, cy, w, h] + class logits/probs.
      2. Normalizes scores using Sigmoid if values outside [0, 1].
      3. Filters by confidence threshold (default: 0.71).
      4. Geometric hallucination suppression (area & aspect ratio guards).
      5. Edge crop suppression (removes truncated bounding boxes at image boundary).
      6. Weighted Box Fusion (WBF) query consolidation.
      7. Alphanumeric fragment merging.
      8. Placard context margin recalibration.
      9. Border shrinkage (insets box to avoid background seams).
      10. Output formatting and coordinate scaling.
    """
    class_list = classes or CLASS_NAMES

    # Squeeze batch dimension if present
    if raw_output.ndim == 3 and raw_output.shape[0] == 1:
        out = raw_output[0]
    else:
        out = raw_output

    boxes = out[:, :4]    # [N, 4] -> cx, cy, w, h (normalized 0..1)
    scores = out[:, 4:]   # [N, num_classes]

    # Convert logits to probabilities via sigmoid if necessary
    if scores.min() < 0.0 or scores.max() > 1.0:
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))

    class_ids = np.argmax(scores, axis=-1)
    max_scores = np.max(scores, axis=-1)

    # 1. Filter by confidence threshold (0.71)
    keep_indices = np.where(max_scores >= conf_threshold)[0]
    cand_boxes: List[List[float]] = []
    cand_scores: List[float] = []
    cand_classes: List[int] = []

    for idx in keep_indices:
        cid = int(class_ids[idx])
        score = float(max_scores[idx])
        cx, cy, w, h = boxes[idx]

        if w <= 0.0 or h <= 0.0 or score <= 0.0:
            continue

        # Area Guard (reject microscopic noise or full-screen boxes)
        box_area = w * h
        if box_area < min_area_ratio or box_area > max_area_ratio:
            continue

        # Aspect Ratio Guard (reject extreme slivers)
        aspect_ratio = w / (h + 1e-6)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        # Convert cxcywh to xyxy (normalized 0..1)
        x1 = max(0.0, float(cx - w / 2.0))
        y1 = max(0.0, float(cy - h / 2.0))
        x2 = min(1.0, float(cx + w / 2.0))
        y2 = min(1.0, float(cy + h / 2.0))

        # Edge Crop Filter (reject objects clipped at border)
        if filter_edge_crops:
            touches_border = (
                x1 <= edge_crop_margin or
                y1 <= edge_crop_margin or
                x2 >= (1.0 - edge_crop_margin) or
                y2 >= (1.0 - edge_crop_margin)
            )
            if touches_border:
                continue

        cand_boxes.append([x1, y1, x2, y2])
        cand_scores.append(score)
        cand_classes.append(cid)

    if not cand_boxes:
        return []

    # 2. Weighted Box Fusion (WBF)
    if enable_wbf and len(cand_boxes) > 1:
        f_boxes, f_scores, f_classes = apply_weighted_box_fusion(
            np.array(cand_boxes, dtype=np.float32),
            np.array(cand_scores, dtype=np.float32),
            np.array(cand_classes, dtype=int),
            iou_thresh=wbf_iou_thresh
        )
    else:
        f_boxes = np.array(cand_boxes, dtype=np.float32)
        f_scores = np.array(cand_scores, dtype=np.float32)
        f_classes = np.array(cand_classes, dtype=int)

    # 3. Fragment Merging
    if enable_fragment_merge and len(f_boxes) > 1:
        f_boxes, f_scores, f_classes = merge_adjacent_placard_fragments(
            f_boxes,
            f_scores,
            f_classes
        )

    detections: List[Dict[str, Any]] = []

    for i in range(len(f_boxes)):
        x1, y1, x2, y2 = f_boxes[i]
        score = float(f_scores[i])
        cid = int(f_classes[i])

        # 4. Context Margin Recalibration
        if enable_margin_recalibration:
            x1, y1, x2, y2 = recalibrate_placard_margins([x1, y1, x2, y2])

        # 5. Border Shrinkage
        if enable_shrinkage and shrink_factor < 1.0:
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            shrunk_w = (x2 - x1) * shrink_factor
            shrunk_h = (y2 - y1) * shrink_factor
            x1 = max(0.0, mid_x - shrunk_w / 2.0)
            y1 = max(0.0, mid_y - shrunk_h / 2.0)
            x2 = min(1.0, mid_x + shrunk_w / 2.0)
            y2 = min(1.0, mid_y + shrunk_h / 2.0)

        # 6. Coordinate Formatting
        if pixel_coordinates:
            px_x1 = round(float(x1 * orig_w), 2)
            px_y1 = round(float(y1 * orig_h), 2)
            px_x2 = round(float(x2 * orig_w), 2)
            px_y2 = round(float(y2 * orig_h), 2)

            if bbox_format == "ymin_xmin_ymax_xmax":
                formatted_box = [px_y1, px_x1, px_y2, px_x2]
            elif bbox_format == "xyxy":
                formatted_box = [px_x1, px_y1, px_x2, px_y2]
            elif bbox_format == "xywh":
                formatted_box = [px_x1, px_y1, round(px_x2 - px_x1, 2), round(px_y2 - px_y1, 2)]
            else:
                formatted_box = [px_y1, px_x1, px_y2, px_x2]
        else:
            if bbox_format == "ymin_xmin_ymax_xmax":
                formatted_box = [round(float(y1), 4), round(float(x1), 4), round(float(y2), 4), round(float(x2), 4)]
            elif bbox_format == "xyxy":
                formatted_box = [round(float(x1), 4), round(float(y1), 4), round(float(x2), 4), round(float(y2), 4)]
            elif bbox_format == "xywh":
                formatted_box = [round(float(x1), 4), round(float(y1), 4), round(float(x2 - x1), 4), round(float(y2 - y1), 4)]
            else:
                formatted_box = [round(float(y1), 4), round(float(x1), 4), round(float(y2), 4), round(float(x2), 4)]

        class_name = class_list[cid] if cid < len(class_list) else f"class_{cid}"

        detections.append({
            "description": class_name,
            "score": round(score, 4),
            "bbox": formatted_box,
            "class_id": cid
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


# ==============================================================================
# 3. Port-Forwarded Client Communicators
# ==============================================================================

class TritonHTTPClient:
    """
    Communicates with port-forwarded Triton Server via HTTP v2 Inference Protocol.
    Default port: 8000.
    """
    def __init__(self, endpoint_url: str = "http://localhost:8000", model_name: str = DEFAULT_MODEL_NAME):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name

    def check_health(self) -> bool:
        url = f"{self.endpoint_url}/v2/health/ready"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Triton HTTP health check failed at {url}: {e}")
            return False

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        url = f"{self.endpoint_url}/v2/models/{self.model_name}/infer"
        payload = {
            "inputs": [
                {
                    "name": DEFAULT_INPUT_NAME,
                    "shape": list(input_tensor.shape),
                    "datatype": "FP32",
                    "data": input_tensor.flatten().tolist()
                }
            ],
            "outputs": [
                {"name": DEFAULT_OUTPUT_NAME}
            ]
        }
        json_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=json_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))

        raw_output = np.array(resp_body["outputs"][0]["data"], dtype=np.float32)
        shape = resp_body["outputs"][0]["shape"]
        return raw_output.reshape(shape)


class TritonGRPCClient:
    """
    Communicates with port-forwarded Triton Server via gRPC.
    Default port: 8001.
    """
    def __init__(self, endpoint_url: str = "localhost:8001", model_name: str = DEFAULT_MODEL_NAME):
        try:
            import tritonclient.grpc as grpcclient
            from tritonclient.grpc import InferInput, InferRequestedOutput
            self.grpcclient = grpcclient
            self.InferInput = InferInput
            self.InferRequestedOutput = InferRequestedOutput
        except ImportError:
            raise ImportError(
                "tritonclient[grpc] is required for Triton gRPC mode. "
                "Install via: pip install 'tritonclient[grpc]'"
            )
        self.client = grpcclient.InferenceServerClient(url=endpoint_url)
        self.model_name = model_name

    def check_health(self) -> bool:
        return self.client.is_server_live() and self.client.is_model_ready(self.model_name)

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        infer_input = self.InferInput(DEFAULT_INPUT_NAME, input_tensor.shape, "FP32")
        infer_input.set_data_from_numpy(input_tensor)
        infer_output = self.InferRequestedOutput(DEFAULT_OUTPUT_NAME)

        response = self.client.infer(
            model_name=self.model_name,
            inputs=[infer_input],
            outputs=[infer_output]
        )
        return response.as_numpy(DEFAULT_OUTPUT_NAME)


class RestAPIGatewayClient:
    """
    Communicates with port-forwarded FastAPI Gateway microservice.
    Default port: 8000 or 8001.
    """
    def __init__(self, endpoint_url: str = "http://localhost:8000"):
        self.endpoint_url = endpoint_url.rstrip("/")

    def check_health(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.endpoint_url}/healthz")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning(f"REST Gateway health check failed: {e}")
            return False

    def infer(self, image_path: Path, conf_threshold: float) -> List[Dict[str, Any]]:
        url = f"{self.endpoint_url}/predict?conf_threshold={conf_threshold}"
        boundary = "----WebKitFormBoundaryRFDETR" + str(int(time.time()))

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        detections = []
        for d in data.get("detections", []):
            x1, y1, x2, y2 = d["box_xyxy"]
            detections.append({
                "description": d["class_name"],
                "score": float(d["confidence"]),
                "bbox": [y1, x1, y2, x2],
                "class_id": d.get("class_id", 0)
            })
        return detections


class LocalONNXRuntimeEngine:
    """
    Runs ONNX model locally using ONNX Runtime for offline validation.
    """
    def __init__(self, model_path: Union[str, Path]):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for local_onnx backend. Install via: pip install onnxruntime")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Local ONNX model not found: {self.model_path}")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(f"Loaded Local ONNX Model: {self.model_path} via {providers[0]}")

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: input_tensor})[0]


# ==============================================================================
# 4. Visualization Helper
# ==============================================================================
def draw_and_save_detections(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    output_path: Path
):
    """Draws color-coded bounding boxes and detection labels on the image."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    color_map = {
        "location_tag": (255, 61, 0),    # Bright Red-Orange
        "blue_aisle":   (0, 229, 255),   # Bright Cyan
        "blue_bay":     (0, 230, 118)    # Bright Green
    }

    for det in detections:
        ymin, xmin, ymax, xmax = det["bbox"]
        score = det["score"]
        label = det["description"]
        color = color_map.get(label, (255, 255, 0))

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        caption = f"{label} {score:.2f}"
        text_bbox = draw.textbbox((xmin + 4, max(0, ymin - 16)), caption)
        draw.rectangle(text_bbox, fill=color)
        draw.text((xmin + 4, max(0, ymin - 16)), caption, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_draw.save(str(output_path), quality=95)
    logger.info(f"✓ Saved visual detection output to: {output_path}")


# ==============================================================================
# 5. Main Execution Script
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="RF-DETR ONNX Inference & Post-Processing via Port-Forwarding")
    parser.add_argument("--endpoint", default="http://localhost:8000",
                        help="Port-forwarded service URL (default: http://localhost:8000)")
    parser.add_argument("--service-type", default="triton_http",
                        choices=["triton_http", "triton_grpc", "rest_api", "local_onnx"],
                        help="Backend service type: triton_http, triton_grpc, rest_api, or local_onnx")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help=f"Triton model name (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--local-model", default="notebook/deployment/models/rfdetr_model.onnx",
                        help="Local path to .onnx file (used when service-type is local_onnx)")
    parser.add_argument("--image", default=None,
                        help="Path to input image (creates synthetic test pattern if omitted)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD,
                        help=f"Confidence threshold (default: {DEFAULT_CONF_THRESHOLD})")
    parser.add_argument("--output", default="predictions_result.jpg",
                        help="Path to save annotated visualization image")
    parser.add_argument("--bbox-format", default="ymin_xmin_ymax_xmax",
                        choices=["ymin_xmin_ymax_xmax", "xyxy", "xywh"],
                        help="Coordinate format for printed bounding boxes")
    parser.add_argument("--disable-wbf", action="store_true",
                        help="Disable Weighted Box Fusion")
    parser.add_argument("--disable-fragment-merge", action="store_true",
                        help="Disable adjacent alphanumeric fragment merging")
    parser.add_argument("--disable-margin-recalibration", action="store_true",
                        help="Disable placard context margin recalibration")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 80)
    print(" RF-DETR ONNX INFERENCE VIA PORT-FORWARDING")
    print("=" * 80)
    print(f" • Service Type:         {args.service_type.upper()}")
    print(f" • Target Endpoint:      {args.endpoint}")
    print(f" • Confidence Threshold: {args.conf}")
    print(f" • Model Name:           {args.model_name}")
    print(f" • Classes:              {CLASS_NAMES}")
    print(f" • Post-Processing:      WBF={not args.disable_wbf}, FragmentMerge={not args.disable_fragment_merge}, MarginRecal={not args.disable_margin_recalibration}")
    print("=" * 80)

    # 1. Resolve or Create Test Image
    if args.image and Path(args.image).exists():
        img_path = Path(args.image)
        pil_img = Image.open(str(img_path)).convert("RGB")
        logger.info(f"Loaded test image: {img_path} ({pil_img.size[0]}x{pil_img.size[1]})")
    else:
        logger.info("No input image specified. Generating synthetic test pattern...")
        pil_img = Image.new("RGB", (1000, 750), color=(235, 235, 235))
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([200, 250, 420, 380], fill=(210, 210, 210), outline=(0, 0, 0), width=3)
        draw.text((220, 300), "D17 LOC TAG", fill=(0, 0, 0))
        img_path = Path("sample_synthetic_test.jpg")
        pil_img.save(str(img_path))
        logger.info(f"Saved temporary synthetic test image to: {img_path}")

    # 2. Preprocess Image
    input_tensor, orig_w, orig_h = preprocess_image(pil_img)
    logger.info(f"Preprocessed tensor shape: {input_tensor.shape} (FP32 NCHW)")

    # 3. Perform Inference via Selected Backend
    t0 = time.time()
    detections = []

    try:
        if args.service_type == "triton_http":
            client = TritonHTTPClient(endpoint_url=args.endpoint, model_name=args.model_name)
            if not client.check_health():
                logger.warning(f"Health check at {args.endpoint} did not respond, attempting inference...")
            raw_output = client.infer(input_tensor)
            latency_ms = (time.time() - t0) * 1000

            # Apply post-processing
            detections = postprocess_rfdetr_output(
                raw_output=raw_output,
                orig_w=orig_w,
                orig_h=orig_h,
                conf_threshold=args.conf,
                bbox_format=args.bbox_format,
                enable_wbf=not args.disable_wbf,
                enable_fragment_merge=not args.disable_fragment_merge,
                enable_margin_recalibration=not args.disable_margin_recalibration
            )

        elif args.service_type == "triton_grpc":
            client = TritonGRPCClient(endpoint_url=args.endpoint, model_name=args.model_name)
            raw_output = client.infer(input_tensor)
            latency_ms = (time.time() - t0) * 1000

            detections = postprocess_rfdetr_output(
                raw_output=raw_output,
                orig_w=orig_w,
                orig_h=orig_h,
                conf_threshold=args.conf,
                bbox_format=args.bbox_format,
                enable_wbf=not args.disable_wbf,
                enable_fragment_merge=not args.disable_fragment_merge,
                enable_margin_recalibration=not args.disable_margin_recalibration
            )

        elif args.service_type == "rest_api":
            client = RestAPIGatewayClient(endpoint_url=args.endpoint)
            detections = client.infer(image_path=img_path, conf_threshold=args.conf)
            latency_ms = (time.time() - t0) * 1000

        elif args.service_type == "local_onnx":
            engine = LocalONNXRuntimeEngine(model_path=args.local_model)
            raw_output = engine.infer(input_tensor)
            latency_ms = (time.time() - t0) * 1000

            detections = postprocess_rfdetr_output(
                raw_output=raw_output,
                orig_w=orig_w,
                orig_h=orig_h,
                conf_threshold=args.conf,
                bbox_format=args.bbox_format,
                enable_wbf=not args.disable_wbf,
                enable_fragment_merge=not args.disable_fragment_merge,
                enable_margin_recalibration=not args.disable_margin_recalibration
            )

        else:
            raise ValueError(f"Unknown service type: {args.service_type}")

    except Exception as e:
        logger.error(f"Inference request failed: {e}")
        print("\n" + "=" * 80)
        print(" [CONNECTION NOTICE] Make sure your port-forwarding tunnel is active:")
        print("   For Triton HTTP (port 8000):")
        print("     kubectl port-forward <triton-pod-name> 8000:8000 -n <namespace>")
        print("   For Triton gRPC (port 8001):")
        print("     kubectl port-forward <triton-pod-name> 8001:8001 -n <namespace>")
        print("   For FastAPI Microservice (port 8000 or 8001):")
        print("     kubectl port-forward <app-pod-name> 8000:8000 -n <namespace>")
        print("=" * 80 + "\n")
        sys.exit(1)

    # 4. Display Results
    print("\n" + "=" * 80)
    print(f" INFERENCE RESULT (Latency: {latency_ms:.2f} ms | Threshold >= {args.conf})")
    print("=" * 80)
    print(f" Total Detections: {len(detections)}")

    for i, det in enumerate(detections, 1):
        print(f"  {i}. Label: {det['description']:<15} | Score: {det['score']:.4f} | BBox ({args.bbox_format}): {det['bbox']}")

    # 5. Draw & Save Annotations
    output_path = Path(args.output)
    draw_and_save_detections(pil_img, detections, output_path)
    print("=" * 80)
    print(f"✓ All tasks completed successfully. Output image saved to: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()

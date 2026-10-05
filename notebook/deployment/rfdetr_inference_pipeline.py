"""
================================================================================
RF-DETR Production Inference Engine (ONNX Runtime, TensorRT Engine / PLAN, Triton gRPC)
================================================================================
Universal Drop-in Inference Module for RF-DETR Multi-Class Object Detection.
Compatible with:
  • Model: RF-DETR (Backbone: DINOv2 / MultiScaleProjector, Resolution: 560x560)
  • Categories: ['blue_aisle', 'blue_bay', 'location_tag']
  • Triton Model Name: 'tagdet_rt' (Platform: 'tensorrt_plan' or Backend: 'onnxruntime')
  • Triton Input Binding:  'images' [1, 3, 640, 480] (FP32, NCHW, ImageNet normalized)
  • Triton Output Binding: 'output' [1, 18900, 7] (FP32: [cx, cy, w, h, s_0, s_1, s_2])

================================================================================
TRITON config.pbtxt REFERENCE SPECIFICATIONS:
================================================================================

--- TensorRT Engine (model.plan) ---
name: "tagdet_rt"
platform: "tensorrt_plan"
max_batch_size: 1

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 480 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 18900, 7 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

default_model_filename: "model.plan"

--- ONNX Runtime (rfdetr_model.onnx) ---
name: "tagdet_rt"
backend: "onnxruntime"
max_batch_size: 1

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 480 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 18900, 7 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

default_model_filename: "rfdetr_model.onnx"
================================================================================
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any, Union, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RFDetrInference")


# ==============================================================================
# 1. CORE PIPELINE CONSTANTS & DEFAULT CONFIGURATION
# ==============================================================================
CLASS_NAMES = ["blue_aisle", "blue_bay", "location_tag"]
NUM_CLASSES = len(CLASS_NAMES)

TRITON_MODEL_NAME = "tagdet_rt"
TRITON_INPUT_NAME = "images"
TRITON_OUTPUT_NAME = "output"

INPUT_HEIGHT = 640
INPUT_WIDTH = 480
TOTAL_OUTPUT_ROWS = 18900
OUTPUT_COLS = 7  # 4 bbox [cx, cy, w, h] + 3 class scores [s_0, s_1, s_2]

# ImageNet Preprocessing Parameters (Standard Vision Transformer / ResNet Norm)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==============================================================================
# 2. IMAGE PREPROCESSING
# ==============================================================================
def preprocess_image(
    image: Union[Image.Image, np.ndarray, str, Path],
    target_width: int = INPUT_WIDTH,
    target_height: int = INPUT_HEIGHT
) -> Tuple[np.ndarray, int, int]:
    """
    Strict Image Preprocessing conforming to tagdet_rt Triton input binding [1, 3, 640, 480]:
    1. Loads or converts input image into RGB PIL format.
    2. Records original dimensions (orig_w, orig_h) for coordinate rescale.
    3. Bilinearly resizes strictly to (target_width, target_height) -> (480, 640).
    4. Scales pixel values to [0.0, 1.0].
    5. Normalizes channels using ImageNet mean & standard deviation: (x - mean) / std.
    6. Transposes from HWC -> CHW and adds batch dimension -> NCHW [1, 3, 640, 480] FP32.

    Args:
        image: PIL Image, numpy array (HWC RGB/BGR), or file path.
        target_width: Width expected by the model (default: 480).
        target_height: Height expected by the model (default: 640).

    Returns:
        input_tensor: np.ndarray of shape [1, 3, target_height, target_width], dtype float32.
        orig_w: Original image width in pixels.
        orig_h: Original image height in pixels.
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(str(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 4:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 3:
            pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            raise ValueError(f"Unsupported numpy image array shape: {image.shape}")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    orig_w, orig_h = pil_img.size

    # Resize to target (width, height) using bilinear interpolation
    resized = pil_img.resize((target_width, target_height), Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0

    # Channel normalization: (x - mean) / std
    img_norm = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW -> NCHW [1, 3, target_height, target_width]
    input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    return input_tensor, orig_w, orig_h


# ==============================================================================
# 3. POST-PROCESSING & COORDINATE CALIBRATION
# ==============================================================================
def postprocess_output(
    raw_output: np.ndarray,
    orig_w: int,
    orig_h: int,
    conf_threshold: float = 0.71,
    classes: Optional[List[str]] = None,
    pixel_coordinates: bool = True,
    bbox_format: str = "ymin_xmin_ymax_xmax",
    filter_edge_crops: bool = True,
    edge_crop_margin: float = 0.005,
    enable_shrinkage: bool = True,
    shrink_factor: float = 0.96
) -> List[Dict[str, Any]]:
    """
    Production Post-Processing conforming to tagdet_rt unified output [1, 18900, 7]:
    1. Unpacks tensor: columns 0..3 are normalized bbox coords [cx, cy, w, h] in [0, 1].
       Columns 4..6 are class probability scores [s_0, s_1, s_2].
    2. Sigmoid normalization applied if scores are unnormalized logits.
    3. Finds best class ID and maximum score per prediction query.
    4. Filters out queries below conf_threshold and discards zero-padded rows (w <= 0 or h <= 0).
    5. Converts coordinates from [cx, cy, w, h] to [x1, y1, x2, y2].
    6. Edge Crop Guard: Filters out false positives clipped by camera frame boundary.
    7. Boundary Shrinkage: Contracts bounding boxes by 4% (0.96) to eliminate background noise.
    8. Formats output coordinates according to bbox_format (e.g. [ymin, xmin, ymax, xmax]).

    Returns:
        List of detection dictionaries:
        [
            {
                "description": "location_tag",
                "score": 0.942,
                "bbox": [ymin, xmin, ymax, xmax],
                "class_id": 2
            }, ...
        ]
    """
    class_list = classes or CLASS_NAMES

    # Ensure shape is [18900, 7]
    if raw_output.ndim == 3 and raw_output.shape[0] == 1:
        out = raw_output[0]
    else:
        out = raw_output

    boxes = out[:, :4]    # [18900, 4] -> cx, cy, w, h
    scores = out[:, 4:]   # [18900, num_classes]

    # Convert logits to probabilities if necessary
    if scores.min() < 0.0 or scores.max() > 1.0:
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))

    class_ids = np.argmax(scores, axis=-1)
    max_scores = np.max(scores, axis=-1)

    # Filter by confidence threshold
    keep_indices = np.where(max_scores >= conf_threshold)[0]
    detections: List[Dict[str, Any]] = []

    for idx in keep_indices:
        cid = int(class_ids[idx])
        score = float(max_scores[idx])
        cx, cy, w, h = boxes[idx]

        # Discard zero-padded rows from the 18900 tensor
        if w <= 0.0 or h <= 0.0 or score <= 0.0:
            continue

        # Convert cxcywh (normalized 0..1) to x1, y1, x2, y2 (normalized 0..1)
        x1 = max(0.0, float(cx - w / 2.0))
        y1 = max(0.0, float(cy - h / 2.0))
        x2 = min(1.0, float(cx + w / 2.0))
        y2 = min(1.0, float(cy + h / 2.0))

        # Edge-Cropped Filtering (reject tags that touch frame boundaries)
        if filter_edge_crops:
            touches_border = (
                x1 <= edge_crop_margin or
                y1 <= edge_crop_margin or
                x2 >= (1.0 - edge_crop_margin) or
                y2 >= (1.0 - edge_crop_margin)
            )
            if touches_border:
                continue

        # Border Shrinkage (contracts box by 4% to suppress background jitter)
        if enable_shrinkage and shrink_factor < 1.0:
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            shrunk_w = (x2 - x1) * shrink_factor
            shrunk_h = (y2 - y1) * shrink_factor
            x1 = max(0.0, mid_x - shrunk_w / 2.0)
            y1 = max(0.0, mid_y - shrunk_h / 2.0)
            x2 = min(1.0, mid_x + shrunk_w / 2.0)
            y2 = min(1.0, mid_y + shrunk_h / 2.0)

        # Scale to pixel coordinates if enabled
        if pixel_coordinates:
            px_x1 = round(x1 * orig_w, 2)
            px_y1 = round(y1 * orig_h, 2)
            px_x2 = round(x2 * orig_w, 2)
            px_y2 = round(y2 * orig_h, 2)

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
                formatted_box = [round(y1, 4), round(x1, 4), round(y2, 4), round(x2, 4)]
            elif bbox_format == "xyxy":
                formatted_box = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
            elif bbox_format == "xywh":
                formatted_box = [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)]
            else:
                formatted_box = [round(y1, 4), round(x1, 4), round(y2, 4), round(x2, 4)]

        class_name = class_list[cid] if cid < len(class_list) else f"class_{cid}"

        detections.append({
            "description": class_name,
            "score": round(score, 4),
            "bbox": formatted_box,
            "class_id": cid
        })

    # Sort descending by score
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


# ==============================================================================
# 4. INFERENCE BACKENDS (ONNX, TENSORRT PLAN, TRITON gRPC)
# ==============================================================================

class BaseRFDetrEngine:
    """Abstract interface for all RF-DETR inference backends."""
    def predict_image(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        conf_threshold: float = 0.71,
        **postprocess_kwargs
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class RFDetrONNXEngine(BaseRFDetrEngine):
    """
    ONNX Runtime Inference Engine.
    Executes 'rfdetr_model.onnx' strictly verifying [1, 3, 640, 480] -> [1, 18900, 7].
    Compatible with onnxruntime==1.16.1.
    """
    def __init__(
        self,
        onnx_model_path: Union[str, Path],
        providers: Optional[List[str]] = None
    ):
        import onnxruntime as ort

        self.model_path = Path(onnx_model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model missing at: {self.model_path}")

        chosen_providers = providers or (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(self.model_path), providers=chosen_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(f"Loaded ONNX Model: {self.model_path} (Providers: {chosen_providers})")
        logger.info(f"Bindings: Input '{self.input_name}', Output '{self.output_name}'")

    def predict_image(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        conf_threshold: float = 0.71,
        **postprocess_kwargs
    ) -> List[Dict[str, Any]]:
        tensor, orig_w, orig_h = preprocess_image(image)
        raw_output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return postprocess_output(raw_output, orig_w, orig_h, conf_threshold=conf_threshold, **postprocess_kwargs)


class RFDetrTensorRTEngine(BaseRFDetrEngine):
    """
    TensorRT Standalone Engine (.plan / .engine).
    Direct GPU memory execution via TensorRT Python runtime and pycuda/torch.
    """
    def __init__(self, plan_path: Union[str, Path]):
        self.plan_path = Path(plan_path)
        if not self.plan_path.exists():
            raise FileNotFoundError(f"TensorRT plan missing at: {self.plan_path}")

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            self.trt = trt
            self.cuda = cuda
        except ImportError as e:
            raise ImportError("TensorRT execution requires 'tensorrt' and 'pycuda' packages.") from e

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(str(self.plan_path), "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate host/device buffers
        self.stream = cuda.Stream()
        self.h_input = cuda.pagelocked_empty((1, 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty((1, TOTAL_OUTPUT_ROWS, OUTPUT_COLS), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]
        logger.info(f"Loaded TensorRT Engine: {self.plan_path}")

    def predict_image(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        conf_threshold: float = 0.71,
        **postprocess_kwargs
    ) -> List[Dict[str, Any]]:
        tensor, orig_w, orig_h = preprocess_image(image)
        np.copyto(self.h_input, tensor)

        # Asynchronous Host-to-Device transfer, inference execution, and Device-to-Host transfer
        self.cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return postprocess_output(self.h_output, orig_w, orig_h, conf_threshold=conf_threshold, **postprocess_kwargs)


class RFDetrTritonEngine(BaseRFDetrEngine):
    """
    Production Triton gRPC Client for tagdet_rt.
    Calls Triton server over gRPC (port 8001) for remote model.plan or onnxruntime execution.
    """
    def __init__(
        self,
        triton_url: str = "localhost:8001",
        model_name: str = TRITON_MODEL_NAME,
        model_version: str = "1",
        input_name: str = TRITON_INPUT_NAME,
        output_name: str = TRITON_OUTPUT_NAME
    ):
        try:
            import tritonclient.grpc as grpcclient
            from tritonclient.grpc import InferInput, InferRequestedOutput
            self.grpcclient = grpcclient
            self.InferInput = InferInput
            self.InferRequestedOutput = InferRequestedOutput
        except ImportError as e:
            raise ImportError("Triton inference requires 'tritonclient[grpc]' package.") from e

        self.client = grpcclient.InferenceServerClient(url=triton_url)
        self.model_name = model_name
        self.model_version = str(model_version)
        self.input_name = input_name
        self.output_name = output_name

        if not self.client.is_server_live():
            raise ConnectionError(f"Triton server at {triton_url} is not responding (not live).")
        logger.info(f"Connected to Triton server: {triton_url} (Model: '{self.model_name}', v{self.model_version})")

    def predict_image(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        conf_threshold: float = 0.71,
        **postprocess_kwargs
    ) -> List[Dict[str, Any]]:
        tensor, orig_w, orig_h = preprocess_image(image)

        infer_input = self.InferInput(self.input_name, tensor.shape, "FP32")
        infer_input.set_data_from_numpy(tensor)
        infer_output = self.InferRequestedOutput(self.output_name)

        response = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[infer_input],
            outputs=[infer_output]
        )
        raw_output = response.as_numpy(self.output_name)
        return postprocess_output(raw_output, orig_w, orig_h, conf_threshold=conf_threshold, **postprocess_kwargs)


# ==============================================================================
# 5. CONVENIENCE FACTORY & STANDALONE CLI DEMO
# ==============================================================================
def load_rfdetr_detector(
    backend: str = "onnx",
    model_path: Optional[Union[str, Path]] = None,
    triton_url: str = "localhost:8001",
    model_version: str = "1"
) -> BaseRFDetrEngine:
    """
    Factory helper to instantiate the appropriate RF-DETR engine.

    Args:
        backend: 'onnx', 'plan' / 'tensorrt', or 'triton'.
        model_path: Path to .onnx file (for onnx) or .plan file (for tensorrt).
        triton_url: Triton gRPC endpoint host:port (for triton).
        model_version: Triton model version string.
    """
    backend = backend.lower().strip()
    if backend == "onnx":
        p = model_path or "deployment/models/rfdetr_model.onnx"
        return RFDetrONNXEngine(onnx_model_path=p)
    elif backend in ("plan", "tensorrt", "trt"):
        p = model_path or "deployment/models/model.plan"
        return RFDetrTensorRTEngine(plan_path=p)
    elif backend == "triton":
        return RFDetrTritonEngine(triton_url=triton_url, model_version=model_version)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: 'onnx', 'plan', 'triton'.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RF-DETR Production Inference Verification")
    parser.add_argument("--backend", default="onnx", choices=["onnx", "plan", "triton"], help="Inference backend")
    parser.add_argument("--model-path", default="deployment/models/rfdetr_model.onnx", help="Path to ONNX or PLAN model file")
    parser.add_argument("--triton-url", default="localhost:8001", help="Triton gRPC endpoint")
    parser.add_argument("--image", default=None, help="Input test image path")
    parser.add_argument("--conf", type=float, default=0.71, help="Confidence threshold")
    args = parser.parse_args()

    print("=" * 80)
    print(" RF-DETR Production Multi-Class Inference Verification")
    print(f" • Backend:    {args.backend}")
    print(f" • Model:      {args.model_path if args.backend != 'triton' else args.triton_url}")
    print(f" • Classes:    {CLASS_NAMES}")
    print(f" • Threshold:  {args.conf}")
    print("=" * 80)

    # Instantiate detector
    detector = load_rfdetr_detector(
        backend=args.backend,
        model_path=args.model_path,
        triton_url=args.triton_url
    )

    # Use user image or create dummy verification image
    if args.image and Path(args.image).exists():
        test_img = Image.open(args.image).convert("RGB")
        print(f"Loaded image: {args.image} ({test_img.size[0]}x{test_img.size[1]})")
    else:
        print("Creating synthetic test image (1000x750)...")
        test_img = Image.new("RGB", (1000, 750), color=(240, 240, 240))

    t0 = time.time()
    results = detector.predict_image(test_img, conf_threshold=args.conf)
    latency_ms = (time.time() - t0) * 1000

    print(f"\nInference completed in {latency_ms:.2f} ms. Detected {len(results)} objects:")
    for i, det in enumerate(results[:10], 1):
        print(f"  {i}. [{det['description']}] Score: {det['score']:.3f} | BBox (ymin, xmin, ymax, xmax): {det['bbox']}")
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more detections.")

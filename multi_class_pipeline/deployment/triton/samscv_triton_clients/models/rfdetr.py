"""
models/rfdetr.py: Production Triton Client for RF-DETR Object Detection.
Part of SamsCV-Triton-Clients repository.

Implements preprocessing, Triton gRPC inference, and postprocessing
to output standard {"description", "score", "bbox"} dictionaries.
"""

from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np
from PIL import Image

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.grpc import InferInput, InferRequestedOutput
except ImportError:
    grpcclient = None
    InferInput = None
    InferRequestedOutput = None

# Mockable base TritonClient if importing standalone
try:
    from samscvtritonclients.client import TritonClient
except ImportError:
    class TritonClient:
        def __init__(self, url: str, model_name: str, model_version: str = "", **kwargs):
            self.url = url
            self.model_name = model_name
            self.model_version = str(model_version) if model_version else ""
            self.client = grpcclient.InferenceServerClient(url=url) if grpcclient else None

        def is_model_ready(self) -> bool:
            if not self.client:
                return False
            return self.client.is_model_ready(self.model_name, self.model_version)


class RFDetrTritonClient(TritonClient):
    """
    Triton Client for RF-DETR Multi-Class Object Detection Model.

    Inputs:
        images: FP32 tensor [batch_size, 3, resolution, resolution]
    Outputs:
        scores: FP32 tensor [batch_size, 300, num_classes]
        boxes:  FP32 tensor [batch_size, 300, 4] (cx, cy, w, h normalized [0, 1])

    Returns:
        List of detection dicts:
        [
            {
                "description": "location_tag",
                "score": 0.942,
                "bbox": [ymin, xmin, ymax, xmax]  # normalized or pixel coordinates
            }, ...
        ]
    """

    DEFAULT_CLASSES = ["blue_aisle", "blue_bay", "location_tag"]
    DEFAULT_RESOLUTION = 560
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        url: str,
        model_name: str = "rfdetr",
        model_version: str = "2",
        classes: Optional[List[str]] = None,
        resolution: int = DEFAULT_RESOLUTION,
        conf_threshold: float = 0.50,
        input_name: str = "images",
        scores_output_name: str = "scores",
        boxes_output_name: str = "boxes",
        bbox_format: str = "ymin_xmin_ymax_xmax",  # standard SamsCV convention
        pixel_coordinates: bool = True,
        shrink_factor: float = 0.96,               # 4% boundary shrinkage
        enable_shrinkage: bool = True,
        **kwargs
    ):
        super().__init__(url=url, model_name=model_name, model_version=model_version, **kwargs)
        self.classes = classes or self.DEFAULT_CLASSES
        self.resolution = resolution
        self.conf_threshold = conf_threshold
        self.input_name = input_name
        self.scores_output_name = scores_output_name
        self.boxes_output_name = boxes_output_name
        self.bbox_format = bbox_format
        self.pixel_coordinates = pixel_coordinates
        self.shrink_factor = shrink_factor
        self.enable_shrinkage = enable_shrinkage

    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> Tuple[np.ndarray, int, int]:
        """
        Step 5: Preprocess input image to match RF-DETR training:
        1. RGB conversion & size extraction
        2. Bilinear resize to resolution x resolution
        3. Normalization (ImageNet mean & std)
        4. Transpose to NCHW [1, 3, H, W] float32
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_img = Image.fromarray(image).convert("RGB")
            elif image.shape[2] == 4:
                pil_img = Image.fromarray(image).convert("RGB")
            else:
                pil_img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        orig_w, orig_h = pil_img.size

        # Resize
        resized = pil_img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img_np = np.array(resized, dtype=np.float32) / 255.0

        # Normalize: (img - mean) / std
        img_norm = (img_np - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # HWC -> CHW -> NCHW [1, 3, H, W]
        input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return input_tensor, orig_w, orig_h

    def infer(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 6: Send tensor using exact input name and call self.client.infer()
        """
        if not self.client:
            raise RuntimeError("Triton gRPC client is not initialized.")

        inputs = [InferInput(self.input_name, input_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_tensor)

        outputs = [
            InferRequestedOutput(self.scores_output_name),
            InferRequestedOutput(self.boxes_output_name)
        ]

        resp = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs
        )

        scores = resp.as_numpy(self.scores_output_name)
        boxes = resp.as_numpy(self.boxes_output_name)
        return scores, boxes

    def postprocess(
        self,
        scores: np.ndarray,
        boxes: np.ndarray,
        orig_w: int,
        orig_h: int
    ) -> List[Dict[str, Any]]:
        """
        Step 7: Postprocess raw outputs:
        1. Sigmoid if logits
        2. Argmax across class dimension
        3. Filter by conf_threshold
        4. Convert cxcywh -> target bbox format [ymin, xmin, ymax, xmax]
        5. Apply boundary-box shrinkage (4%) to match warehouse annotations
        6. Return standard [{"description", "score", "bbox"}, ...]
        """
        batch_scores = scores[0]  # [300, num_classes]
        batch_boxes = boxes[0]    # [300, 4]

        # Apply sigmoid if scores appear to be unnormalized logits
        if batch_scores.min() < 0.0 or batch_scores.max() > 1.0:
            batch_scores = 1.0 / (1.0 + np.exp(-batch_scores))

        class_ids = np.argmax(batch_scores, axis=-1)
        max_scores = np.max(batch_scores, axis=-1)

        keep_indices = np.where(max_scores >= self.conf_threshold)[0]
        results = []

        for idx in keep_indices:
            cid = int(class_ids[idx])
            score = float(max_scores[idx])
            cx, cy, w, h = batch_boxes[idx]

            # Box conversion: cxcywh (0-1) -> x1, y1, x2, y2
            x1 = max(0.0, float(cx - w / 2.0))
            y1 = max(0.0, float(cy - h / 2.0))
            x2 = min(1.0, float(cx + w / 2.0))
            y2 = min(1.0, float(cy + h / 2.0))

            # Boundary-Box Shrinkage (contracts box by 4% to suppress background jitter)
            if self.enable_shrinkage and self.shrink_factor < 1.0:
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0
                shrunk_w = (x2 - x1) * self.shrink_factor
                shrunk_h = (y2 - y1) * self.shrink_factor
                x1 = max(0.0, mid_x - shrunk_w / 2.0)
                y1 = max(0.0, mid_y - shrunk_h / 2.0)
                x2 = min(1.0, mid_x + shrunk_w / 2.0)
                y2 = min(1.0, mid_y + shrunk_h / 2.0)

            # Coordinate scaling
            if self.pixel_coordinates:
                px1 = round(x1 * orig_w, 2)
                py1 = round(y1 * orig_h, 2)
                px2 = round(x2 * orig_w, 2)
                py2 = round(y2 * orig_h, 2)
            else:
                px1, py1, px2, py2 = round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)

            # Format bbox according to convention
            if self.bbox_format == "ymin_xmin_ymax_xmax":
                bbox = [py1, px1, py2, px2]
            else:
                bbox = [px1, py1, px2, py2]

            label = self.classes[cid] if cid < len(self.classes) else f"class_{cid}"

            results.append({
                "description": label,
                "score": round(score, 4),
                "bbox": bbox
            })

        # Sort detections by confidence descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def predict(self, image: Union[Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Complete end-to-end inference pipeline:
        preprocess -> infer -> postprocess -> detections
        """
        input_tensor, orig_w, orig_h = self.preprocess(image)
        scores, boxes = self.infer(input_tensor)
        return self.postprocess(scores, boxes, orig_w, orig_h)

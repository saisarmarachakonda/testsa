"""
models/rfdetr.py: Production Triton Client for RF-DETR Object Detection.
Part of SamsCV-Triton-Clients repository.

Supports both:
1. Production unified tensor layout (tagdet_rt):
   - Input:  images [1, 3, 640, 480]
   - Output: output [1, 18900, 7] ([cx, cy, w, h, s_0, s_1, s_2])
2. Decoupled tensors layout:
   - Input:  images [B, 3, H, W]
   - Output: scores [B, 300, C], boxes [B, 300, 4]
"""

import os
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
        images: FP32 tensor [1, 3, 640, 480] (or [batch, 3, H, W])
    Outputs:
        output: FP32 tensor [1, 18900, 7] (or decoupled scores [300, C], boxes [300, 4])

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

    # Categories discovered in multi_class_train_rfdetr.ipynb (contiguously 0-indexed)
    DEFAULT_CLASSES = ["blue_aisle", "blue_bay", "location_tag"]
    DEFAULT_MODEL_NAME = os.getenv("RFDETR_MODEL_NAME", "tagdet_rt")
    DEFAULT_INPUT_HEIGHT = int(os.getenv("RFDETR_INPUT_HEIGHT", "640"))
    DEFAULT_INPUT_WIDTH = int(os.getenv("RFDETR_INPUT_WIDTH", "480"))
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        url: str,
        model_name: str = DEFAULT_MODEL_NAME,
        model_version: str = "2",
        classes: Optional[List[str]] = None,
        input_height: int = DEFAULT_INPUT_HEIGHT,
        input_width: int = DEFAULT_INPUT_WIDTH,
        conf_threshold: float = 0.50,
        input_name: str = "images",
        output_name: str = "output",
        scores_output_name: Optional[str] = None,
        boxes_output_name: Optional[str] = None,
        bbox_format: str = "ymin_xmin_ymax_xmax",  # standard SamsCV convention
        pixel_coordinates: bool = True,
        shrink_factor: float = 0.96,               # 4% boundary shrinkage
        enable_shrinkage: bool = True,
        filter_edge_crops: bool = True,            # Automatically drop detections touching frame borders
        edge_crop_margin: float = 0.005,           # Border margin threshold (0.5% of frame)
        **kwargs
    ):
        super().__init__(url=url, model_name=model_name, model_version=model_version, **kwargs)
        self.classes = classes or self.DEFAULT_CLASSES
        self.input_height = input_height
        self.input_width = input_width
        self.conf_threshold = conf_threshold
        self.input_name = input_name
        self.output_name = output_name
        self.scores_output_name = scores_output_name
        self.boxes_output_name = boxes_output_name
        self.bbox_format = bbox_format
        self.pixel_coordinates = pixel_coordinates
        self.shrink_factor = shrink_factor
        self.enable_shrinkage = enable_shrinkage
        self.filter_edge_crops = filter_edge_crops
        self.edge_crop_margin = edge_crop_margin

    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> Tuple[np.ndarray, int, int]:
        """
        Step 5: Preprocess input image to match tagdet_rt Triton input binding [1, 3, 640, 480]:
        1. RGB conversion & size extraction
        2. Bilinear resize to (input_width, input_height) -> (480, 640)
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

        # Resize to (width, height)
        resized = pil_img.resize((self.input_width, self.input_height), Image.BILINEAR)
        img_np = np.array(resized, dtype=np.float32) / 255.0

        # Normalize: (img - mean) / std
        img_norm = (img_np - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # HWC -> CHW -> NCHW [1, 3, H, W]
        input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return input_tensor, orig_w, orig_h

    def infer(self, input_tensor: np.ndarray) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Step 6: Send tensor using exact input name and call self.client.infer()
        Supports both single output tensor [1, 18900, 7] and decoupled (scores, boxes).
        """
        if not self.client:
            raise RuntimeError("Triton gRPC client is not initialized.")

        inputs = [InferInput(self.input_name, input_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_tensor)

        # Mode A: Decoupled scores and boxes
        if self.scores_output_name and self.boxes_output_name:
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

        # Mode B: Production tagdet_rt single tensor "output" [1, 18900, 7]
        outputs = [InferRequestedOutput(self.output_name)]
        resp = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs
        )
        return resp.as_numpy(self.output_name)

    def postprocess(
        self,
        inference_output: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
        orig_w: int,
        orig_h: int
    ) -> List[Dict[str, Any]]:
        """
        Step 7: Postprocess raw outputs:
        - If tuple (scores, boxes): unpacks directly.
        - If single array [1, 18900, 7]: splits columns 0..3 (boxes) and 4..6 (scores).
        Filters by conf_threshold, applies 4% boundary shrinkage, converts to bbox format.
        """
        if isinstance(inference_output, tuple):
            scores, boxes = inference_output
            batch_scores = scores[0]  # [N, num_classes]
            batch_boxes = boxes[0]    # [N, 4]
        else:
            raw_out = inference_output[0]  # [18900, 7]
            batch_boxes = raw_out[:, :4]   # [18900, 4] cx, cy, w, h
            batch_scores = raw_out[:, 4:]  # [18900, num_classes]

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

            # Discard padding or invalid boxes
            if w <= 0.0 or h <= 0.0 or score <= 0.0:
                continue

            # Box conversion: cxcywh (0-1) -> x1, y1, x2, y2
            x1 = max(0.0, float(cx - w / 2.0))
            y1 = max(0.0, float(cy - h / 2.0))
            x2 = min(1.0, float(cx + w / 2.0))
            y2 = min(1.0, float(cy + h / 2.0))

            # Edge-Cropped Guard: Automatically drop boxes clipped by or touching the camera frame borders
            if self.filter_edge_crops:
                is_touching_border = (x1 <= self.edge_crop_margin) or                                      (y1 <= self.edge_crop_margin) or                                      (x2 >= (1.0 - self.edge_crop_margin)) or                                      (y2 >= (1.0 - self.edge_crop_margin))
                if is_touching_border:
                    continue

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
        raw_output = self.infer(input_tensor)
        return self.postprocess(raw_output, orig_w, orig_h)

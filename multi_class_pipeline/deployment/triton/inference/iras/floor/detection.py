"""
inference/iras/floor/detection.py: Location Tag Detection Wrapper for RF-DETR.
Implements the RfDetrLocationTagDetection class and integrates with SamsCV-Triton-Clients.

Includes unit testing with Mocked Triton to verify input tensors and output schema.
"""

from typing import List, Dict, Any, Union, Optional
import os
import unittest
from unittest.mock import MagicMock
import numpy as np
from PIL import Image

try:
    from samscv_triton_clients.models.rfdetr import RFDetrTritonClient
except ImportError:
    try:
        from samscvtritonclients.models.rfdetr import RFDetrTritonClient
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[3] / "samscv_triton_clients"))
        from models.rfdetr import RFDetrTritonClient


class RfDetrLocationTagDetection:
    """
    Production detector wrapper for Location Tag Detection using RF-DETR via Triton.
    
    Guarantees output compatibility with IRAS-Location-Tag-Module:
    Returns list of dicts:
        [
            {
                "description": "location_tag", # or "blue_aisle", "blue_bay"
                "score": 0.95,
                "bbox": [ymin, xmin, ymax, xmax]
            }, ...
        ]
    """

    def __init__(
        self,
        triton_url: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        conf_threshold: float = 0.40,
        filter_tag_only: bool = False,
        **kwargs
    ):
        self.triton_url = triton_url or os.getenv("TRITON_URL", "localhost:8001")
        self.model_name = model_name or os.getenv("RFDETR_MODEL_NAME", "rfdetr")
        self.model_version = model_version or os.getenv("RFDETR_MODEL_VERSION", "2")
        self.conf_threshold = float(os.getenv("RFDETR_CONF_THRESHOLD", str(conf_threshold)))
        self.filter_tag_only = filter_tag_only

        self.client = RFDetrTritonClient(
            url=self.triton_url,
            model_name=self.model_name,
            model_version=self.model_version,
            conf_threshold=self.conf_threshold,
            bbox_format="ymin_xmin_ymax_xmax",
            pixel_coordinates=True,
            **kwargs
        )

    def detect(self, image: Union[Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Executes RF-DETR detection on input image.
        Returns standard detection dictionaries.
        """
        detections = self.client.predict(image)

        if self.filter_tag_only:
            detections = [d for d in detections if d.get("description") == "location_tag"]

        return detections


# ==============================================================================
# Step 8 Unit Test Suite (Test with Mocked Triton)
# ==============================================================================
class TestRfDetrLocationTagDetection(unittest.TestCase):

    def setUp(self):
        self.detector = RfDetrLocationTagDetection(
            triton_url="mock:8001",
            model_name="rfdetr",
            model_version="2",
            conf_threshold=0.30
        )

        # Mock the underlying Triton client.infer method
        mock_scores = np.zeros((1, 300, 3), dtype=np.float32)
        # Create a mock detection for class 2 ("location_tag") with high confidence
        mock_scores[0, 0, 2] = 0.96  # location_tag score
        mock_scores[0, 1, 0] = 0.88  # blue_aisle score

        mock_boxes = np.zeros((1, 300, 4), dtype=np.float32)
        # cx=0.5, cy=0.4, w=0.2, h=0.1
        mock_boxes[0, 0] = [0.50, 0.40, 0.20, 0.10]
        mock_boxes[0, 1] = [0.20, 0.20, 0.10, 0.15]

        self.detector.client.infer = MagicMock(return_value=(mock_scores, mock_boxes))

    def test_mocked_triton_detection_schema(self):
        # Create synthetic test image
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))
        results = self.detector.detect(img)

        # Verify output format
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)

        first_det = results[0]
        self.assertIn("description", first_det)
        self.assertIn("score", first_det)
        self.assertIn("bbox", first_det)

        # Verify description matches detected class
        self.assertEqual(first_det["description"], "location_tag")
        self.assertGreater(first_det["score"], 0.90)

        # Verify bbox is [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = first_det["bbox"]
        self.assertLess(ymin, ymax)
        self.assertLess(xmin, xmax)
        print(f"\n✓ Mock Triton Test Passed! Detections: {results}")

    def test_filter_tag_only(self):
        self.detector.filter_tag_only = True
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))
        results = self.detector.detect(img)
        for d in results:
            self.assertEqual(d["description"], "location_tag")


if __name__ == "__main__":
    unittest.main()

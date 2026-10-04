"""
tag_detection.py: Location Tag Detection Backend Switcher with Canary Club Rollout.
Part of IRAS-Location-Tag-Module repository.

Implements feature-flagged routing between legacy detector and RF-DETR Triton backend
using the RELEASE_CLUB_LIST canary pattern.
"""

import os
import logging
from typing import List, Dict, Any, Union, Optional
from PIL import Image

logger = logging.getLogger("IRAS.LocationTagModule")

# ==============================================================================
# Configuration Flags & Environment Variables
# ==============================================================================
# 1. Primary Feature Flag to enable RF-DETR
USE_RFDETR_TAGDET = os.getenv("USE_RFDETR_TAGDET", "false").lower() in ("true", "1", "yes")

# 2. Triton Model Name & Version
RFDETR_MODEL_NAME = os.getenv("RFDETR_MODEL_NAME", "tagdet_rt")
RFDETR_MODEL_VERSION = os.getenv("RFDETR_MODEL_VERSION", "2")
TRITON_URL = os.getenv("TRITON_URL", "localhost:8001")

# 3. Canary Club Rollout Pattern (e.g. "club_101,club_102,club_105")
# When set, only clubs in this list will use RF-DETR even if USE_RFDETR_TAGDET is true
RELEASE_CLUB_LIST = [
    c.strip() for c in os.getenv("RELEASE_CLUB_LIST", "").split(",") if c.strip()
]


# ==============================================================================
# Detector Initialization
# ==============================================================================
class LegacyTagDetector:
    """Mock/Placeholder for existing legacy detector backend."""
    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        # Existing legacy model detection logic
        return [{"description": "location_tag", "score": 0.85, "bbox": [100.0, 100.0, 200.0, 300.0]}]


# Instantiate Detectors (Lazy or Eager)
_legacy_detector = None
_rfdetr_detector = None

def get_legacy_detector():
    global _legacy_detector
    if _legacy_detector is None:
        _legacy_detector = LegacyTagDetector()
    return _legacy_detector

def get_rfdetr_detector():
    global _rfdetr_detector
    if _rfdetr_detector is None:
        try:
            from samscvtritonclients.models.rfdetr import RFDetrTritonClient
            from inference.iras.floor.detection import RfDetrLocationTagDetection
        except ImportError:
            # Fallback to local import path
            import sys
            from pathlib import Path
            base = Path(__file__).resolve().parents[1]
            sys.path.append(str(base / "samscv_triton_clients"))
            sys.path.append(str(base))
            from inference.iras.floor.detection import RfDetrLocationTagDetection

        logger.info(f"Initializing RF-DETR Triton Detector ({RFDETR_MODEL_NAME} v{RFDETR_MODEL_VERSION} on {TRITON_URL})...")
        _rfdetr_detector = RfDetrLocationTagDetection(
            triton_url=TRITON_URL,
            model_name=RFDETR_MODEL_NAME,
            model_version=RFDETR_MODEL_VERSION,
            filter_tag_only=False
        )
    return _rfdetr_detector


def should_use_rfdetr(club_id: Optional[str] = None) -> bool:
    """
    Step 9 & 10: Determines whether to route request to RF-DETR.
    1. If USE_RFDETR_TAGDET is False -> False
    2. If RELEASE_CLUB_LIST is populated -> True only if club_id in RELEASE_CLUB_LIST
    3. If RELEASE_CLUB_LIST is empty and USE_RFDETR_TAGDET is True -> True (Full Rollout)
    """
    if not USE_RFDETR_TAGDET:
        return False

    if RELEASE_CLUB_LIST:
        if not club_id:
            return False
        return str(club_id).strip() in RELEASE_CLUB_LIST

    # 100% rollout enabled
    return True


def detect_location_tags(
    image: Image.Image,
    club_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Main entry point for Location Tag Detection in IRAS-Location-Tag-Module.
    
    Routes seamlessly between RF-DETR and Legacy detector:
    - Verifies club rollout eligibility
    - Returns standardized list of {"description", "score", "bbox"}
    """
    use_rfdetr = should_use_rfdetr(club_id)

    if use_rfdetr:
        try:
            logger.info(f"Executing RF-DETR detection (club_id={club_id})...")
            detector = get_rfdetr_detector()
            return detector.detect(image)
        except Exception as e:
            logger.error(f"RF-DETR Triton inference failed ({e}). Falling back to legacy detector.")
            # Fallback guarantee to prevent service disruption
            return get_legacy_detector().detect(image)
    else:
        logger.debug(f"Routing to legacy detector (club_id={club_id})...")
        return get_legacy_detector().detect(image)

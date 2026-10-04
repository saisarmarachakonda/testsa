#!/usr/bin/env python3
"""
COCO File & JSON Validator Utility
----------------------------------
Validates that files in `coco_files/` (or any specified directory) are:
1. Valid JSON files (.json extension).
2. Syntactically valid JSON (parsable without errors).
3. Fully compliant with the COCO annotation format (contains 'images', 'annotations', 'categories').
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("COCOValidator")


def validate_single_file(file_path: Path) -> Dict[str, Any]:
    """Inspects a single file and determines if it is a valid COCO JSON."""
    result = {
        "file_name": file_path.name,
        "path": str(file_path),
        "is_json_extension": file_path.suffix.lower() == ".json",
        "file_size_bytes": file_path.stat().st_size,
        "is_valid_json": False,
        "is_coco_schema": False,
        "images_count": 0,
        "annotations_count": 0,
        "categories_count": 0,
        "categories_list": [],
        "errors": []
    }

    if not result["is_json_extension"]:
        result["errors"].append(f"Not a .json file (found extension: '{file_path.suffix}')")
        return result

    if result["file_size_bytes"] == 0:
        result["errors"].append("File is completely empty (0 bytes).")
        return result

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result["is_valid_json"] = True
    except json.JSONDecodeError as e:
        result["errors"].append(f"JSON syntax decode error: {e}")
        return result
    except Exception as e:
        result["errors"].append(f"Could not read file: {e}")
        return result

    if not isinstance(data, dict):
        result["errors"].append(f"Root JSON entity must be a dictionary/object, got: {type(data).__name__}")
        return result

    # Check required COCO top-level keys
    required_keys = ["images", "annotations", "categories"]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        result["errors"].append(f"Missing standard COCO keys: {missing_keys}")
    else:
        result["is_coco_schema"] = True

    # Image records
    images = data.get("images", [])
    if isinstance(images, list):
        result["images_count"] = len(images)
    else:
        result["errors"].append("'images' key is not a list")

    # Annotations records
    annotations = data.get("annotations", [])
    if isinstance(annotations, list):
        result["annotations_count"] = len(annotations)
    else:
        result["errors"].append("'annotations' key is not a list")

    # Categories records
    categories = data.get("categories", [])
    if isinstance(categories, list):
        result["categories_count"] = len(categories)
        result["categories_list"] = [c.get("name", str(c.get("id"))) for c in categories if isinstance(c, dict)]
    else:
        result["errors"].append("'categories' key is not a list")

    return result


def audit_coco_directory(target_dir: Path) -> List[Dict[str, Any]]:
    """Recursively checks all files in the target directory."""
    if not target_dir.exists():
        logger.error(f"Target directory does not exist: {target_dir}")
        return []

    all_files = [p for p in target_dir.glob("**/*") if p.is_file() and not p.name.startswith(".")]
    logger.info(f"Auditing directory: {target_dir} ({len(all_files)} total files found)")

    reports = []
    for f in all_files:
        # Ignore image files in images/ folder when auditing annotation files
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            continue
        report = validate_single_file(f)
        reports.append(report)

    return reports


def print_validation_summary(reports: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("                COCO JSON VALIDATION REPORT")
    print("=" * 80)

    if not reports:
        print("No annotation files found in directory.")
        print("Note: To ingest COCO datasets, run: multi_class_pipeline/00_download_coco_files.ipynb")
        print("=" * 80 + "\n")
        return

    for r in reports:
        status_icon = "✓" if (r["is_valid_json"] and r["is_coco_schema"]) else "✗"
        print(f"\n[{status_icon}] File: {r['file_name']}")
        print(f"    Path:            {r['path']}")
        print(f"    Size:            {r['file_size_bytes'] / 1024:.2f} KB")
        print(f"    JSON Extension:  {'Yes' if r['is_json_extension'] else 'No'}")
        print(f"    Valid JSON:      {'Yes' if r['is_valid_json'] else 'No'}")
        print(f"    COCO Compliant:  {'Yes' if r['is_coco_schema'] else 'No'}")
        if r["is_coco_schema"]:
            print(f"    Images Count:    {r['images_count']}")
            print(f"    Annotations:     {r['annotations_count']}")
            print(f"    Categories ({r['categories_count']}): {r['categories_list']}")
        if r["errors"]:
            print(f"    Errors/Warnings: {'; '.join(r['errors'])}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    target = repo_root / "coco_files"
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).resolve()

    reports = audit_coco_directory(target)
    print_validation_summary(reports)

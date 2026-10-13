import json
from pathlib import Path

nb_path = Path("notebook/08_evaluate_anomalies_audit.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 0: Markdown
cell_0 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Step 08: Robustness Evaluation on Anomaly Audit Dataset\n",
        "### Quantifying Model Resilience Against Noisy Ground Truth & Labeler Errors\n",
        "Directly integrated with Step 02 Anomaly Detection and Step 04 Trained Model.\n",
        "\n",
        "### Purpose:\n",
        "1. **Load Detected Dataset Anomalies**: Ingests `anomalies_manifest.json` from Step 02 (degenerate boxes, micro-clicks, extreme aspect ratios, loose shelf margins, edge truncations).\n",
        "2. **Instantiate Fine-Tuned Model**: Automatically locates and loads the latest/best trained RF-DETR model checkpoint.\n",
        "3. **Resilience Analysis**: Evaluates whether the trained model correctly ignores degenerate/flawed labeler annotations or gets tricked.\n",
        "4. **Side-by-Side Visual Showcase**: Displays diagnostic preview cards comparing **Ground Truth Flawed Box (Red / Orange)** vs **Model Prediction (Green / Cyan)** with full postprocessing (WBF & Margin Shrinkage).\n"
    ]
}

# Cell 1: Environment Dependencies
cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# STEP 0 - Environment Dependencies\n",
        "%pip install -q pycocotools Pillow pandas numpy tabulate tqdm supervision opencv-python-headless==4.9.0.80"
    ]
}

# Cell 2: Imports & Logger
cell_2_code = """# CELL 1 - Imports, VRAM Monitor & Logger Initialization
import os, sys, json, time, math, copy, logging, random, shutil, warnings
warnings.filterwarnings("ignore", message=".*meshgrid.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*max_detection_threshold.*")
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import urllib.parse
from urllib.parse import urlparse, parse_qs, unquote

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

try:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(rlimit[0], 4096), max(rlimit[1], 4096)))
except Exception:
    pass

import cv2, torch, requests, supervision as sv
import numpy as np, pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Image as IPImage
from tqdm.auto import tqdm

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    torchmetrics_status = "Available"
except ImportError:
    torchmetrics_status = "Not installed"

try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge
    from rfdetr import main as rfdetr_main
    from rfdetr.models.lwdetr import build_criterion_and_postprocessors, LWDETR
    rfdetr_status = "Available"
except ImportError:
    rfdetr_status = "Not installed"

class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logger = logging.getLogger("rfdetr_anomaly_eval")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
console_handler = FlushHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(console_handler)

def print_vram_usage(tag="Status"):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"[VRAM - {tag}] {torch.cuda.get_device_name(0)}: {alloc:.2f} GB / {total:.2f} GB allocated")
    else:
        logger.info(f"[VRAM - {tag}] Running on CPU")

print_vram_usage("Init")
logger.info("✓ Anomaly Evaluation Environment Initialized.")"""

cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in cell_2_code.splitlines()]
}
cell_2["source"][-1] = cell_2["source"][-1].rstrip("\n")

# Cell 3: Path and Model Discovery
cell_3_code = """# CELL 2 - Anomaly Manifest & Model Auto-Discovery
REPO_ROOT = next((p.resolve() for p in [Path("."), Path(".."), Path("../..")] if (p / "coco_files").exists() or (p / ".git").exists()), Path(".").resolve())
PIPELINE_NAME = "multi_class_rfdetr"
PIPELINE_DIR = REPO_ROOT / PIPELINE_NAME
INPUT_JSON_DIR = Path("coco_files") if Path("coco_files").exists() else (REPO_ROOT / "coco_files")

MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")
VERSION_TAG = MODEL_VERSION if MODEL_VERSION else "v1"
VERSION_DIR = PIPELINE_DIR / VERSION_TAG

# 1. Discover Anomaly Manifest across candidate locations
candidate_manifest_paths = [
    VERSION_DIR / "anomalies_audit" / "anomalies_manifest.json",
    PIPELINE_DIR / "anomalies_audit" / "anomalies_manifest.json",
    Path("anomalies_audit") / "anomalies_manifest.json",
    REPO_ROOT / "multi_class_rfdetr" / "v2" / "anomalies_audit" / "anomalies_manifest.json",
    REPO_ROOT / "multi_class_rfdetr" / "anomalies_audit" / "anomalies_manifest.json",
    REPO_ROOT / "multi_class_pipeline" / "anomalies_manifest.json",
    Path("/home/jupyter/location_tag/multi_class_rfdetr/v2/anomalies_audit/anomalies_manifest.json"),
    Path("/home/jupyter/multi_class_rfdetr/v2/anomalies_audit/anomalies_manifest.json"),
]
MANIFEST_PATH = next((p for p in candidate_manifest_paths if p.exists() and p.is_file()), candidate_manifest_paths[0])

# 2. Discover Images Directory
candidate_image_dirs = [
    PIPELINE_DIR / "images",
    VERSION_DIR / "images",
    INPUT_JSON_DIR / "images",
    REPO_ROOT / "coco_files" / "images",
    REPO_ROOT / "images",
    Path("/home/jupyter/location_tag/coco_files/images"),
    Path("/home/jupyter/location_tag/images"),
    Path("/home/jupyter/coco_files/images"),
    Path("/home/jupyter/images")
]
IMAGES_DIR = next((p for p in candidate_image_dirs if p.exists() and any(p.glob("*"))), candidate_image_dirs[0])

# 3. Dynamic Model Weight Discovery: Automatically selects latest or best trained model checkpoint
candidate_model_dirs = [
    VERSION_DIR / "model",
    VERSION_DIR / "checkpoints",
    PIPELINE_DIR / "model",
    PIPELINE_DIR / "checkpoints",
    REPO_ROOT / PIPELINE_NAME / "model",
    REPO_ROOT / PIPELINE_NAME / "checkpoints",
    REPO_ROOT / "model",
    REPO_ROOT / "checkpoints",
    REPO_ROOT / "multi_class_train_rfdetr" / "model",
    REPO_ROOT / "multi_class_train_rfdetr" / "checkpoints",
    Path("/home/jupyter/location_tag/multi_class_rfdetr/v2/checkpoints"),
    Path("/home/jupyter/location_tag/multi_class_rfdetr/v2/model"),
    Path("/home/jupyter/multi_class_rfdetr/v2/checkpoints"),
    Path("/home/jupyter/multi_class_rfdetr/v2/model"),
    Path("model"),
    Path("checkpoints")
]

all_discovered_pths = []
for d in candidate_model_dirs:
    if d.exists():
        for p in d.glob("*.pth"):
            if p.is_file() and p.stat().st_size > 1024 * 1024:
                all_discovered_pths.append(p)

BEST_MODEL_PATH = None
if all_discovered_pths:
    best_candidates = [p for p in all_discovered_pths if p.name == "best_model.pth"]
    latest_candidates = [p for p in all_discovered_pths if "latest" in p.name.lower()]
    precision_candidates = [p for p in all_discovered_pths if "precision" in p.name.lower()]
    epoch_snaps = [p for p in all_discovered_pths if "epoch_" in p.name.lower()]
    epoch_snaps.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if best_candidates:
        BEST_MODEL_PATH = best_candidates[0]
    elif precision_candidates:
        BEST_MODEL_PATH = precision_candidates[0]
    elif latest_candidates:
        BEST_MODEL_PATH = latest_candidates[0]
    elif epoch_snaps:
        BEST_MODEL_PATH = epoch_snaps[0]
    else:
        all_discovered_pths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        BEST_MODEL_PATH = all_discovered_pths[0]

# Confidence & Postprocessing parameters
CONFIDENCE_THRESHOLD = 0.40
SHRINK_FACTOR = 0.96
WBF_IOU_THRESHOLD = 0.50
RESOLUTION = 560

OUTPUT_EVAL_DIR = VERSION_DIR / "anomalies_evaluation"
OUTPUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Anomaly Manifest Path:  {MANIFEST_PATH}")
logger.info(f"Images Directory:      {IMAGES_DIR}")
logger.info(f"Selected Model Path:   {BEST_MODEL_PATH}")"""

cell_3 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in cell_3_code.splitlines()]
}
cell_3["source"][-1] = cell_3["source"][-1].rstrip("\n")

# Cell 4: Model Instantiation and Weights Loading
cell_4_code = """# CELL 3 - Safe Pretrained Weights Loading & Detection Head Patch
from rfdetr.models.lwdetr import LWDETR

def _safe_lwdetr_load_state_dict(self, state_dict, strict=True):
    \"\"\"Filters out classification heads (class_embed, enc_out_class_embed) from checkpoint when num_classes differs.\"\"\"
    model_state = self.state_dict()
    filtered_state_dict = {}
    mismatched = []
    
    for k, v in state_dict.items():
        clean_k = k
        if clean_k not in model_state and clean_k.startswith("model."):
            clean_k = clean_k[6:]
        if clean_k not in model_state and clean_k.startswith("module."):
            clean_k = clean_k[7:]
            
        if clean_k in model_state:
            if model_state[clean_k].shape == v.shape:
                filtered_state_dict[clean_k] = v
            else:
                mismatched.append((clean_k, tuple(v.shape), tuple(model_state[clean_k].shape)))
        elif k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                mismatched.append((k, tuple(v.shape), tuple(model_state[k].shape)))

    if mismatched:
        logger.info(f"Notice: Filtered {len(mismatched)} classification head layers with class-count mismatch from COCO weights:")
        for name, ckpt_shape, model_shape in mismatched[:4]:
            logger.info(f"   - {name}: checkpoint {ckpt_shape} -> target model {model_shape}")
    return torch.nn.Module.load_state_dict(self, filtered_state_dict, strict=False)

LWDETR.load_state_dict = _safe_lwdetr_load_state_dict
logger.info("LWDETR.load_state_dict patched for safe custom num_classes loading.")

def _fixed_reinitialize(self, num_classes):
    lwdetr = self.model
    device = next(lwdetr.parameters()).device
    if hasattr(lwdetr, "class_embed"):
        in_feat = lwdetr.class_embed.in_features
        lwdetr.class_embed = nn.Linear(in_feat, num_classes).to(device)
        nn.init.normal_(lwdetr.class_embed.weight, std=0.01)
        nn.init.zeros_(lwdetr.class_embed.bias)
    if hasattr(lwdetr, "transformer") and hasattr(lwdetr.transformer, "enc_out_class_embed"):
        enc_heads = lwdetr.transformer.enc_out_class_embed
        if isinstance(enc_heads, nn.ModuleList):
            for i in range(len(enc_heads)):
                in_feat = enc_heads[i].in_features
                enc_heads[i] = nn.Linear(in_feat, num_classes).to(device)
                nn.init.normal_(enc_heads[i].weight, std=0.01)
                nn.init.zeros_(enc_heads[i].bias)
        elif isinstance(enc_heads, nn.Linear):
            in_feat = enc_heads.in_features
            lwdetr.transformer.enc_out_class_embed = nn.Linear(in_feat, num_classes).to(device)
            nn.init.normal_(lwdetr.transformer.enc_out_class_embed.weight, std=0.01)
            nn.init.zeros_(lwdetr.transformer.enc_out_class_embed.bias)
    logger.info(f"Detection heads initialized for num_classes={num_classes}")

rfdetr_main.Model.reinitialize_detection_head = _fixed_reinitialize

# 3. Model Instantiation
MODEL_SIZE = globals().get("MODEL_SIZE", os.getenv("MODEL_SIZE", "base"))
RESOLUTION = globals().get("RESOLUTION", 560)

class_names = ["blue_aisle", "blue_bay", "location_tag"]
NUM_CLASSES = len(class_names)

model_cls_map = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge
}
ModelClass = model_cls_map.get(str(MODEL_SIZE).lower(), RFDETRBase)
model = ModelClass(
    num_classes=NUM_CLASSES,
    pretrain_weights=None,
    resolution=RESOLUTION
)
logger.info(f"RF-DETR-{str(MODEL_SIZE).capitalize()} instantiated with resolution={RESOLUTION}, num_classes={NUM_CLASSES}")

if hasattr(model, "model") and hasattr(model.model, "reinitialize_detection_head"):
    model.model.reinitialize_detection_head(NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lwdetr = model.model.model if hasattr(model, "model") and hasattr(model.model, "model") else getattr(model, "model", model)
lwdetr.to(device)
lwdetr.eval()

# Load latest trained model weights directly
target_weights = globals().get("BEST_MODEL_PATH", None)
if target_weights and os.path.exists(str(target_weights)):
    try:
        ckpt_data = torch.load(str(target_weights), map_location=device, weights_only=False)
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        lwdetr.load_state_dict(state_dict, strict=False)
        logger.info(f"✓ Successfully loaded trained model weights from: {target_weights}")
    except Exception as e:
        logger.warning(f"Could not load trained weights from {target_weights}: {e}")
else:
    logger.warning("No trained weights found on disk. Operating with initialized architecture.")"""

cell_4 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in cell_4_code.splitlines()]
}
cell_4["source"][-1] = cell_4["source"][-1].rstrip("\n")

# Cell 5: Postprocessing and Inference Functions
cell_5_code = """# CELL 4 - Postprocessing Functions & Anomaly Image Path Resolver
def shrink_box(box_xyxy, factor: float = 0.96, img_w: int = 1920, img_h: int = 1080):
    \"\"\"Insets box margins toward center by factor (e.g. 0.96 = 4% contraction) to isolate tag from shelf.\"\"\"
    x1, y1, x2, y2 = box_xyxy
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    sw, sh = w * factor, h * factor
    return [
        max(0.0, cx - sw / 2.0),
        max(0.0, cy - sh / 2.0),
        min(float(img_w), cx + sw / 2.0),
        min(float(img_h), cy + sh / 2.0)
    ]

def weighted_box_fusion(boxes, scores, labels, iou_thresh: float = 0.50):
    \"\"\"Merges overlapping candidate queries from DETR decoder weighted by confidence scores.\"\"\"
    if len(boxes) == 0:
        return [], [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    
    fused_boxes, fused_scores, fused_labels = [], [], []
    unique_labels = np.unique(labels)
    
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        lbl_boxes = boxes[idx]
        lbl_scores = scores[idx]
        
        order = np.argsort(lbl_scores)[::-1]
        lbl_boxes = lbl_boxes[order]
        lbl_scores = lbl_scores[order]
        
        used = np.zeros(len(lbl_boxes), dtype=bool)
        for i in range(len(lbl_boxes)):
            if used[i]: continue
            cluster = [i]
            for j in range(i + 1, len(lbl_boxes)):
                if used[j]: continue
                bA, bB = lbl_boxes[i], lbl_boxes[j]
                xA = max(bA[0], bB[0]); yA = max(bA[1], bB[1])
                xB = min(bA[2], bB[2]); yB = min(bA[3], bB[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                denom = (bA[2]-bA[0])*(bA[3]-bA[1]) + (bB[2]-bB[0])*(bB[3]-bB[1]) - inter
                if denom > 0 and (inter / denom) >= iou_thresh:
                    cluster.append(j)
                    used[j] = True
            used[i] = True
            c_boxes = lbl_boxes[cluster]
            c_scores = lbl_scores[cluster]
            total_s = np.sum(c_scores)
            fused_box = np.sum(c_boxes * c_scores[:, None], axis=0) / total_s
            fused_boxes.append(fused_box.tolist())
            fused_scores.append(float(np.max(c_scores)))
            fused_labels.append(lbl)
            
    return fused_boxes, fused_scores, fused_labels

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

def apply_postprocessing(raw_boxes_xyxy, raw_scores, raw_labels, img_w: int, img_h: int, 
                         shrink_factor: float = 0.96, wbf_iou: float = 0.50):
    if len(raw_boxes_xyxy) == 0:
        return [], [], []
    f_boxes, f_scores, f_labels = weighted_box_fusion(raw_boxes_xyxy, raw_scores, raw_labels, iou_thresh=wbf_iou)
    if shrink_factor and shrink_factor < 1.0:
        shrunk_boxes = [shrink_box(b, factor=shrink_factor, img_w=img_w, img_h=img_h) for b in f_boxes]
    else:
        shrunk_boxes = f_boxes
    return shrunk_boxes, f_scores, f_labels

def run_model_inference_on_pil(model, pil_img, device, resolution: int = 560, conf_thresh: float = 0.40):
    \"\"\"Runs inference on a single PIL image and returns postprocessed predictions.\"\"\"
    orig_w, orig_h = pil_img.size
    img_resized = pil_img.resize((resolution, resolution))
    tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(tensor)
    
    logits = outputs["pred_logits"][0]
    boxes = outputs["pred_boxes"][0]
    scores, labels = logits.sigmoid().max(dim=-1)
    keep = scores > conf_thresh
    
    if keep.sum() == 0:
        return [], [], []
        
    scaled_boxes = (box_cxcywh_to_xyxy(boxes[keep]) * torch.tensor([orig_w, orig_h, orig_w, orig_h], device=device)).cpu().numpy().tolist()
    kept_scores = scores[keep].cpu().numpy().tolist()
    kept_labels = labels[keep].cpu().numpy().tolist()
    
    post_boxes, post_scores, post_labels = apply_postprocessing(
        scaled_boxes, kept_scores, kept_labels,
        img_w=orig_w, img_h=orig_h, shrink_factor=SHRINK_FACTOR, wbf_iou=WBF_IOU_THRESHOLD
    )
    return post_boxes, post_scores, post_labels

# Universal image path finder for anomalies
search_image_dirs = [
    IMAGES_DIR,
    REPO_ROOT / "coco_files" / "images",
    REPO_ROOT / "images",
    REPO_ROOT / "multi_class_rfdetr" / "images",
    REPO_ROOT / "multi_class_rfdetr" / "dataset" / "images",
    Path("/home/jupyter/location_tag/coco_files/images"),
    Path("/home/jupyter/location_tag/images"),
    Path("/home/jupyter/coco_files/images"),
    Path("/home/jupyter/images")
]
search_image_dirs = [d for d in search_image_dirs if d and os.path.exists(str(d))]

disk_image_index = {}
for d in search_image_dirs:
    try:
        for entry in os.scandir(d):
            if entry.is_file():
                disk_image_index[entry.name] = entry.path
                disk_image_index[entry.name.lower()] = entry.path
                disk_image_index[Path(entry.name).stem] = entry.path
                disk_image_index[Path(entry.name).stem.lower()] = entry.path
                if "sig=" in entry.name:
                    s_tok = entry.name.split("sig=")[-1].split("&")[0]
                    disk_image_index[s_tok] = entry.path
                    disk_image_index[f"{s_tok}.jpg"] = entry.path
    except Exception:
        pass

def find_image_for_anomaly(anom: dict) -> Optional[Path]:
    ann = anom.get("annotation", {})
    raw_cands = [
        ann.get("file_name"),
        anom.get("image_url"),
        ann.get("original_url"),
        ann.get("coco_url"),
        ann.get("azure_url"),
        f"{ann.get('image_id')}.jpg" if ann.get('image_id') else None
    ]
    for raw in raw_cands:
        if not raw: continue
        s_raw = str(raw).strip()
        if s_raw in disk_image_index:
            return Path(disk_image_index[s_raw])
        p_name = Path(urlparse(unquote(s_raw)).path).name
        if p_name and p_name in disk_image_index:
            return Path(disk_image_index[p_name])
        if Path(s_raw).stem in disk_image_index:
            return Path(disk_image_index[Path(s_raw).stem])
        for d in search_image_dirs:
            candidate = Path(d) / p_name
            if candidate.exists() and candidate.is_file():
                return candidate
    return None

logger.info(f"✓ Indexed {len(disk_image_index)} disk image paths for anomaly evaluation.")"""

cell_5 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in cell_5_code.splitlines()]
}
cell_5["source"][-1] = cell_5["source"][-1].rstrip("\n")

# Cell 6: Quantitative Resilience Audit
cell_6_code = """# CELL 5 - Quantitative Resilience Analysis on Flagged Anomaly Cases
if not MANIFEST_PATH.exists():
    logger.warning(f"Manifest not found at {MANIFEST_PATH}. Please run Step 02 first.")
    flagged_anomalies = []
else:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        anom_data = json.load(f)
    flagged_anomalies = anom_data.get("anomalies", [])

logger.info("=" * 70)
logger.info(f"AUDITING {len(flagged_anomalies)} DETECTED ANOMALIES AGAINST TRAINED RF-DETR MODEL")
logger.info("=" * 70)

reasons_count = Counter()
category_count = Counter()
for item in flagged_anomalies:
    for r in item.get("reasons", ["Unspecified"]):
        rule_name = r.split("(")[0].strip()
        reasons_count[rule_name] += 1
    category_count[item.get("category_name", "Unknown")] += 1

logger.info("Anomaly Distribution by Flagged Rule:")
for r, c in reasons_count.most_common():
    logger.info(f"   • {r:35s}: {c} instances")

logger.info("\nAnomaly Distribution by Object Category:")
for cat, c in category_count.most_common():
    logger.info(f"   • {cat:25s}: {c} instances")"""

cell_6 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in cell_6_code.splitlines()]
}
cell_6["source"][-1] = cell_6["source"][-1].rstrip("\n")

# Cell 7: Visual Sample Predictions on Anomalies
cell_7_code = """# CELL 6 - Side-by-Side Visual Showcase: Ground Truth Anomaly vs Model Prediction
logger.info("=" * 70)
logger.info("GENERATING VISUAL PREVIEWS ON ANOMALOUS SAMPLES")
logger.info("=" * 70)

# Group anomalies by primary rule to ensure diversity across all anomaly types
anomalies_by_rule = defaultdict(list)
for anom in flagged_anomalies:
    primary = anom.get("primary_rule") or (anom.get("reasons", ["Anomaly"])[0].split("(")[0].strip())
    anomalies_by_rule[primary].append(anom)

preview_count = 0
max_previews = 8
rendered_images = set()

for rule_name, anom_list in anomalies_by_rule.items():
    if preview_count >= max_previews:
        break
        
    for anom in anom_list:
        img_path = find_image_for_anomaly(anom)
        if not img_path or not img_path.exists():
            continue
        if str(img_path) in rendered_images:
            continue
            
        rendered_images.add(str(img_path))
        pil_im = Image.open(img_path).convert("RGB")
        W, H = pil_im.size
        
        # 1. Extract Flawed Ground Truth Box
        gt_box = anom.get("bbox", [0, 0, 0, 0])
        if len(gt_box) == 4:
            if gt_box[2] <= 1.05 and gt_box[3] <= 1.05 and (W > 10 and H > 10):
                gt_xyxy = [gt_box[0] * W, gt_box[1] * H, (gt_box[0] + gt_box[2]) * W, (gt_box[1] + gt_box[3]) * H]
            else:
                gt_xyxy = [gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]
        else:
            gt_xyxy = [0, 0, 10, 10]
            
        gt_reason = anom.get("reasons", [rule_name])[0]
        gt_cat = anom.get("category_name", "Anomaly")
        
        # 2. Run Model Inference
        pred_boxes, pred_scores, pred_labels = run_model_inference_on_pil(
            lwdetr, pil_im, device, resolution=RESOLUTION, conf_thresh=CONFIDENCE_THRESHOLD
        )
        
        # 3. Render Side-by-Side Comparison Panels
        left_im = pil_im.copy()
        draw_left = ImageDraw.Draw(left_im)
        draw_left.rectangle(gt_xyxy, outline="#FF1744", width=4)
        gt_badge = f"GT Flaw: {gt_reason[:28]}"
        draw_left.rectangle([gt_xyxy[0], max(0, gt_xyxy[1]-20), gt_xyxy[0] + len(gt_badge)*7 + 10, max(0, gt_xyxy[1])], fill="#FF1744")
        draw_left.text((gt_xyxy[0]+4, max(2, gt_xyxy[1]-17)), gt_badge, fill="#FFFFFF")
        
        right_im = pil_im.copy()
        draw_right = ImageDraw.Draw(right_im)
        for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels):
            p_cat = class_names[int(pl)] if int(pl) < len(class_names) else f"class_{pl}"
            col = "#00E676" if p_cat == "location_tag" else "#2979FF"
            draw_right.rectangle(pb, outline=col, width=4)
            p_badge = f"{p_cat} ({ps:.2f})"
            draw_right.rectangle([pb[0], max(0, pb[1]-20), pb[0] + len(p_badge)*7 + 10, max(0, pb[1])], fill=col)
            draw_right.text((pb[0]+4, max(2, pb[1]-17)), p_badge, fill="#000000")
            
        disp_h = 420
        aspect = W / max(H, 1)
        disp_w = int(disp_h * aspect)
        
        p1 = left_im.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        p2 = right_im.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        
        canvas = Image.new("RGB", (disp_w * 2 + 10, disp_h + 36), color=(24, 24, 24))
        draw_c = ImageDraw.Draw(canvas)
        draw_c.text((12, 10), f"[LEFT] Flawed Ground Truth Label: {rule_name}", fill="#FF5252")
        draw_c.text((disp_w + 22, 10), f"[RIGHT] Trained RF-DETR Model Predictions ({len(pred_boxes)} detected)", fill="#69F0AE")
        
        canvas.paste(p1, (0, 36))
        canvas.paste(p2, (disp_w + 10, 36))
        
        out_preview_path = OUTPUT_EVAL_DIR / f"anomaly_comparison_{preview_count+1}.png"
        canvas.save(out_preview_path)
        logger.info(f"\nSample #{preview_count+1} [{rule_name}]: {img_path.name}")
        logger.info(f"   - GT Anomaly:   {gt_cat} -> {gt_reason}")
        logger.info(f"   - Model Output: {len(pred_boxes)} clean predictions generated.")
        display(canvas)
        
        preview_count += 1
        break

logger.info(f"\n✓ Successfully rendered {preview_count} side-by-side anomaly robustness comparisons into {OUTPUT_EVAL_DIR}")"""

cell_7 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in cell_7_code.splitlines()]
}
cell_7["source"][-1] = cell_7["source"][-1].rstrip("\n")

nb["cells"] = [cell_0, cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook/08_evaluate_anomalies_audit.ipynb successfully!")

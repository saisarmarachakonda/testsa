# ==============================================================================
# RF-DETR ONNX Inference + PaddleOCR Microservice (FastAPI)
# ==============================================================================
import os
import io
import time
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort

MODEL_PATH = os.getenv("MODEL_PATH", "rfdetr_model.onnx")
RESOLUTION = int(os.getenv("RESOLUTION", "1008"))
CLASSES_ENV = os.getenv("CLASSES", '["blue_aisle", "blue_bay", "location_tag"]')
try:
    import json
    CLASSES = json.loads(CLASSES_ENV)
except Exception:
    CLASSES = ["blue_aisle", "blue_bay", "location_tag"]

app = FastAPI(
    title="RF-DETR + PaddleOCR Inference API",
    description="Private ONNX Runtime Detection and PaddleOCR Text Recognition on AKS",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engines
session = None
input_name = None
output_names = []
ocr_engine = None

@app.on_event("startup")
def startup_engines():
    global session, input_name, output_names, ocr_engine

    # 1. Initialize RF-DETR ONNX Session
    if not os.path.exists(MODEL_PATH):
        print(f"[STARTUP WARNING] ONNX model file not found at: {MODEL_PATH}")
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = int(os.getenv("ORT_NUM_THREADS", str(os.cpu_count() or 4)))
        session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=providers)
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        print(f"[STARTUP] Loaded RF-DETR ONNX model '{MODEL_PATH}' using {session.get_providers()[0]}")

    # 2. Initialize PaddleOCR Engine
    try:
        from paddleocr import PaddleOCR
        use_gpu = (ort.get_device() == "GPU")
        ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=use_gpu, show_log=False)
        print(f"[STARTUP] PaddleOCR initialized successfully (use_gpu={use_gpu}).")
    except Exception as e:
        print(f"[STARTUP NOTICE] PaddleOCR could not be initialized: {e}")
        ocr_engine = None

# Response Schemas
class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    box_xyxy: List[float] = Field(..., description="[x1, y1, x2, y2] in original image pixel coordinates")
    fused_count: Optional[int] = Field(1, description="Number of candidate queries fused via weighted box averaging")
    ocr_text: Optional[str] = Field(None, description="Recognized text inside bounding box using PaddleOCR")
    ocr_confidence: Optional[float] = Field(None, description="PaddleOCR recognition confidence score")

class PredictionResponse(BaseModel):
    success: bool
    image_width: int
    image_height: int
    inference_time_ms: float
    ocr_time_ms: Optional[float] = None
    detections_count: int
    detections: List[Detection]

class OCRResult(BaseModel):
    text: str
    confidence: float
    box: Optional[List[List[float]]] = None

class DirectOCRResponse(BaseModel):
    success: bool
    ocr_time_ms: float
    results_count: int
    results: List[OCRResult]

@app.get("/healthz", status_code=status.HTTP_200_OK, tags=["Monitoring"])
def health_check():
    """Kubernetes liveness and readiness probe endpoint."""
    if session is None and ocr_engine is None:
        raise HTTPException(status_code=503, detail="Engines not ready")
    return {
        "status": "healthy",
        "rfdetr_loaded": session is not None,
        "paddleocr_loaded": ocr_engine is not None,
        "provider": session.get_providers()[0] if session else "none"
    }

@app.get("/metadata", tags=["Metadata"])
def get_metadata():
    """Returns model architecture, input resolution, classes, and OCR capabilities."""
    return {
        "model_path": MODEL_PATH,
        "resolution": RESOLUTION,
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "rfdetr_loaded": session is not None,
        "paddleocr_loaded": ocr_engine is not None,
        "execution_provider": session.get_providers()[0] if session else "uninitialized"
    }

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    resized = pil_img.resize((RESOLUTION, RESOLUTION), Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_np - mean) / std
    
    return np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

def run_paddle_ocr_on_crop(crop_img: Image.Image) -> Tuple[Optional[str], Optional[float]]:
    if ocr_engine is None:
        return None, None
    try:
        crop_np = np.array(crop_img.convert("RGB"))
        if crop_np.shape[0] < 4 or crop_np.shape[1] < 4:
            return None, None
            
        ocr_res = ocr_engine.ocr(crop_np, cls=True)
        if not ocr_res or not ocr_res[0]:
            return None, None
            
        texts = []
        confs = []
        for line in ocr_res[0]:
            txt = line[1][0].strip()
            score = float(line[1][1])
            if txt:
                texts.append(txt)
                confs.append(score)
                
        if texts:
            return " ".join(texts), round(float(np.mean(confs)), 4)
        return None, None
    except Exception as e:
        print(f"[OCR WARNING] Failed to run PaddleOCR on patch: {e}")
        return None, None

def recalibrate_and_average_boxes(
    boxes: List[List[float]],
    scores: List[float],
    class_ids: List[int],
    class_names: List[str] = None,
    enable_averaging: bool = True,
    enable_shrinkage: bool = True,
    shrink_factor: float = 0.96,
    fusion_iou: float = 0.50,
    max_w: float = 1e6,
    max_h: float = 1e6
) -> List[dict]:
    """
    Recalibrate detections via:
    1. Boundary-Box Shrinkage: Contracts loose query boundaries towards center by shrink_factor
       (e.g., 0.96 shrinks width and height by 4%), eliminating background margin jitter
       and tightening crops for downstream PaddleOCR.
    2. Bounding-Box Averaging: Clusters overlapping queries of the same class (IoU >= fusion_iou)
       and performs confidence-weighted coordinate averaging, suppressing duplicate false alarms.
    """
    if not boxes:
        return []
        
    shrunk = []
    for i, b in enumerate(boxes):
        if enable_shrinkage and shrink_factor < 1.0:
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            w = (b[2] - b[0]) * shrink_factor
            h = (b[3] - b[1]) * shrink_factor
            x1 = max(0.0, cx - w / 2.0)
            y1 = max(0.0, cy - h / 2.0)
            x2 = min(float(max_w), cx + w / 2.0)
            y2 = min(float(max_h), cy + h / 2.0)
            shrunk_b = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        else:
            shrunk_b = [round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)]
            
        cname = class_names[i] if class_names else str(class_ids[i])
        shrunk.append({
            "box": shrunk_b,
            "score": float(scores[i]),
            "class_id": int(class_ids[i]),
            "class_name": cname
        })
        
    if not enable_averaging or len(shrunk) <= 1:
        return shrunk
        
    # Bounding-box weighted averaging (WBF)
    fused = []
    distinct_classes = set(p["class_id"] for p in shrunk)
    for cid in distinct_classes:
        c_preds = [p for p in shrunk if p["class_id"] == cid]
        c_preds.sort(key=lambda x: x["score"], reverse=True)
        
        clusters = []
        for p in c_preds:
            matched = False
            for cl in clusters:
                b1, b2 = p["box"], cl[0]["box"]
                xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
                xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / max(1e-7, a1 + a2 - inter)
                if iou >= fusion_iou:
                    cl.append(p)
                    matched = True
                    break
            if not matched:
                clusters.append([p])
                
        for cl in clusters:
            if len(cl) == 1:
                fused.append(cl[0])
            else:
                weights = [max(0.01, c["score"]) for c in cl]
                total_w = sum(weights)
                avg_x1 = sum(c["box"][0] * w for c, w in zip(cl, weights)) / total_w
                avg_y1 = sum(c["box"][1] * w for c, w in zip(cl, weights)) / total_w
                avg_x2 = sum(c["box"][2] * w for c, w in zip(cl, weights)) / total_w
                avg_y2 = sum(c["box"][3] * w for c, w in zip(cl, weights)) / total_w
                consensus_score = min(1.0, max(c["score"] for c in cl) + 0.01 * (len(cl) - 1))
                fused.append({
                    "box": [round(avg_x1, 1), round(avg_y1, 1), round(avg_x2, 1), round(avg_y2, 1)],
                    "score": round(float(consensus_score), 4),
                    "class_id": cid,
                    "class_name": cl[0]["class_name"],
                    "fused_count": len(cl)
                })
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="JPEG/PNG image file"),
    conf_threshold: float = Query(0.25, ge=0.0, le=1.0, description="Minimum detection confidence score"),
    run_ocr: bool = Query(True, description="Run PaddleOCR on detected tag bounding boxes"),
    enable_box_averaging: bool = Query(True, description="Fuse overlapping candidate queries with weighted box averaging"),
    enable_shrinkage: bool = Query(True, description="Recalibrate loose borders with boundary-box shrinkage"),
    shrink_factor: float = Query(0.96, ge=0.80, le=1.0, description="Boundary shrinkage factor (0.96 = 4% border contraction)"),
    fusion_iou: float = Query(0.50, ge=0.10, le=0.90, description="IoU threshold for box averaging clustering")
):
    """
    Two-Stage Detection & Recognition Pipeline:
    1. RF-DETR detects objects / location tags.
    2. PaddleOCR extracts text inside each detected bounding box.
    """
    if session is None:
        raise HTTPException(status_code=503, detail="RF-DETR model is not initialized")
        
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_w, orig_h = pil_img.size
        tensor = preprocess_image(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")
        
    t_start = time.perf_counter()
    raw_outs = session.run(output_names, {input_name: tensor})
    inf_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
    
    out_scores = raw_outs[0][0]  # (queries, num_classes)
    out_boxes  = raw_outs[1][0]  # (queries, 4) in cx, cy, w, h
    
    if out_scores.min() < 0.0 or out_scores.max() > 1.0:
        out_scores = 1.0 / (1.0 + np.exp(-out_scores))
        
    max_scores = np.max(out_scores, axis=-1)
    class_indices = np.argmax(out_scores, axis=-1)
    
    keep_mask = max_scores >= conf_threshold
    filtered_scores = max_scores[keep_mask]
    filtered_classes = class_indices[keep_mask]
    filtered_boxes = out_boxes[keep_mask]
    
    # Extract raw pixel coordinates
    raw_boxes_xyxy: List[List[float]] = []
    raw_scores_list: List[float] = []
    raw_class_ids: List[int] = []
    raw_class_names: List[str] = []
    
    for sc, cid, (cx, cy, w, h) in zip(filtered_scores, filtered_classes, filtered_boxes):
        x1 = float(max(0.0, (cx - w / 2.0) * orig_w))
        y1 = float(max(0.0, (cy - h / 2.0) * orig_h))
        x2 = float(min(float(orig_w), (cx + w / 2.0) * orig_w))
        y2 = float(min(float(orig_h), (cy + h / 2.0) * orig_h))
        c_name = CLASSES[cid] if cid < len(CLASSES) else f"class_{cid}"
        
        raw_boxes_xyxy.append([x1, y1, x2, y2])
        raw_scores_list.append(float(sc))
        raw_class_ids.append(int(cid))
        raw_class_names.append(c_name)
        
    # Recalibrate Detections: Boundary-Box Shrinkage & Bounding-Box Averaging
    recalibrated = recalibrate_and_average_boxes(
        boxes=raw_boxes_xyxy,
        scores=raw_scores_list,
        class_ids=raw_class_ids,
        class_names=raw_class_names,
        enable_averaging=enable_box_averaging,
        enable_shrinkage=enable_shrinkage,
        shrink_factor=shrink_factor,
        fusion_iou=fusion_iou,
        max_w=orig_w,
        max_h=orig_h
    )

    detections: List[Detection] = []
    t_ocr_start = time.perf_counter()
    
    for item in recalibrated:
        sc = item["score"]
        cid = item["class_id"]
        c_name = item["class_name"]
        x1, y1, x2, y2 = item["box"]
        
        # Run PaddleOCR on tightly recalibrated crop
        ocr_text, ocr_conf = None, None
        if run_ocr and ocr_engine is not None and (x2 - x1 >= 5) and (y2 - y1 >= 5):
            crop = pil_img.crop((int(x1), int(y1), int(x2), int(y2)))
            ocr_text, ocr_conf = run_paddle_ocr_on_crop(crop)
            
        detections.append(Detection(
            class_name=c_name,
            class_id=int(cid),
            confidence=round(float(sc), 4),
            box_xyxy=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            fused_count=item.get("fused_count", 1),
            ocr_text=ocr_text,
            ocr_confidence=ocr_conf
        ))
        
    ocr_time_ms = round((time.perf_counter() - t_ocr_start) * 1000, 2) if run_ocr else None
    
    return PredictionResponse(
        success=True,
        image_width=orig_w,
        image_height=orig_h,
        inference_time_ms=inf_time_ms,
        ocr_time_ms=ocr_time_ms,
        detections_count=len(detections),
        detections=detections
    )

@app.post("/ocr", response_model=DirectOCRResponse, tags=["Inference"])
async def direct_ocr(
    file: UploadFile = File(..., description="JPEG/PNG image to run PaddleOCR on directly")
):
    """Runs PaddleOCR directly on the full image."""
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="PaddleOCR engine is not initialized")
        
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")
        
    t_start = time.perf_counter()
    ocr_res = ocr_engine.ocr(img_np, cls=True)
    ocr_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
    
    results = []
    if ocr_res and ocr_res[0]:
        for line in ocr_res[0]:
            box_coords = [[float(p[0]), float(p[1])] for p in line[0]]
            txt = line[1][0]
            score = round(float(line[1][1]), 4)
            results.append(OCRResult(text=txt, confidence=score, box=box_coords))
            
    return DirectOCRResponse(
        success=True,
        ocr_time_ms=ocr_time_ms,
        results_count=len(results),
        results=results
    )

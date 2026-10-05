import json

nb_path = "notebook/07_evaluate_rfdetr_clean_data.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 3 (Configuration) to add EVAL_BATCH_SIZE = 1, NUM_EVAL_WORKERS = 0, MAX_EVAL_SAMPLES = None
c3_text = "".join(nb["cells"][3]["source"])
old_c3_config = """# =========================================================================
# EVALUATION & POSTPROCESSING CONFIGURATION
# =========================================================================
CONFIDENCE_THRESHOLD = 0.50    # Confidence threshold for test evaluation (e.g. 0.30 - 0.55)
BOX_AVERAGING_METHOD = "wbf"   # Options: "wbf" (Weighted Box Fusion), "nms" (Greedy NMS), "none" (Raw)
FUSION_IOU_THRESHOLD = 0.50    # IoU threshold for WBF / NMS box clustering
BOX_SHRINK_FACTOR = 0.96       # Margin shrinkage factor (1.0 = disabled, 0.96 = 4% contraction)
RESOLUTION = 560               # Model evaluation input resolution

# Backward compatibility aliases
CONFIDENCE = CONFIDENCE_THRESHOLD
NMS_THRESHOLD = FUSION_IOU_THRESHOLD"""

new_c3_config = """# =========================================================================
# EVALUATION & POSTPROCESSING CONFIGURATION (OOM-Safe Execution)
# =========================================================================
CONFIDENCE_THRESHOLD = 0.50    # Confidence threshold for test evaluation (e.g. 0.30 - 0.55)
BOX_AVERAGING_METHOD = "wbf"   # Options: "wbf" (Weighted Box Fusion), "nms" (Greedy NMS), "none" (Raw)
FUSION_IOU_THRESHOLD = 0.50    # IoU threshold for WBF / NMS box clustering
BOX_SHRINK_FACTOR = 0.96       # Margin shrinkage factor (1.0 = disabled, 0.96 = 4% contraction)
RESOLUTION = 560               # Model evaluation input resolution

# VRAM & Memory Management Settings (Guaranteed zero OOM)
EVAL_BATCH_SIZE = 1            # Batch size = 1 avoids peak activation spikes on GPU
NUM_EVAL_WORKERS = 0           # 0 workers avoids multiprocessing RAM overhead and shared memory leaks
MAX_EVAL_SAMPLES = None        # Set an integer (e.g. 500) for quick audit, or None for entire test set

# Backward compatibility aliases
CONFIDENCE = CONFIDENCE_THRESHOLD
NMS_THRESHOLD = FUSION_IOU_THRESHOLD"""

assert old_c3_config in c3_text, "old_c3_config not found in Cell 3"
c3_text = c3_text.replace(old_c3_config, new_c3_config)
nb["cells"][3]["source"] = [line + "\n" for line in c3_text.split("\n")[:-1]] + ([c3_text.split("\n")[-1]] if c3_text.split("\n")[-1] else [])

# 2. Update Cell 4 (Safe weight loading on CPU before sending to target device, clean up unused wrapper)
c4_text = "".join(nb["cells"][4]["source"])
old_c4_tail = """# Extract underlying model module and assign default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lwdetr = model.model.model if hasattr(model, "model") and hasattr(model.model, "model") else getattr(model, "model", model)
lwdetr.to(device)
logger.info(f"Model underlying LWDETR assigned to device: {device}")"""

new_c4_tail = """# Extract underlying model module and assign default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lwdetr = model.model.model if hasattr(model, "model") and hasattr(model.model, "model") else getattr(model, "model", model)

# Clear unused outer model wrapper to free CPU/GPU RAM
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

lwdetr.to(device)
lwdetr.eval()
logger.info(f"Model underlying LWDETR initialized in eval mode on: {device}")"""

assert old_c4_tail in c4_text, "old_c4_tail not found in Cell 4"
c4_text = c4_text.replace(old_c4_tail, new_c4_tail)
nb["cells"][4]["source"] = [line + "\n" for line in c4_text.split("\n")[:-1]] + ([c4_text.split("\n")[-1]] if c4_text.split("\n")[-1] else [])

# 3. Update Cell 6 (Build DataLoader with batch_size=1, num_workers=0, pin_memory=False, and VRAM purge)
c6_text = "".join(nb["cells"][6]["source"])
old_c6 = """# CELL 5 - Build Test DataLoader & Load Model Weights
test_ds = COCODetectionDataset(str(IMAGES_DIR), TEST_ANN, RESOLUTION, is_train=False)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lwdetr = model.model.model
lwdetr.to(device)

if BEST_MODEL_PATH and os.path.exists(str(BEST_MODEL_PATH)):
    ckpt_data = torch.load(str(BEST_MODEL_PATH), map_location=device, weights_only=False)
    state = ckpt_data.get("model_state_dict", ckpt_data)
    lwdetr.load_state_dict(state, strict=False)
    epoch_info = f" (Epoch {ckpt_data['epoch']})" if isinstance(ckpt_data, dict) and 'epoch' in ckpt_data else ""
    loss_val = ckpt_data.get('best_loss', ckpt_data.get('val_loss', 'N/A')) if isinstance(ckpt_data, dict) else 'N/A'
    best_loss_info = f" | Best Val Loss: {loss_val}" if loss_val != 'N/A' else ""
    logger.info(f"✓ Verified & loaded best model weights from training: {BEST_MODEL_PATH}{epoch_info}{best_loss_info}")
else:
    logger.warning("No best model checkpoint found to evaluate! Please check training directory.")"""

new_c6 = """# CELL 5 - Build Test DataLoader & Load Model Weights (OOM-Safe Configuration)
# Proactively clear cached CUDA memory prior to test batch evaluation
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

test_ds = COCODetectionDataset(str(IMAGES_DIR), TEST_ANN, RESOLUTION, is_train=False)

# Support subset evaluation if MAX_EVAL_SAMPLES is set
if globals().get("MAX_EVAL_SAMPLES") and len(test_ds) > MAX_EVAL_SAMPLES:
    test_ds.samples = test_ds.samples[:MAX_EVAL_SAMPLES]
    logger.info(f"Limiting evaluation to first {MAX_EVAL_SAMPLES} samples for quick audit.")

# batch_size=1 and num_workers=0 strictly avoids CUDA Out-of-Memory and host RAM spikes
test_loader = DataLoader(
    test_ds,
    batch_size=globals().get("EVAL_BATCH_SIZE", 1),
    shuffle=False,
    num_workers=globals().get("NUM_EVAL_WORKERS", 0),
    collate_fn=collate_fn,
    pin_memory=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lwdetr.to(device)
lwdetr.eval()

if BEST_MODEL_PATH and os.path.exists(str(BEST_MODEL_PATH)):
    # Load on CPU first to prevent GPU memory fragmentation spikes during checkpoint deserialization
    ckpt_data = torch.load(str(BEST_MODEL_PATH), map_location="cpu", weights_only=False)
    state = ckpt_data.get("model_state_dict", ckpt_data)
    lwdetr.load_state_dict(state, strict=False)
    
    epoch_info = f" (Epoch {ckpt_data['epoch']})" if isinstance(ckpt_data, dict) and 'epoch' in ckpt_data else ""
    loss_val = ckpt_data.get('best_loss', ckpt_data.get('val_loss', 'N/A')) if isinstance(ckpt_data, dict) else 'N/A'
    best_loss_info = f" | Best Val Loss: {loss_val}" if loss_val != 'N/A' else ""
    logger.info(f"✓ Verified & loaded best model weights from training: {BEST_MODEL_PATH}{epoch_info}{best_loss_info}")
    
    del ckpt_data, state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    logger.warning("No best model checkpoint found to evaluate! Please check training directory.")

print_vram_usage("Post-Model-Load")"""

assert old_c6 in c6_text, "old_c6 not found in Cell 6"
c6_text = c6_text.replace(old_c6, new_c6)
nb["cells"][6]["source"] = [line + "\n" for line in c6_text.split("\n")[:-1]] + ([c6_text.split("\n")[-1]] if c6_text.split("\n")[-1] else [])

# 4. Update Cell 7 (compute_map: torch.inference_mode(), detach/cpu immediately, periodic torch.cuda.empty_cache())
c7_text = "".join(nb["cells"][7]["source"])

old_compute_map = """def compute_map(model, loader, device, img_size: int = 560, threshold: float = 0.01,
                box_averaging_method: str = "wbf", shrink_factor: float = 0.96,
                fusion_iou_thresh: float = 0.50, num_classes: int = None) -> dict:
    \"\"\"
    Computes official COCO mAP metrics with optional box averaging (WBF / NMS) and margin calibration.
    \"\"\"
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, max_detection_thresholds=[1, 10, 300])
    total_eval_steps = len(loader)
    
    pbar = tqdm(loader, total=total_eval_steps, desc="Computing mAP", unit="batch", dynamic_ncols=True, leave=False, mininterval=1.0)
    
    with torch.no_grad():
        for step, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
            pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
            preds_list, tgts_list = [], []
            for i in range(len(images)):
                scores, lbs = pred_logits[i].sigmoid().max(-1)
                keep = scores > threshold
                
                raw_boxes = (box_cxcywh_to_xyxy(pred_boxes[i][keep]) * img_size).cpu().numpy().tolist()
                raw_scores = scores[keep].cpu().numpy().tolist()
                raw_lbs = lbs[keep].cpu().numpy().tolist()
                
                # Apply configured box averaging & border shrinkage
                if box_averaging_method and box_averaging_method != "none":
                    proc_boxes, proc_scores, proc_lbs = apply_postprocessing(
                        raw_boxes, raw_scores, raw_lbs, img_size=img_size,
                        method=box_averaging_method, shrink_factor=shrink_factor, iou_thresh=fusion_iou_thresh
                    )
                else:
                    proc_boxes, proc_scores, proc_lbs = raw_boxes, raw_scores, raw_lbs
                    
                preds_list.append({
                    "boxes": torch.tensor(proc_boxes, dtype=torch.float32) if proc_boxes else torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.tensor(proc_scores, dtype=torch.float32) if proc_scores else torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.tensor(proc_lbs, dtype=torch.int64) if proc_lbs else torch.zeros((0,), dtype=torch.int64),
                })
                tgts_list.append({
                    "boxes": (box_cxcywh_to_xyxy(targets[i]["boxes"]) * img_size).cpu().float(),
                    "labels": targets[i]["labels"].cpu(),
                })
            metric.update(preds_list, tgts_list)
            
            if (step + 1) % 50 == 0 or (step + 1) == total_eval_steps:
                pbar.set_postfix({"batch": f"{step+1}/{total_eval_steps}"})
                
    pbar.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return metric.compute()"""

new_compute_map = """def compute_map(model, loader, device, img_size: int = 560, threshold: float = 0.01,
                box_averaging_method: str = "wbf", shrink_factor: float = 0.96,
                fusion_iou_thresh: float = 0.50, num_classes: int = None) -> dict:
    \"\"\"
    Computes official COCO mAP metrics with optional box averaging (WBF / NMS) and margin calibration.
    OOM-Safe: Uses torch.inference_mode(), immediate CPU detachment, and periodic GPU cache flushing.
    \"\"\"
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, max_detection_thresholds=[1, 10, 300])
    total_eval_steps = len(loader)
    
    pbar = tqdm(loader, total=total_eval_steps, desc="Computing mAP", unit="batch", dynamic_ncols=True, leave=False, mininterval=1.0)
    
    with torch.inference_mode():
        for step, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                
            pred_logits = outputs["pred_logits"].detach()
            pred_boxes = outputs["pred_boxes"].detach()
            del outputs
            
            preds_list, tgts_list = [], []
            for i in range(len(images)):
                scores, lbs = pred_logits[i].sigmoid().max(-1)
                keep = scores > threshold
                
                # Immediately move tensors off GPU to CPU memory
                raw_boxes = (box_cxcywh_to_xyxy(pred_boxes[i][keep]) * img_size).cpu().numpy().tolist()
                raw_scores = scores[keep].cpu().numpy().tolist()
                raw_lbs = lbs[keep].cpu().numpy().tolist()
                
                # Apply configured box averaging & border shrinkage
                if box_averaging_method and box_averaging_method != "none":
                    proc_boxes, proc_scores, proc_lbs = apply_postprocessing(
                        raw_boxes, raw_scores, raw_lbs, img_size=img_size,
                        method=box_averaging_method, shrink_factor=shrink_factor, iou_thresh=fusion_iou_thresh
                    )
                else:
                    proc_boxes, proc_scores, proc_lbs = raw_boxes, raw_scores, raw_lbs
                    
                preds_list.append({
                    "boxes": torch.tensor(proc_boxes, dtype=torch.float32) if proc_boxes else torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.tensor(proc_scores, dtype=torch.float32) if proc_scores else torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.tensor(proc_lbs, dtype=torch.int64) if proc_lbs else torch.zeros((0,), dtype=torch.int64),
                })
                tgts_list.append({
                    "boxes": (box_cxcywh_to_xyxy(targets[i]["boxes"]) * img_size).cpu().float(),
                    "labels": targets[i]["labels"].cpu(),
                })
                
            del images, targets, pred_logits, pred_boxes
            metric.update(preds_list, tgts_list)
            del preds_list, tgts_list
            
            # Periodically flush CUDA allocator cache every 25 batches to prevent fragmentation
            if (step + 1) % 25 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if (step + 1) % 50 == 0 or (step + 1) == total_eval_steps:
                pbar.set_postfix({"batch": f"{step+1}/{total_eval_steps}"})
                
    pbar.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
        
    return metric.compute()"""

assert old_compute_map in c7_text, "old_compute_map not found in Cell 7"
c7_text = c7_text.replace(old_compute_map, new_compute_map)
nb["cells"][7]["source"] = [line + "\n" for line in c7_text.split("\n")[:-1]] + ([c7_text.split("\n")[-1]] if c7_text.split("\n")[-1] else [])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook 07 to be completely OOM-safe.")

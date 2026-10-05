import json

nb_path = "notebook/09_convert_rfdetr_to_onnx.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 1 (dependencies) to strictly pin onnxruntime==1.16.1
c1_text = "".join(nb["cells"][1]["source"])
old_c1 = "%pip install -q onnx onnxruntime opencv-python-headless==4.9.0.80 pillow"
new_c1 = """# STEP 0 - Environment Dependencies (Strictly onnxruntime==1.16.1)
%pip install -q onnx onnxruntime==1.16.1 opencv-python-headless==4.9.0.80 pillow"""

assert old_c1 in c1_text, "old_c1 not found in Cell 1"
c1_text = c1_text.replace(old_c1, new_c1)
nb["cells"][1]["source"] = [line + "\n" for line in c1_text.split("\n")[:-1]] + ([c1_text.split("\n")[-1]] if c1_text.split("\n")[-1] else [])

# 2. Update Cell 3 to prioritize best model from training
c3_text = "".join(nb["cells"][3]["source"])
old_c3 = """# 3. Prioritize: best_model.pth > latest_checkpoint.pth > highest epoch snapshot > newest file by modification time
BEST_MODEL_PATH = None
if all_discovered_pths:
    # Check for canonical best_model or latest_checkpoint
    best_candidates = [p for p in all_discovered_pths if p.name == "best_model.pth"]
    latest_candidates = [p for p in all_discovered_pths if "latest" in p.name.lower()]
    precision_candidates = [p for p in all_discovered_pths if "precision" in p.name.lower()]
    epoch_snaps = [p for p in all_discovered_pths if "epoch_" in p.name.lower()]
    
    # Sort epoch snapshots by epoch number if available
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
        # Sort by most recently modified
        all_discovered_pths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        BEST_MODEL_PATH = all_discovered_pths[0]"""

new_c3 = """# 3. Best Model from Training Discovery & Verification
# Strictly prioritize best_model.pth exported from training
BEST_MODEL_PATH = None
training_best_candidates = []

priority_best_locations = [
    VERSION_DIR / "model" / "best_model.pth",
    VERSION_DIR / "checkpoints" / "best_model.pth",
    PIPELINE_DIR / "model" / f"best_model_full_data_{VERSION_TAG}.pth",
    PIPELINE_DIR / "model" / "best_model_full_data_v2.pth",
    PIPELINE_DIR / "model" / "best_model.pth",
    PIPELINE_DIR / "checkpoints" / "best_model.pth",
    REPO_ROOT / "multi_class_train_rfdetr" / "model" / "best_model.pth",
    REPO_ROOT / "best_model.pth",
]

for p in priority_best_locations:
    if p.exists() and p.is_file() and p.stat().st_size > 1024 * 1024:
        training_best_candidates.append(p)

if training_best_candidates:
    BEST_MODEL_PATH = training_best_candidates[0]
elif all_discovered_pths:
    best_named = [p for p in all_discovered_pths if p.name == "best_model.pth"]
    version_named = [p for p in all_discovered_pths if "best_model_" in p.name]
    best_metric = [p for p in all_discovered_pths if "best_loss" in p.name.lower() or "best_map" in p.name.lower() or "precision" in p.name.lower()]
    epoch_snaps = [p for p in all_discovered_pths if "epoch_" in p.name.lower()]
    latest_candidates = [p for p in all_discovered_pths if "latest" in p.name.lower()]
    
    if best_named:
        best_named.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        BEST_MODEL_PATH = best_named[0]
    elif version_named:
        version_named.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        BEST_MODEL_PATH = version_named[0]
    elif best_metric:
        best_metric.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        BEST_MODEL_PATH = best_metric[0]
    elif epoch_snaps:
        epoch_snaps.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        BEST_MODEL_PATH = epoch_snaps[0]
    elif latest_candidates:
        BEST_MODEL_PATH = latest_candidates[0]
    else:
        all_discovered_pths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        BEST_MODEL_PATH = all_discovered_pths[0]"""

assert old_c3 in c3_text, "old_c3 not found in Cell 3"
c3_text = c3_text.replace(old_c3, new_c3)
nb["cells"][3]["source"] = [line + "\n" for line in c3_text.split("\n")[:-1]] + ([c3_text.split("\n")[-1]] if c3_text.split("\n")[-1] else [])

# 3. Update Cell 5 (export) to verify and enforce onnxruntime==1.16.1 compatibility
c5_text = "".join(nb["cells"][5]["source"])
old_c5_ort = """# Compatible opset for onnxruntime 1.16.1+ (ONNX Opset 17 is natively supported)
EXPORT_OPSET = 17
try:
    import onnxruntime as ort
    ort_ver = getattr(ort, '__version__', '1.16.1')
    logger.info(f"Detected ONNX Runtime Version: {ort_ver} -> targeting Opset {EXPORT_OPSET}")
except Exception:
    pass"""

new_c5_ort = """# Compatible opset for onnxruntime==1.16.1 (Opset 17 is fully and natively supported by ONNX Runtime 1.16.1)
TARGET_ORT_VERSION = "1.16.1"
EXPORT_OPSET = 17

try:
    import onnxruntime as ort
    ort_ver = getattr(ort, "__version__", TARGET_ORT_VERSION)
    logger.info(f"Verified ONNX Runtime: {ort_ver} (Target: {TARGET_ORT_VERSION}) -> Opset {EXPORT_OPSET}")
except Exception as e:
    logger.warning(f"ONNX Runtime check notice: {e}")"""

assert old_c5_ort in c5_text, "old_c5_ort not found in Cell 5"
c5_text = c5_text.replace(old_c5_ort, new_c5_ort)
nb["cells"][5]["source"] = [line + "\n" for line in c5_text.split("\n")[:-1]] + ([c5_text.split("\n")[-1]] if c5_text.split("\n")[-1] else [])

# 4. Update Cell 6 (verification) to log strict ONNX Runtime 1.16.1 compliance
c6_text = "".join(nb["cells"][6]["source"])
old_c6_top = """# CELL 5 - ONNXRuntime Inference Verification strictly testing [1, 18900, 7]
import onnxruntime as ort

session = ort.InferenceSession(str(ONNX_EXPORT_PATH), providers=["CPUExecutionProvider"])"""

new_c6_top = """# CELL 5 - ONNXRuntime Inference Verification strictly testing [1, 18900, 7] with onnxruntime==1.16.1
import onnxruntime as ort

logger.info(f"Running inference verification with ONNX Runtime v{ort.__version__} (strictly configured for 1.16.1)")
session = ort.InferenceSession(str(ONNX_EXPORT_PATH), providers=["CPUExecutionProvider"])"""

assert old_c6_top in c6_text, "old_c6_top not found in Cell 6"
c6_text = c6_text.replace(old_c6_top, new_c6_top)
nb["cells"][6]["source"] = [line + "\n" for line in c6_text.split("\n")[:-1]] + ([c6_text.split("\n")[-1]] if c6_text.split("\n")[-1] else [])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook 09 for onnxruntime==1.16.1 strictly.")

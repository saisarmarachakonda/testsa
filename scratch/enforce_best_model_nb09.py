import json

nb_path = "notebook/09_convert_rfdetr_to_onnx.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 3
c3_text = "".join(nb["cells"][3]["source"])

old_c3_discovery = """# 3. Best Model from Training Discovery & Verification
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

new_c3_discovery = """# 3. Best Model from Training Discovery & Strict Verification
# RULE: Step 09 ONNX conversion MUST strictly export only the best trained checkpoint: 'best_model.pth'
BEST_MODEL_PATH = None

# Prioritized locations strictly for 'best_model.pth'
strict_best_model_locations = [
    VERSION_DIR / "model" / "best_model.pth",
    VERSION_DIR / "checkpoints" / "best_model.pth",
    PIPELINE_DIR / "model" / "best_model.pth",
    PIPELINE_DIR / "checkpoints" / "best_model.pth",
    REPO_ROOT / PIPELINE_NAME / "model" / "best_model.pth",
    REPO_ROOT / PIPELINE_NAME / "checkpoints" / "best_model.pth",
    REPO_ROOT / "multi_class_train_rfdetr" / "model" / "best_model.pth",
    REPO_ROOT / "multi_class_train_rfdetr" / "checkpoints" / "best_model.pth",
    Path("/home/jupyter/location_tag/multi_class_rfdetr/v2/model/best_model.pth"),
    Path("/home/jupyter/location_tag/multi_class_rfdetr/v2/checkpoints/best_model.pth"),
    Path("/home/jupyter/multi_class_rfdetr/v2/model/best_model.pth"),
    Path("/home/jupyter/multi_class_rfdetr/v2/checkpoints/best_model.pth"),
    REPO_ROOT / "model" / "best_model.pth",
    REPO_ROOT / "best_model.pth",
    Path("model/best_model.pth"),
    Path("checkpoints/best_model.pth")
]

for cand in strict_best_model_locations:
    if cand.exists() and cand.is_file() and cand.stat().st_size > 1024 * 1024:
        BEST_MODEL_PATH = cand
        break

# Secondary search: any discovered .pth strictly named 'best_model.pth'
if BEST_MODEL_PATH is None and all_discovered_pths:
    exact_best = [p for p in all_discovered_pths if p.name == "best_model.pth"]
    if exact_best:
        exact_best.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        BEST_MODEL_PATH = exact_best[0]

# Assertion check: enforce that only best_model.pth is selected
if BEST_MODEL_PATH is None:
    raise FileNotFoundError(
        "CRITICAL ERROR: 'best_model.pth' not found! Notebook 09 requires 'best_model.pth' from training. "
        "Please complete training or verify that 'best_model.pth' exists in the model/checkpoints directory."
    )
elif BEST_MODEL_PATH.name != "best_model.pth":
    raise ValueError(
        f"CRITICAL ERROR: Selected model '{BEST_MODEL_PATH}' is not 'best_model.pth'! "
        "Notebook 09 must always export strictly 'best_model.pth'."
    )

logger.info(f"✓ Confirmed strictly using 'best_model.pth': {BEST_MODEL_PATH} ({BEST_MODEL_PATH.stat().st_size / 1e6:.1f} MB)")"""

assert old_c3_discovery in c3_text, "old_c3_discovery not found in Cell 3"
c3_text = c3_text.replace(old_c3_discovery, new_c3_discovery)
nb["cells"][3]["source"] = [line + "\n" for line in c3_text.split("\n")[:-1]] + ([c3_text.split("\n")[-1]] if c3_text.split("\n")[-1] else [])

# Update Cell 4 to log exact best_model metadata
c4_text = "".join(nb["cells"][4]["source"])

old_c4_weights = """# Load latest trained model weights directly into the initialized architecture
target_weights = globals().get("BEST_MODEL_PATH", None)
if target_weights and os.path.exists(str(target_weights)):
    try:
        ckpt_data = torch.load(str(target_weights), map_location="cpu", weights_only=False)
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        model.model.model.load_state_dict(state_dict, strict=False)
        logger.info(f"✓ Successfully loaded trained model weights from: {target_weights}")
    except Exception as e:
        logger.warning(f"Could not load trained weights from {target_weights}: {e}")
else:
    logger.warning("No trained weights found on disk. Initialized with fresh detection head.")"""

new_c4_weights = """# Load best_model.pth weights strictly into the initialized architecture
target_weights = globals().get("BEST_MODEL_PATH", None)
assert target_weights and os.path.exists(str(target_weights)), f"Cannot proceed: best_model.pth missing at {target_weights}"
assert Path(target_weights).name == "best_model.pth", f"Invalid model file: {target_weights}. Must strictly be 'best_model.pth'!"

ckpt_data = torch.load(str(target_weights), map_location="cpu", weights_only=False)
state_dict = ckpt_data.get("model_state_dict", ckpt_data)
model.model.model.load_state_dict(state_dict, strict=False)

epoch_info = f" (Epoch {ckpt_data['epoch']})" if isinstance(ckpt_data, dict) and 'epoch' in ckpt_data else ""
loss_info = f" | Best Val Loss: {ckpt_data.get('best_loss', ckpt_data.get('val_loss', 'N/A'))}" if isinstance(ckpt_data, dict) and ('best_loss' in ckpt_data or 'val_loss' in ckpt_data) else ""
logger.info(f"✓ Strictly verified and loaded weights from best_model.pth: {target_weights}{epoch_info}{loss_info}")"""

assert old_c4_weights in c4_text, "old_c4_weights not found in Cell 4"
c4_text = c4_text.replace(old_c4_weights, new_c4_weights)
nb["cells"][4]["source"] = [line + "\n" for line in c4_text.split("\n")[:-1]] + ([c4_text.split("\n")[-1]] if c4_text.split("\n")[-1] else [])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook 09 to strictly require and load only 'best_model.pth'.")

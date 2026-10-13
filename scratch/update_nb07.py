import json
from pathlib import Path

nb_path = "notebook/07_evaluate_rfdetr_clean_data.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 3
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
# Training exports the best model checkpoint as 'best_model.pth' and 'best_model_{RUN_ID}.pth'
# in FINAL_MODEL_DIR (e.g. multi_class_rfdetr/v2/model/) and CHECKPOINT_DIR (multi_class_rfdetr/v2/checkpoints/).
# Prioritize the true best model checkpoint from training:
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

# Update Cell 6
c6_text = "".join(nb["cells"][6]["source"])
old_c6 = """if BEST_MODEL_PATH and os.path.exists(BEST_MODEL_PATH):
    ckpt_data = torch.load(BEST_MODEL_PATH, map_location=device)
    state = ckpt_data.get("model_state_dict", ckpt_data)
    lwdetr.load_state_dict(state, strict=False)
    logger.info(f"✓ Loaded evaluation weights from: {BEST_MODEL_PATH}")"""

new_c6 = """if BEST_MODEL_PATH and os.path.exists(str(BEST_MODEL_PATH)):
    ckpt_data = torch.load(str(BEST_MODEL_PATH), map_location=device, weights_only=False)
    state = ckpt_data.get("model_state_dict", ckpt_data)
    lwdetr.load_state_dict(state, strict=False)
    epoch_info = f" (Epoch {ckpt_data['epoch']})" if isinstance(ckpt_data, dict) and 'epoch' in ckpt_data else ""
    loss_val = ckpt_data.get('best_loss', ckpt_data.get('val_loss', 'N/A')) if isinstance(ckpt_data, dict) else 'N/A'
    best_loss_info = f" | Best Val Loss: {loss_val}" if loss_val != 'N/A' else ""
    logger.info(f"✓ Verified & loaded best model weights from training: {BEST_MODEL_PATH}{epoch_info}{best_loss_info}")
else:
    logger.warning("No best model checkpoint found to evaluate! Please check training directory.")"""

assert old_c6 in c6_text, "old_c6 not found in Cell 6"
c6_text = c6_text.replace(old_c6, new_c6)
nb["cells"][6]["source"] = [line + "\n" for line in c6_text.split("\n")[:-1]] + ([c6_text.split("\n")[-1]] if c6_text.split("\n")[-1] else [])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated 07 notebook successfully.")

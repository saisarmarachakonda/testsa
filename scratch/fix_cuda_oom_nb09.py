import json

nb_path = "notebook/09_convert_rfdetr_to_onnx.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 4: Instantiate and keep model exclusively on CPU, purge GPU cache completely
c4_text = "".join(nb["cells"][4]["source"])

old_c4_tail = """# Load best_model.pth weights strictly into the initialized architecture
target_weights = globals().get("BEST_MODEL_PATH", None)
assert target_weights and os.path.exists(str(target_weights)), f"Cannot proceed: best_model.pth missing at {target_weights}"
assert Path(target_weights).name == "best_model.pth", f"Invalid model file: {target_weights}. Must strictly be 'best_model.pth'!"

ckpt_data = torch.load(str(target_weights), map_location="cpu", weights_only=False)
state_dict = ckpt_data.get("model_state_dict", ckpt_data)
model.model.model.load_state_dict(state_dict, strict=False)

epoch_info = f" (Epoch {ckpt_data['epoch']})" if isinstance(ckpt_data, dict) and 'epoch' in ckpt_data else ""
loss_info = f" | Best Val Loss: {ckpt_data.get('best_loss', ckpt_data.get('val_loss', 'N/A'))}" if isinstance(ckpt_data, dict) and ('best_loss' in ckpt_data or 'val_loss' in ckpt_data) else ""
logger.info(f"✓ Strictly verified and loaded weights from best_model.pth: {target_weights}{epoch_info}{loss_info}")

# Extract underlying model module and assign default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lwdetr = model.model.model if hasattr(model, "model") and hasattr(model.model, "model") else getattr(model, "model", model)
lwdetr.to(device)
logger.info(f"Model underlying LWDETR assigned to device: {device}")"""

new_c4_tail = """# Proactively flush any residual CUDA VRAM allocations from preceding notebook steps
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load best_model.pth weights strictly into the initialized architecture on CPU
target_weights = globals().get("BEST_MODEL_PATH", None)
assert target_weights and os.path.exists(str(target_weights)), f"Cannot proceed: best_model.pth missing at {target_weights}"
assert Path(target_weights).name == "best_model.pth", f"Invalid model file: {target_weights}. Must strictly be 'best_model.pth'!"

ckpt_data = torch.load(str(target_weights), map_location="cpu", weights_only=False)
state_dict = ckpt_data.get("model_state_dict", ckpt_data)
model.model.model.load_state_dict(state_dict, strict=False)

epoch_info = f" (Epoch {ckpt_data['epoch']})" if isinstance(ckpt_data, dict) and 'epoch' in ckpt_data else ""
loss_info = f" | Best Val Loss: {ckpt_data.get('best_loss', ckpt_data.get('val_loss', 'N/A'))}" if isinstance(ckpt_data, dict) and ('best_loss' in ckpt_data or 'val_loss' in ckpt_data) else ""
logger.info(f"✓ Strictly verified and loaded weights from best_model.pth: {target_weights}{epoch_info}{loss_info}")

del ckpt_data, state_dict
gc.collect()

# Extract underlying model module on CPU (Zero GPU VRAM usage during ONNX conversion)
device = torch.device("cpu")
lwdetr = model.model.model if hasattr(model, "model") and hasattr(model.model, "model") else getattr(model, "model", model)
lwdetr.to(device)
lwdetr.eval()

# Delete outer model wrapper
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

logger.info("✓ Model initialized on CPU for zero-VRAM ONNX graph compilation.")"""

assert old_c4_tail in c4_text, "old_c4_tail not found in Cell 4"
c4_text = c4_text.replace(old_c4_tail, new_c4_tail)
nb["cells"][4]["source"] = [line + "\n" for line in c4_text.split("\n")[:-1]] + ([c4_text.split("\n")[-1]] if c4_text.split("\n")[-1] else [])

# 2. Update Cell 5: Ensure device is strictly CPU and CUDA memory is explicitly purged
c5_text = "".join(nb["cells"][5]["source"])

old_c5_device_block = """device = torch.device("cpu")
lwdetr = model.model.model
lwdetr.to(device).eval()

wrapper = TagDetRTProductionWrapper(
    lwdetr,
    internal_size=INTERNAL_RESOLUTION,
    total_rows=TOTAL_OUTPUT_ROWS,
    num_classes=OUTPUT_COLS - 4
)
wrapper.eval()"""

new_c5_device_block = """# CPU Export Pipeline: Prevents any CUDA Out-of-Memory errors during ONNX symbolic graph expansion
device = torch.device("cpu")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

wrapper = TagDetRTProductionWrapper(
    lwdetr,
    internal_size=INTERNAL_RESOLUTION,
    total_rows=TOTAL_OUTPUT_ROWS,
    num_classes=OUTPUT_COLS - 4
)
wrapper.to(device)
wrapper.eval()"""

assert old_c5_device_block in c5_text, "old_c5_device_block not found in Cell 5"
c5_text = c5_text.replace(old_c5_device_block, new_c5_device_block)
nb["cells"][5]["source"] = [line + "\n" for line in c5_text.split("\n")[:-1]] + ([c5_text.split("\n")[-1]] if c5_text.split("\n")[-1] else [])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook 09 for zero-VRAM CPU export successfully.")

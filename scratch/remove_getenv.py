import json

# 1. Update notebook 10
nb10_path = "notebook/10_evaluate_compare_rfdetr_multiclass.ipynb"
with open(nb10_path, "r", encoding="utf-8") as f:
    nb10 = json.load(f)

# Replace os.getenv in Cell 3 of Notebook 10
c3_10 = "".join(nb10["cells"][3]["source"])
old_c3_10_env = 'MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")'
new_c3_10_env = 'MODEL_VERSION = "v2"  # Hardcoded: strictly avoids loading unverified settings from environment'

assert old_c3_10_env in c3_10, "old_c3_10_env not found in Cell 3 of nb 10"
c3_10 = c3_10.replace(old_c3_10_env, new_c3_10_env)
nb10["cells"][3]["source"] = [line + "\n" for line in c3_10.split("\n")[:-1]] + ([c3_10.split("\n")[-1]] if c3_10.split("\n")[-1] else [])

# Update Cell 7 in Notebook 10: Robust try/except OOM recovery and per-step cleanup
c7_10 = "".join(nb10["cells"][7]["source"])

old_c7_forward_block = """            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                out = rfdetr_model(batch_tensors)
                
            batch_logits = out["pred_logits"].detach().cpu()   # Immediately move predictions to CPU
            batch_boxes = out["pred_boxes"].detach().cpu()     # Immediately move predictions to CPU
            del out, batch_tensors"""

new_c7_forward_block = """            try:
                with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    out = rfdetr_model(batch_tensors)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    logger.warning(f"[VRAM RECOVERY] CUDA OOM at step {step_count}. Flushing allocator cache and executing on CPU fallback...")
                    torch.cuda.empty_cache()
                    # Execute single sample on CPU if GPU runs out of memory
                    out = rfdetr_model.to("cpu")(batch_tensors.cpu())
                    rfdetr_model.to(device)
                else:
                    raise e
                    
            batch_logits = out["pred_logits"].detach().cpu()   # Immediately move predictions to CPU
            batch_boxes = out["pred_boxes"].detach().cpu()     # Immediately move predictions to CPU
            del out, batch_tensors"""

assert old_c7_forward_block in c7_10, "old_c7_forward_block not found in Cell 7 of nb 10"
c7_10 = c7_10.replace(old_c7_forward_block, new_c7_forward_block)
nb10["cells"][7]["source"] = [line + "\n" for line in c7_10.split("\n")[:-1]] + ([c7_10.split("\n")[-1]] if c7_10.split("\n")[-1] else [])

with open(nb10_path, "w", encoding="utf-8") as f:
    json.dump(nb10, f, indent=1)

print("Updated notebook 10 successfully.")

# 2. Update notebooks 07, 08, 09 to eliminate os.getenv
for nb_path in [
    "notebook/07_evaluate_rfdetr_clean_data.ipynb",
    "notebook/08_evaluate_anomalies_audit.ipynb",
    "notebook/09_convert_rfdetr_to_onnx.ipynb"
]:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    # Cell 3
    c3 = "".join(nb["cells"][3]["source"])
    if 'MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")' in c3:
        c3 = c3.replace('MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")', 'MODEL_VERSION = "v2"  # Direct configuration: no environment override')
        nb["cells"][3]["source"] = [line + "\n" for line in c3.split("\n")[:-1]] + ([c3.split("\n")[-1]] if c3.split("\n")[-1] else [])
        
    # Cell 4
    c4 = "".join(nb["cells"][4]["source"])
    if 'MODEL_SIZE = globals().get("MODEL_SIZE", os.getenv("MODEL_SIZE", "base"))' in c4:
        c4 = c4.replace('MODEL_SIZE = globals().get("MODEL_SIZE", os.getenv("MODEL_SIZE", "base"))', 'MODEL_SIZE = globals().get("MODEL_SIZE", "base")  # Direct architecture configuration')
        nb["cells"][4]["source"] = [line + "\n" for line in c4.split("\n")[:-1]] + ([c4.split("\n")[-1]] if c4.split("\n")[-1] else [])

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {nb_path} to remove os.getenv successfully.")

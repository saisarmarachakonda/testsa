import json

nb_path = "notebook/10_evaluate_compare_rfdetr_multiclass.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 3: Set RESOLUTION = 560 (or auto-aligned), EVAL_BATCH_SIZE = 1, NUM_EVAL_WORKERS = 0
c3_text = "".join(nb["cells"][3]["source"])

old_c3_res_block = """MODEL_SIZE = "base"
TARGET_RESOLUTION = 1008
DIVISOR = 56 if MODEL_SIZE == "base" else 32
RESOLUTION = max(round(TARGET_RESOLUTION / DIVISOR) * DIVISOR, DIVISOR)

# Parameterized Confidence Threshold for New Model (e.g., 0.15, 0.25, 0.45, 0.50)
CONFIDENCE_THRESHOLD = 0.51
IOU_THRESHOLD = 0.50"""

new_c3_res_block = """MODEL_SIZE = "base"
# Model was trained at resolution 560x560 (divisible by patch size 14 and backbone divisor 56: 56 * 10 = 560).
# Using 560 avoids catastrophic CUDA OOM during deformable attention and query cross-attention.
TARGET_RESOLUTION = 560
DIVISOR = 56 if MODEL_SIZE == "base" else 32
RESOLUTION = max(round(TARGET_RESOLUTION / DIVISOR) * DIVISOR, DIVISOR)

# Parameterized Confidence Threshold for New Model (e.g., 0.15, 0.25, 0.45, 0.50)
CONFIDENCE_THRESHOLD = 0.51
IOU_THRESHOLD = 0.50"""

assert old_c3_res_block in c3_text, "old_c3_res_block not found in Cell 3"
c3_text = c3_text.replace(old_c3_res_block, new_c3_res_block)

old_c3_eval_batch = """# OOM-Safe High-Throughput Inference & Batching Settings
# Setting pin_memory=False and workers=0 prevents "RuntimeError: CUDA error: out of memory in pin memory thread"
EVAL_BATCH_SIZE = 2       # Conservative batch size (2 avoids CUDA allocation spikes at 1008x1008)
NUM_EVAL_WORKERS = 0      # 0 workers avoids host pinned memory depletion and IPC synchronization stalls"""

new_c3_eval_batch = """# OOM-Safe High-Throughput Inference & Batching Settings
# Batch size = 1 strictly prevents CUDA out of memory errors during ViT windowed attention & query generation
EVAL_BATCH_SIZE = 1       # Batch size 1 guarantees zero OOM on all GPU tiers
NUM_EVAL_WORKERS = 0      # 0 workers avoids host pinned memory depletion and IPC synchronization stalls"""

assert old_c3_eval_batch in c3_text, "old_c3_eval_batch not found in Cell 3"
c3_text = c3_text.replace(old_c3_eval_batch, new_c3_eval_batch)

nb["cells"][3]["source"] = [line + "\n" for line in c3_text.split("\n")[:-1]] + ([c3_text.split("\n")[-1]] if c3_text.split("\n")[-1] else [])

# 2. Update Cell 7: Ensure batch_tensors are deleted immediately, outputs detached, and cache purged every step if needed
c7_text = "".join(nb["cells"][7]["source"])

old_c7_forward = """            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                out = rfdetr_model(batch_tensors)
                
            batch_logits = out["pred_logits"].detach()   # [B, queries, num_classes]
            batch_boxes = out["pred_boxes"].detach()     # [B, queries, 4]
            del out, batch_tensors"""

new_c7_forward = """            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                out = rfdetr_model(batch_tensors)
                
            batch_logits = out["pred_logits"].detach().cpu()   # Immediately move predictions to CPU
            batch_boxes = out["pred_boxes"].detach().cpu()     # Immediately move predictions to CPU
            del out, batch_tensors"""

assert old_c7_forward in c7_text, "old_c7_forward not found in Cell 7"
c7_text = c7_text.replace(old_c7_forward, new_c7_forward)

# Update periodic empty_cache to run every 10 steps
old_c7_cache = """            # Periodically flush CUDA allocator cache every 25 batches to prevent fragmentation
            if step_count % 25 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()"""

new_c7_cache = """            # Periodically flush CUDA allocator cache every 10 batches to prevent fragmentation
            if step_count % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()"""

assert old_c7_cache in c7_text, "old_c7_cache not found in Cell 7"
c7_text = c7_text.replace(old_c7_cache, new_c7_cache)

nb["cells"][7]["source"] = [line + "\n" for line in c7_text.split("\n")[:-1]] + ([c7_text.split("\n")[-1]] if c7_text.split("\n")[-1] else [])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook 10: RESOLUTION=560, EVAL_BATCH_SIZE=1, CPU-detached output tensors.")

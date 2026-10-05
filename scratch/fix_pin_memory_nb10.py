import json

nb_path = "notebook/10_evaluate_compare_rfdetr_multiclass.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 3: Set EVAL_BATCH_SIZE = 2, NUM_EVAL_WORKERS = 0 to prevent pin_memory thread OOM
c3_text = "".join(nb["cells"][3]["source"])

old_c3_batching = """EVAL_BATCH_SIZE = 4       # Mini-batch size (e.g. 4 or 8 for 6-12+ FPS; 2 for low VRAM)
NUM_EVAL_WORKERS = 4 if torch.cuda.is_available() else 0  # Background CPU workers for asynchronous pre-fetching"""

new_c3_batching = """# OOM-Safe High-Throughput Inference & Batching Settings
# Setting pin_memory=False and workers=0 prevents "RuntimeError: CUDA error: out of memory in pin memory thread"
EVAL_BATCH_SIZE = 2       # Conservative batch size (2 avoids CUDA allocation spikes at 1008x1008)
NUM_EVAL_WORKERS = 0      # 0 workers avoids host pinned memory depletion and IPC synchronization stalls"""

assert old_c3_batching in c3_text, "old_c3_batching not found in Cell 3"
c3_text = c3_text.replace(old_c3_batching, new_c3_batching)
nb["cells"][3]["source"] = [line + "\n" for line in c3_text.split("\n")[:-1]] + ([c3_text.split("\n")[-1]] if c3_text.split("\n")[-1] else [])

# 2. Update Cell 6: Discard unused model wrapper and clean CUDA cache immediately after loading LWDETR
c6_text = "".join(nb["cells"][6]["source"])

old_c6_tail = """        load_result = lwdetr.load_state_dict(state, strict=False)
        logger.info(f"Fine-tuned weights loaded into LWDETR: {load_result}")
        
        lwdetr.to(device)
        lwdetr.eval()
        rfdetr_model = lwdetr
        
        param_count = sum(p.numel() for p in rfdetr_model.parameters()) / 1e6
        logger.info(f"RF-DETR Multi-Class Model Ready on {device} ({param_count:.1f}M parameters).")
        
    except Exception as e:
        logger.error(f"Failed to load RF-DETR model: {e}")
        rfdetr_model = None
else:
    logger.warning(f"RF-DETR checkpoint not found at: {RFDETR_CHECKPOINT_PATH}")"""

new_c6_tail = """        load_result = lwdetr.load_state_dict(state, strict=False)
        logger.info(f"Fine-tuned weights loaded into LWDETR: {load_result}")
        
        # Discard outer wrappers and checkpoint dictionaries to recover GPU/host memory
        del wrapper, ckpt, state
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        lwdetr.to(device)
        lwdetr.eval()
        rfdetr_model = lwdetr
        
        param_count = sum(p.numel() for p in rfdetr_model.parameters()) / 1e6
        logger.info(f"RF-DETR Multi-Class Model Ready on {device} ({param_count:.1f}M parameters).")
        
    except Exception as e:
        logger.error(f"Failed to load RF-DETR model: {e}")
        rfdetr_model = None
else:
    logger.warning(f"RF-DETR checkpoint not found at: {RFDETR_CHECKPOINT_PATH}")"""

assert old_c6_tail in c6_text, "old_c6_tail not found in Cell 6"
c6_text = c6_text.replace(old_c6_tail, new_c6_tail)
nb["cells"][6]["source"] = [line + "\n" for line in c6_text.split("\n")[:-1]] + ([c6_text.split("\n")[-1]] if c6_text.split("\n")[-1] else [])

# 3. Update Cell 7: In evaluate_rfdetr, set pin_memory=False, num_workers=0, detach tensors, and periodically flush CUDA cache
c7_text = "".join(nb["cells"][7]["source"])

old_c7_loader = """    # Initialize PyTorch DataLoader for asynchronous multi-worker prefetching
    eval_dataset = RFDETRTestDataset(test_samples, resolution=RESOLUTION)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )
    
    t0 = time.time()
    
    with torch.inference_mode():
        for batch_tensors, orig_ws, orig_hs, sample_indices in tqdm(
            eval_loader,
            desc=f"RF-DETR Fast Eval (batch={batch_size})",
            unit="img",
            unit_scale=batch_size
        ):
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                out = rfdetr_model(batch_tensors)
                
            batch_logits = out["pred_logits"]   # [B, queries, num_classes]
            batch_boxes = out["pred_boxes"]     # [B, queries, 4]"""

new_c7_loader = """    # Pre-clean GPU cache before launching evaluation loader
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize PyTorch DataLoader (OOM-Safe: pin_memory=False prevents CUDA memory exhaustion in background thread)
    eval_dataset = RFDETRTestDataset(test_samples, resolution=RESOLUTION)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    
    t0 = time.time()
    step_count = 0
    
    with torch.inference_mode():
        for batch_tensors, orig_ws, orig_hs, sample_indices in tqdm(
            eval_loader,
            desc=f"RF-DETR Fast Eval (batch={batch_size})",
            unit="img",
            unit_scale=batch_size
        ):
            step_count += 1
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                out = rfdetr_model(batch_tensors)
                
            batch_logits = out["pred_logits"].detach()   # [B, queries, num_classes]
            batch_boxes = out["pred_boxes"].detach()     # [B, queries, 4]
            del out, batch_tensors"""

assert old_c7_loader in c7_text, "old_c7_loader not found in Cell 7"
c7_text = c7_text.replace(old_c7_loader, new_c7_loader)

# Also add periodic cache flushing inside loop
old_c7_end_loop = """                eval_results.append({
                    "image_id": img_id,
                    "file_name": file_name,
                    "gt_boxes": gt_boxes,
                    "gt_classes": gt_classes,
                    "annotated_preds": annotated_preds
                })
                
    duration = time.time() - t0"""

new_c7_end_loop = """                eval_results.append({
                    "image_id": img_id,
                    "file_name": file_name,
                    "gt_boxes": gt_boxes,
                    "gt_classes": gt_classes,
                    "annotated_preds": annotated_preds
                })
            
            # Periodically flush CUDA allocator cache every 25 batches to prevent fragmentation
            if step_count % 25 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    duration = time.time() - t0"""

assert old_c7_end_loop in c7_text, "old_c7_end_loop not found in Cell 7"
c7_text = c7_text.replace(old_c7_end_loop, new_c7_end_loop)

nb["cells"][7]["source"] = [line + "\n" for line in c7_text.split("\n")[:-1]] + ([c7_text.split("\n")[-1]] if c7_text.split("\n")[-1] else [])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook 10 to resolve pin_memory CUDA OOM successfully.")

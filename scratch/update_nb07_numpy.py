import json

nb_path = "notebook/07_evaluate_rfdetr_clean_data.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 1: Environment dependencies with numpy<2.0.0 pinned
new_c1 = """# STEP 0 - Environment Dependencies (Strictly numpy<2.0 for ABI compatibility)
# Pinning numpy<2.0.0 completely prevents 'AttributeError: _ARRAY_API not found' and 'numpy.core.multiarray failed to import'
%pip install -q "numpy<2.0.0" opencv-python-headless==4.9.0.80 pillow python-dotenv
"""
nb["cells"][1]["source"] = [line + "\n" for line in new_c1.split("\n")[:-1]] + ([new_c1.split("\n")[-1]] if new_c1.split("\n")[-1] else [])

# 2. Update Cell 2: Decouple non-essential C-extensions and protect cv2/supervision imports
c2_text = "".join(nb["cells"][2]["source"])

old_c2_import = "import cv2, torch, requests, supervision as sv"
new_c2_import = """import torch

# Decoupled optional third-party C-extensions to avoid binary collisions
try:
    import cv2
except Exception:
    cv2 = None

try:
    import requests
except Exception:
    pass

try:
    import supervision as sv
except Exception:
    sv = None"""

assert old_c2_import in c2_text, "old_c2_import not found in Cell 2"
c2_text = c2_text.replace(old_c2_import, new_c2_import)
nb["cells"][2]["source"] = [line + "\n" for line in c2_text.split("\n")[:-1]] + ([c2_text.split("\n")[-1]] if c2_text.split("\n")[-1] else [])

# 3. Update Cell 5: Make image decoding resilient with OpenCV primary + PIL seamless fallback
c5_text = "".join(nb["cells"][5]["source"])

old_c5_decode = """        cv_img = cv2.imread(img_path)
        if cv_img is None:
            if idx not in self._warned_missing:
                self._warned_missing.add(idx)
                print(f"[DATASET NOTICE] Corrupt image {img_path}. Falling back to next sample.")
            return self.__getitem__((idx + 1) % len(self))

        orig_h, orig_w = cv_img.shape[:2]
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if (x <= 1.05 and y <= 1.05 and w <= 1.05 and h <= 1.05) and (w > 0 or h > 0) and (orig_w > 10 and orig_h > 10):
                x, y, w, h = x * orig_w, y * orig_h, w * orig_w, h * orig_h
            if w > 0 and h > 0:
                boxes.append([x, y, w, h])
                labels.append(self.cat_to_id.get(ann["category_id"], 0))

        if self.use_augmentation and len(boxes) > 0:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                boxes = [[orig_w - (bx + bw), by, bw, bh] for bx, by, bw, bh in boxes]
            if random.random() < 0.3:
                alpha = random.uniform(0.8, 1.2)
                beta = random.randint(-15, 15)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        img_resized = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        # Robust array-to-tensor conversion: handles numpy 1.x/2.x reload boundaries and non-contiguous buffers
        img_arr = np.ascontiguousarray(np.asarray(img_resized, dtype=np.uint8))
        try:
            img_t = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
        except TypeError:
            img_t = torch.as_tensor(img_arr, dtype=torch.float32).permute(2, 0, 1) / 255.0"""

new_c5_decode = """        # Resilient image loader: uses OpenCV if available; falls back to pure PIL
        img = None
        orig_w, orig_h = 0, 0
        
        if cv2 is not None:
            try:
                cv_img = cv2.imread(img_path)
                if cv_img is not None:
                    orig_h, orig_w = cv_img.shape[:2]
                    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            except Exception:
                img = None
                
        if img is None:
            try:
                with Image.open(img_path) as pil_im:
                    pil_rgb = pil_im.convert("RGB")
                    orig_w, orig_h = pil_rgb.size
                    img = np.array(pil_rgb, dtype=np.uint8)
            except Exception:
                if idx not in self._warned_missing:
                    self._warned_missing.add(idx)
                    print(f"[DATASET NOTICE] Corrupt or unreadable image {img_path}. Falling back to next sample.")
                return self.__getitem__((idx + 1) % len(self))

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if (x <= 1.05 and y <= 1.05 and w <= 1.05 and h <= 1.05) and (w > 0 or h > 0) and (orig_w > 10 and orig_h > 10):
                x, y, w, h = x * orig_w, y * orig_h, w * orig_w, h * orig_h
            if w > 0 and h > 0:
                boxes.append([x, y, w, h])
                labels.append(self.cat_to_id.get(ann["category_id"], 0))

        if self.use_augmentation and len(boxes) > 0:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                boxes = [[orig_w - (bx + bw), by, bw, bh] for bx, by, bw, bh in boxes]
            if random.random() < 0.3:
                alpha = random.uniform(0.8, 1.2)
                beta = random.randint(-15, 15)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        if cv2 is not None:
            try:
                img_resized = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
            except Exception:
                img_resized = np.array(Image.fromarray(img).resize((self.resolution, self.resolution), Image.Resampling.BILINEAR), dtype=np.uint8)
        else:
            img_resized = np.array(Image.fromarray(img).resize((self.resolution, self.resolution), Image.Resampling.BILINEAR), dtype=np.uint8)

        # Robust array-to-tensor conversion: handles numpy 1.x/2.x reload boundaries and non-contiguous buffers
        img_arr = np.ascontiguousarray(np.asarray(img_resized, dtype=np.uint8))
        try:
            img_t = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
        except Exception:
            img_t = torch.as_tensor(img_arr, dtype=torch.float32).permute(2, 0, 1) / 255.0"""

assert old_c5_decode in c5_text, "old_c5_decode not found in Cell 5"
c5_text = c5_text.replace(old_c5_decode, new_c5_decode)
nb["cells"][5]["source"] = [line + "\n" for line in c5_text.split("\n")[:-1]] + ([c5_text.split("\n")[-1]] if c5_text.split("\n")[-1] else [])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook 07 with numpy<2.0 and resilient PIL/OpenCV image decoding.")

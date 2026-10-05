import json

nb_path = "notebook/09_convert_rfdetr_to_onnx.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 1: Ensure numpy is pinned to <2.0 (compatible with PyTorch 2.x and onnxruntime 1.16.1), and remove opencv install since it's not needed
new_c1 = """# STEP 0 - Environment Dependencies (Strictly onnxruntime==1.16.1 & numpy<2.0)
# Note: numpy<2.0 prevents 'AttributeError: _ARRAY_API not found' / 'numpy.core.multiarray failed to import'
%pip install -q onnx onnxruntime==1.16.1 "numpy<2.0.0" pillow
"""
nb["cells"][1]["source"] = [line + "\n" for line in new_c1.split("\n")[:-1]] + ([new_c1.split("\n")[-1]] if new_c1.split("\n")[-1] else [])

# 2. Update Cell 2: Replace 'import cv2, torch, requests, supervision as sv' with safe 'import torch' and optional cv2 import inside try/except
c2_text = "".join(nb["cells"][2]["source"])

old_c2_import = "import cv2, torch, requests, supervision as sv"
new_c2_import = """import torch
# Notebook 09 performs pure PyTorch -> ONNX tensor tracing with torch.randn dummy tensors.
# cv2 and supervision are optional and wrapped in try/except to prevent C-extension binary collisions.
try:
    import cv2
except Exception:
    pass

try:
    import requests
except Exception:
    pass

try:
    import supervision as sv
except Exception:
    pass"""

assert old_c2_import in c2_text, "old_c2_import not found in Cell 2"
c2_text = c2_text.replace(old_c2_import, new_c2_import)
nb["cells"][2]["source"] = [line + "\n" for line in c2_text.split("\n")[:-1]] + ([c2_text.split("\n")[-1]] if c2_text.split("\n")[-1] else [])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook 09: numpy<2.0 pinned and cv2 import protected.")

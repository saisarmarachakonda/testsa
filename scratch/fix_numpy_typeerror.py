import json

for nb_path in ["notebook/07_evaluate_rfdetr_clean_data.ipynb", "notebook/06_predict_rfdetr_images.ipynb"]:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    c5_text = "".join(nb["cells"][5]["source"])
    
    old_target = """        img_resized = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0"""

    new_target = """        img_resized = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        # Robust array-to-tensor conversion: handles numpy 1.x/2.x reload boundaries and non-contiguous buffers
        img_arr = np.ascontiguousarray(np.asarray(img_resized, dtype=np.uint8))
        try:
            img_t = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
        except TypeError:
            img_t = torch.as_tensor(img_arr, dtype=torch.float32).permute(2, 0, 1) / 255.0"""

    assert old_target in c5_text, f"old_target not found in {nb_path}"
    c5_text = c5_text.replace(old_target, new_target)
    nb["cells"][5]["source"] = [line + "\n" for line in c5_text.split("\n")[:-1]] + ([c5_text.split("\n")[-1]] if c5_text.split("\n")[-1] else [])

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Updated {nb_path} with robust tensor conversion.")

import json

nb_path = "notebook/09_convert_rfdetr_to_onnx.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c5_text = "".join(nb["cells"][5]["source"])

old_c5_export_call = """# Perform ONNX export under antialias-safe tracing context
with torch.no_grad(), disable_interpolate_antialias():
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(ONNX_EXPORT_PATH),
        input_names=[TRITON_INPUT_NAME],
        output_names=[TRITON_OUTPUT_NAME],
        opset_version=EXPORT_OPSET,
        do_constant_folding=True
    )
logger.info(f"✓ ONNX export complete! Size: {ONNX_EXPORT_PATH.stat().st_size / 1e6:.1f} MB (Opset {EXPORT_OPSET})")"""

new_c5_export_call = """# Suppress standard PyTorch TracerWarnings for clean console logs
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
logger.info("Exporting ONNX graph (TracerWarnings suppressed for clean output)...")

# Perform ONNX export under antialias-safe tracing context
with torch.no_grad(), disable_interpolate_antialias():
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(ONNX_EXPORT_PATH),
        input_names=[TRITON_INPUT_NAME],
        output_names=[TRITON_OUTPUT_NAME],
        opset_version=EXPORT_OPSET,
        do_constant_folding=True
    )
logger.info(f"✓ ONNX export complete! Size: {ONNX_EXPORT_PATH.stat().st_size / 1e6:.1f} MB (Opset {EXPORT_OPSET})")"""

assert old_c5_export_call in c5_text, "old_c5_export_call not found in Cell 5"
c5_text = c5_text.replace(old_c5_export_call, new_c5_export_call)
nb["cells"][5]["source"] = [line + "\n" for line in c5_text.split("\n")[:-1]] + ([c5_text.split("\n")[-1]] if c5_text.split("\n")[-1] else [])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated Cell 5 to suppress TracerWarnings successfully.")

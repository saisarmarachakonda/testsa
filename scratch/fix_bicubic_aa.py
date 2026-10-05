import json

nb_path = "notebook/09_convert_rfdetr_to_onnx.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c5_text = "".join(nb["cells"][5]["source"])

old_c5_export_block = """# Compatible opset for onnxruntime==1.16.1 (Opset 17 is fully and natively supported by ONNX Runtime 1.16.1)
TARGET_ORT_VERSION = "1.16.1"
EXPORT_OPSET = 17

try:
    import onnxruntime as ort
    ort_ver = getattr(ort, "__version__", TARGET_ORT_VERSION)
    logger.info(f"Verified ONNX Runtime: {ort_ver} (Target: {TARGET_ORT_VERSION}) -> Opset {EXPORT_OPSET}")
except Exception as e:
    logger.warning(f"ONNX Runtime check notice: {e}")

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

new_c5_export_block = """# =========================================================================
# ONNX Export Compatibility Patch for 'aten::_upsample_bicubic2d_aa'
# In PyTorch, DINOv2 / Vision Transformers use bicubic interpolation with anti-aliasing (aa=True).
# The PyTorch ONNX exporter lacks a native symbolic for aten::_upsample_bicubic2d_aa in Opset 17.
# We apply a dual patch:
# 1. Context manager that transparently disables antialias during ONNX graph tracing.
# 2. Custom ONNX symbolic registration mapping aten::_upsample_bicubic2d_aa to onnx::Resize (cubic mode).
# =========================================================================
from contextlib import contextmanager
from torch.onnx import register_custom_op_symbolic

def _symbolic_upsample_bicubic2d_aa(g, input, output_size, align_corners, scales_h_w=None):
    empty_scales = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
    empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
    align_corners_val = bool(align_corners) if isinstance(align_corners, (bool, int)) else False
    coord_mode = "align_corners" if align_corners_val else "half_pixel"
    return g.op(
        "Resize",
        input,
        empty_roi,
        empty_scales,
        output_size,
        coordinate_transformation_mode_s=coord_mode,
        mode_s="cubic",
        cubic_coeff_a_f=-0.75
    )

try:
    for opset_v in range(11, 21):
        register_custom_op_symbolic("::_upsample_bicubic2d_aa", _symbolic_upsample_bicubic2d_aa, opset_v)
        register_custom_op_symbolic("aten::_upsample_bicubic2d_aa", _symbolic_upsample_bicubic2d_aa, opset_v)
    logger.info("✓ Registered ONNX symbolic handler for 'aten::_upsample_bicubic2d_aa' -> onnx::Resize(mode='cubic')")
except Exception as e:
    logger.warning(f"Symbolic registration note: {e}")

@contextmanager
def disable_interpolate_antialias():
    orig_interpolate = F.interpolate
    def patched_interpolate(*args, **kwargs):
        if "antialias" in kwargs:
            kwargs["antialias"] = False
        return orig_interpolate(*args, **kwargs)
    F.interpolate = patched_interpolate
    try:
        yield
    finally:
        F.interpolate = orig_interpolate

TARGET_ORT_VERSION = "1.16.1"
EXPORT_OPSET = 17

try:
    import onnxruntime as ort
    ort_ver = getattr(ort, "__version__", TARGET_ORT_VERSION)
    logger.info(f"Verified ONNX Runtime: {ort_ver} (Target: {TARGET_ORT_VERSION}) -> Opset {EXPORT_OPSET}")
except Exception as e:
    logger.warning(f"ONNX Runtime check notice: {e}")

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

assert old_c5_export_block in c5_text, "old_c5_export_block not found in Cell 5"
c5_text = c5_text.replace(old_c5_export_block, new_c5_export_block)
nb["cells"][5]["source"] = [line + "\n" for line in c5_text.split("\n")[:-1]] + ([c5_text.split("\n")[-1]] if c5_text.split("\n")[-1] else [])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook 09 with bicubic_aa fix successfully.")

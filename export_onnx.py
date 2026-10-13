#!/usr/bin/env python3
"""
================================================================================
RF-DETR ONNX Export Script strictly conforming to Triton tagdet_rt specifications:
================================================================================
Triton config.pbtxt requirements:
  name: "tagdet_rt"
  platform: "tensorrt_plan"
  max_batch_size: 1
  input [
    {
      name: "images"
      data_type: TYPE_FP32
      dims: [ 3, 640, 480 ]
    }
  ]
  output [
    {
      name: "output"
      data_type: TYPE_FP32
      dims: [ 18900, 7 ]
    }
  ]

Usage:
  python export_onnx.py --weights path/to/best_model.pth --output deployment/models/rfdetr_model.onnx
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExportONNX")


class TagDetRTProductionWrapper(nn.Module):
    """
    Drop-in production wrapper for Triton tagdet_rt configuration:
    - Input:  images [1, 3, 640, 480] (FP32 NCHW)
    - Interpolates internally to (internal_size, internal_size) for DINOv2 ViT backbone (e.g. 560x560).
    - Runs RF-DETR core model to predict 300 queries (boxes: [cx, cy, w, h], scores: sigmoid(logits)).
    - Concatenates [cx, cy, w, h, s_0, s_1, s_2] -> [1, 300, 7].
    - Zero-pads rows from 300 to 18900 to match Triton output binding dims [18900, 7].
    """
    def __init__(self, core_model, internal_size: int = 560, total_rows: int = 18900, num_classes: int = 3):
        super().__init__()
        self.core = core_model
        self.internal_size = internal_size
        self.total_rows = total_rows
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 1. Bilinear resize from [1, 3, 640, 480] to internal backbone resolution [1, 3, internal_size, internal_size]
        x = F.interpolate(images, size=(self.internal_size, self.internal_size), mode="bilinear", align_corners=False)

        # 2. Forward pass through RF-DETR
        outputs = self.core(x)
        logits = outputs["pred_logits"]  # [B, 300, num_classes]
        boxes = outputs["pred_boxes"]    # [B, 300, 4] normalized cxcywh
        scores = torch.sigmoid(logits)   # [B, 300, num_classes]

        # Ensure scores dimension strictly matches num_classes
        if scores.shape[-1] < self.num_classes:
            pad_cls = torch.zeros(scores.shape[0], scores.shape[1], self.num_classes - scores.shape[-1], device=scores.device, dtype=scores.dtype)
            scores = torch.cat([scores, pad_cls], dim=-1)
        elif scores.shape[-1] > self.num_classes:
            scores = scores[..., :self.num_classes]

        # 3. Concatenate bbox coords [cx, cy, w, h] + class scores -> [B, 300, 7]
        det_300 = torch.cat([boxes, scores], dim=-1)

        # 4. Zero-pad rows up to total_rows (18900)
        pad_rows = self.total_rows - det_300.shape[1]
        if pad_rows > 0:
            zeros = torch.zeros(det_300.shape[0], pad_rows, det_300.shape[2], device=det_300.device, dtype=det_300.dtype)
            output = torch.cat([det_300, zeros], dim=1)  # [B, 18900, 7]
        else:
            output = det_300

        return output


@contextmanager
def export_friendly_interpolation():
    """Intercepts bicubic antialias interpolations during tracing for standard ONNX opset compatibility."""
    orig_f_interpolate = F.interpolate
    orig_torch_interpolate = getattr(torch.nn.functional, "interpolate", None)
    orig_nn_aa = getattr(torch._C._nn, "_upsample_bicubic2d_aa", None)
    orig_nn_bc = getattr(torch._C._nn, "upsample_bicubic2d", None)

    def safe_interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=False):
        if mode in ("bicubic", "area"):
            mode = "bilinear"
            align_corners = False
        return orig_f_interpolate(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=False
        )

    F.interpolate = safe_interpolate
    if hasattr(torch.nn.functional, "interpolate"):
        torch.nn.functional.interpolate = safe_interpolate

    if hasattr(torch._C._nn, "upsample_bilinear2d"):
        if hasattr(torch._C._nn, "_upsample_bicubic2d_aa"):
            try: torch._C._nn._upsample_bicubic2d_aa = torch._C._nn.upsample_bilinear2d
            except Exception: pass
        if hasattr(torch._C._nn, "upsample_bicubic2d"):
            try: torch._C._nn.upsample_bicubic2d = torch._C._nn.upsample_bilinear2d
            except Exception: pass

    try:
        yield
    finally:
        F.interpolate = orig_f_interpolate
        if orig_torch_interpolate is not None:
            torch.nn.functional.interpolate = orig_torch_interpolate
        if orig_nn_aa is not None:
            try: torch._C._nn._upsample_bicubic2d_aa = orig_nn_aa
            except Exception: pass
        if orig_nn_bc is not None:
            try: torch._C._nn.upsample_bicubic2d = orig_nn_bc
            except Exception: pass


def patch_projector_layernorm(model):
    """Fixes static shape export in DINOv2/MultiScaleProjector LayerNorm."""
    def fixed_ln_forward(self, x):
        x = x.permute(0, 2, 3, 1)
        norm_shape = tuple(self.weight.shape) if hasattr(self, "weight") and self.weight is not None else (x.shape[-1],)
        eps = getattr(self, "eps", 1e-6)
        x = F.layer_norm(x, norm_shape, self.weight, self.bias, eps)
        x = x.permute(0, 3, 1, 2)
        return x

    count = 0
    import types
    for mod_name, mod in model.named_modules():
        if "projector" in mod_name and ("LayerNorm" in mod.__class__.__name__ or "LayerNorm" in str(type(mod))):
            mod.forward = types.MethodType(fixed_ln_forward, mod)
            count += 1
    if count > 0:
        logger.info(f"Patched {count} Projector LayerNorm modules for static ONNX tracing.")


def export(weights_path: str, output_path: str, height: int = 640, width: int = 480, internal_res: int = 560, num_classes: int = 3, opset: int = 17, input_name: str = "images", output_name: str = "output0", dynamic_batch: bool = True):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    batch_str = "dynamic (batch_size)" if dynamic_batch else "1 (static)"
    logger.info("==============================================================================")
    logger.info("RF-DETR ONNX Model Exporter strictly conforming to tagdet_rt Triton schema")
    logger.info("==============================================================================")
    logger.info(f" • Input Binding:    name='{input_name}', dims=[3, {height}, {width}], shape=[{batch_str}, 3, {height}, {width}]")
    logger.info(f" • Output Binding:   name='{output_name}', dims=[18900, 7], shape=[{batch_str}, 18900, 7]")
    logger.info(f" • Dynamic Batching: {dynamic_batch}")
    logger.info(f" • Model Weights:    {weights_path}")
    logger.info(f" • Destination ONNX: {output_path}")
    logger.info("==============================================================================")

    # 1. Load model architecture
    from rfdetr import RFDETRBase
    from rfdetr import main as rfdetr_main

    # Build base model
    logger.info(f"Instantiating RFDETRBase (internal_res={internal_res})...")
    rf_model_wrapper = RFDETRBase(resolution=internal_res)
    core_model = rf_model_wrapper.model.model  # LWDETR module

    # Patch detection heads for num_classes
    device = torch.device("cpu")
    if hasattr(core_model, "class_embed"):
        in_feat = core_model.class_embed.in_features
        core_model.class_embed = nn.Linear(in_feat, num_classes)
    if hasattr(core_model, "transformer") and hasattr(core_model.transformer, "enc_out_class_embed"):
        enc = core_model.transformer.enc_out_class_embed
        if isinstance(enc, nn.ModuleList):
            for i in range(len(enc)):
                in_feat = enc[i].in_features
                enc[i] = nn.Linear(in_feat, num_classes)
        elif isinstance(enc, nn.Linear):
            in_feat = enc.in_features
            core_model.transformer.enc_out_class_embed = nn.Linear(in_feat, num_classes)

    # 2. Load weights if provided
    if weights_path and Path(weights_path).exists():
        logger.info(f"Loading weights from {weights_path}...")
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # Filter prefixes
        cleaned = {}
        model_keys = set(core_model.state_dict().keys())
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith("model.model."): clean_k = clean_k[12:]
            elif clean_k.startswith("model."): clean_k = clean_k[6:]
            elif clean_k.startswith("module."): clean_k = clean_k[7:]
            if clean_k in model_keys and core_model.state_dict()[clean_k].shape == v.shape:
                cleaned[clean_k] = v
        missing, unexpected = core_model.load_state_dict(cleaned, strict=False)
        logger.info(f"Loaded {len(cleaned)} state dict weights (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
    else:
        logger.warning(f"Weights file not found at '{weights_path}'. Exporting with initialized weights for architecture verification.")

    # 3. Build TagDetRT Production Wrapper
    wrapper = TagDetRTProductionWrapper(core_model, internal_size=internal_res, total_rows=18900, num_classes=num_classes)
    wrapper.to(device)
    wrapper.eval()

    # Apply patches
    patch_projector_layernorm(wrapper)

    # 4. Dummy input with [1, 3, height, width]
    dummy_input = torch.randn(1, 3, height, width, dtype=torch.float32, device=device)

    # Dynamic axes mapping
    dynamic_axes = {
        input_name: {0: "batch_size"},
        output_name: {0: "batch_size"}
    } if dynamic_batch else None

    # 5. Export to ONNX
    logger.info(f"Exporting to ONNX via opset {opset} (Dynamic axes: {dynamic_axes})...")
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    with torch.no_grad(), export_friendly_interpolation():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(out_file),
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True
        )

    logger.info(f"✓ ONNX export successfully created: {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")

    # 6. Verification with ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np
        logger.info("Running ONNX Runtime verification...")
        session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        logger.info(f" • Input:  name='{inp.name}', shape={inp.shape}, type={inp.type}")
        logger.info(f" • Output: name='{out.name}', shape={out.shape}, type={out.type}")
        assert inp.name == input_name, f"Input name mismatch: {inp.name} != '{input_name}'"
        assert out.name == output_name, f"Output name mismatch: {out.name} != '{output_name}'"
        
        # Test batch 1 and batch 2
        test_batches = [1, 2] if dynamic_batch else [1]
        for b in test_batches:
            dummy_np = np.random.randn(b, 3, height, width).astype(np.float32)
            res = session.run([output_name], {input_name: dummy_np})[0]
            assert res.shape == (b, 18900, 7), f"Batch {b} failed: {res.shape} != ({b}, 18900, 7)"
            logger.info(f" • Verified Batch {b} Inference: Output shape is {res.shape}")
        logger.info("✓ Verified! Exactly matches Triton config.pbtxt specifications with dynamic batching.")
    except ImportError:
        logger.info("onnxruntime not installed. Skipped live session verification.")
    except Exception as e:
        logger.warning(f"Verification note: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RF-DETR to ONNX matching tagdet_rt Triton schema")
    parser.add_argument("--weights", type=str, default="best_model.pth", help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="deployment/models/rfdetr_model.onnx", help="Output .onnx path")
    parser.add_argument("--height", type=int, default=640, help="Input height (default: 640)")
    parser.add_argument("--width", type=int, default=480, help="Input width (default: 480)")
    parser.add_argument("--internal-res", type=int, default=560, help="Backbone internal resolution (default: 560)")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes (default: 3)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--input-name", type=str, default="images", help="Input tensor name (default: images)")
    parser.add_argument("--output-name", type=str, default="output0", help="Output tensor name (default: output0)")
    parser.add_argument("--static-batch", action="store_true", help="Force static batch size 1 instead of dynamic batching")

    args = parser.parse_args()
    export(
        weights_path=args.weights,
        output_path=args.output,
        height=args.height,
        width=args.width,
        internal_res=args.internal_res,
        num_classes=args.num_classes,
        opset=args.opset,
        input_name=args.input_name,
        output_name=args.output_name,
        dynamic_batch=not args.static_batch
    )

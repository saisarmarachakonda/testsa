#!/usr/bin/env python3
"""
install_until_plan_loads.py
===========================
Automated script that starts testing from TensorRT 8.4 onwards,
inspects the target .plan engine, and iteratively installs matching
TensorRT packages via pip until the engine is successfully loaded on GPU.

Usage:
    python install_until_plan_loads.py
    python install_until_plan_loads.py --plan location_tag_text_dat/1/model.plan
"""

import os
import sys
import re
import struct
import subprocess
import argparse

CANDIDATE_PATHS = [
    "location_tag_text_dat/1/model.plan",
    "location_tag_text_det/1/model.plan",
    "../location_tag_text_dat/1/model.plan",
    "../location_tag_text_det/1/model.plan",
    "location_tag_text_dat/model.plan",
    "location_tag_text_det/model.plan",
    "1/model.plan",
    "model.plan",
    "../model.plan",
    "location_tag_text_det.engine",
    "location_tag_text_det.plan",
]

# Prioritized list starting from TensorRT 8.4 onwards
DEFAULT_VERSION_CANDIDATES = [
    # 1. Start from 8.4 series
    "8.4.3.1",
    "8.4.2.4",
    "8.4.1.5",
    "8.4.0.6",
    # 2. TensorRT 8.5 series
    "8.5.3.1",
    "8.5.2.2",
    "8.5.1.7",
    # 3. TensorRT 8.6 series
    "8.6.1",
    "8.6.1.post1",
    "8.6.0",
    # 4. Older 8.x fallback
    "8.2.5.1",
    "8.2.4.2",
    "8.0.3.4",
    # 5. TensorRT 10.x series
    "10.0.1",
    "10.1.0",
    "10.2.0",
    "10.3.0",
    "10.4.0",
    "10.5.0",
    "10.6.0",
    "10.7.0",
    "10.8.0.43",
]


def find_plan_file(specified_path=None):
    if specified_path and os.path.exists(specified_path):
        return specified_path
    for p in CANDIDATE_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            return p
    for root_dir in [".", ".."]:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "node_modules", ".git")]
                for fn in files:
                    if fn.endswith(".plan") or fn.endswith(".engine"):
                        candidate = os.path.join(root, fn)
                        if os.path.getsize(candidate) > 1024:
                            return candidate
    return None


def extract_versions_from_plan(plan_path):
    """Parses binary header of .plan file to find embedded version strings."""
    candidates = []
    if not os.path.exists(plan_path):
        return candidates
    try:
        with open(plan_path, "rb") as f:
            chunk = f.read(65536)
        matches = re.findall(rb"(\d{1,2}\.\d{1,2}\.\d{1,2}(?:\.\d{1,4})?)", chunk)
        for m in matches:
            s = m.decode("ascii", errors="ignore")
            if s.startswith(("8.", "9.", "10.", "11.")) and s not in candidates:
                candidates.append(s)
        for offset in range(0, min(len(chunk) - 16, 512)):
            if chunk[offset:offset+4].startswith(b"ptrt") or chunk[offset:offset+3] == b"TRT":
                for sub_off in [4, 8, 12, 16, 20]:
                    if offset + sub_off + 16 <= len(chunk):
                        nums = struct.unpack("<IIII", chunk[offset+sub_off:offset+sub_off+16])
                        if nums[0] in (8, 9, 10, 11) and nums[1] < 20 and nums[2] < 20:
                            v_str = f"{nums[0]}.{nums[1]}.{nums[2]}"
                            if nums[3] > 0 and nums[3] < 100:
                                v_str += f".{nums[3]}"
                            if v_str not in candidates:
                                candidates.append(v_str)
    except Exception as e:
        print(f"Notice while inspecting engine header: {e}")
    return candidates


def test_engine_in_subprocess(plan_path):
    """Runs deserialization in an isolated Python process to prevent shared library caching."""
    script = f"""
import sys, os
try:
    import tensorrt as trt
except ImportError as e:
    print("NO_TRT|" + str(e))
    sys.exit(1)

plan_path = {repr(plan_path)}
if not os.path.exists(plan_path):
    print("FILE_MISSING")
    sys.exit(2)

errors = []
class LogCollector(trt.ILogger):
    def __init__(self):
        super().__init__()
    def log(self, severity, msg):
        errors.append(msg)

logger = LogCollector()
try:
    with open(plan_path, "rb") as f:
        data = f.read()
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(data)
    if engine is not None:
        print(f"LOAD_SUCCESS|{{trt.__version__}}")
        sys.exit(0)
    else:
        err_msg = " ; ".join(errors)
        print(f"LOAD_FAILED|{{trt.__version__}}|{{err_msg}}")
        sys.exit(3)
except Exception as e:
    err_msg = str(e) + " ; " + " ; ".join(errors)
    print(f"EXCEPTION|{{getattr(trt, '__version__', 'unknown')}}|{{err_msg}}")
    sys.exit(4)
"""
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def install_package(pkg_spec):
    """Installs a specific tensorrt package using pip with NVIDIA index."""
    cmd = [
        sys.executable, "-m", "pip", "install", "-U",
        "--extra-index-url", "https://pypi.nvidia.com",
        pkg_spec, "cuda-python"
    ]
    print(f"\n[PIP] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cmd2 = [sys.executable, "-m", "pip", "install", "-U", pkg_spec, "cuda-python"]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        return result2.returncode == 0
    return True


def run():
    parser = argparse.ArgumentParser(description="Install TensorRT versions (starting from 8.4) until .plan loads.")
    parser.add_argument("--plan", type=str, default=None, help="Path to model.plan file")
    args = parser.parse_args()

    print("=" * 80)
    print("   TENSORRT .PLAN ENGINE PACKAGE RESOLVER (Starting from TRT 8.4)")
    print("=" * 80)

    # 1. Locate Plan File
    plan_path = find_plan_file(args.plan)
    if not plan_path:
        print("❌ Error: Could not locate a .plan or .engine file.")
        print("Please provide the path via: python install_until_plan_loads.py --plan <path>")
        sys.exit(1)

    plan_size_mb = os.path.getsize(plan_path) / 1e6
    print(f"Target Plan Engine: {plan_path} ({plan_size_mb:.1f} MB)")

    # 2. Inspect Header for Hints
    header_hints = extract_versions_from_plan(plan_path)
    print(f"Header Version Hints: {header_hints if header_hints else 'None found'}")

    # 3. Test current environment first
    print("\n--- Step 1: Testing Currently Installed TensorRT ---")
    ret, out, err = test_engine_in_subprocess(plan_path)
    
    if ret == 0 and "LOAD_SUCCESS" in out:
        cur_ver = out.split("|")[1]
        print(f"🎉 SUCCESS! Current TensorRT version ({cur_ver}) can ALREADY load the engine!")
        print("No packages need to be installed. Your environment is ready.")
        sys.exit(0)

    # Parse error message for version requirement
    extracted_version = None
    gpu_arch_mismatch = False

    if "LOAD_FAILED" in out or "EXCEPTION" in out:
        parts = out.split("|", 2)
        cur_ver = parts[1] if len(parts) > 1 else "unknown"
        error_log = parts[2] if len(parts) > 2 else err
        print(f"Current TensorRT ({cur_ver}) failed to load engine.")
        print(f"Diagnostic Log: {error_log}")

        m_exp = re.search(r"expecting library version\s*([0-9\.]+)", error_log, re.IGNORECASE)
        if m_exp:
            extracted_version = m_exp.group(1).strip()
            print(f"🎯 Exact TensorRT version requested by engine: {extracted_version}")

        if "compute capability" in error_log.lower() or "sm_" in error_log.lower():
            gpu_arch_mismatch = True
            print("⚠️ Notice: A GPU architecture / compute capability mismatch was detected.")

    # 4. Build prioritized list of candidate package specs STARTING FROM 8.4
    test_queue = []
    
    # If the engine explicitly named a version in its error message, put it first
    if extracted_version:
        test_queue.append(f"tensorrt=={extracted_version}")
        parts = extracted_version.split(".")
        if len(parts) >= 3:
            test_queue.append(f"tensorrt~={parts[0]}.{parts[1]}.{parts[2]}")

    # If header had hints matching 8.4+, prioritize those
    for hint in header_hints:
        pkg = f"tensorrt=={hint}"
        if pkg not in test_queue:
            test_queue.append(pkg)

    # Queue the standard candidate list (starts from 8.4 series!)
    for v in DEFAULT_VERSION_CANDIDATES:
        pkg = f"tensorrt=={v}"
        if pkg not in test_queue:
            test_queue.append(pkg)

    print(f"\n--- Step 2: Testing Candidate Versions ({len(test_queue)} to test, starting at 8.4) ---")

    for idx, pkg_spec in enumerate(test_queue, 1):
        print(f"\n[{idx}/{len(test_queue)}] Testing candidate: {pkg_spec} ...")
        installed = install_package(pkg_spec)
        if not installed:
            print(f"   Package {pkg_spec} could not be downloaded/installed from PyPI. Skipping...")
            continue

        ret, out, err = test_engine_in_subprocess(plan_path)
        if ret == 0 and "LOAD_SUCCESS" in out:
            success_ver = out.split("|")[1]
            print("\n" + "=" * 80)
            print(f"🎉 SUCCESS! Successfully resolved TensorRT version!")
            print(f"   - Installed Package: {pkg_spec}")
            print(f"   - Active TRT Version: {success_ver}")
            print(f"   - Loaded Engine:     {plan_path}")
            print("=" * 80)
            print("\nYou can now run 'predict_tags_5_images.ipynb' cleanly!")
            sys.exit(0)
        else:
            msg = out.split("|")[-1] if "|" in out else (out or err)
            print(f"   ❌ Engine failed with {pkg_spec}: {msg[:140]}")

    print("\n" + "=" * 80)
    print("❌ All candidate TensorRT versions were attempted.")
    if gpu_arch_mismatch:
        print("Note: The failure is due to a GPU Architecture / Compute Capability mismatch.")
        print("      The .plan file was compiled on a different NVIDIA GPU model.")
        print("      Solution: Use the ONNX model fallback via ONNX Runtime in the notebook.")
    else:
        print("Recommendation: Use ONNX Runtime with .onnx model as configured in predict_tags_5_images.ipynb.")
    print("=" * 80)


if __name__ == "__main__":
    run()

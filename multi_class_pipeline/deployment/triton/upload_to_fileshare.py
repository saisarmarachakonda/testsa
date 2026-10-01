#!/usr/bin/env python3
"""
Step 3: Upload model.plan to Triton File Share & Verify is_model_ready()

Usage:
    python upload_to_fileshare.py --plan-file model.plan --model-version 2 --model-name rfdetr
"""

import os
import sys
import time
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Upload model.plan to Triton FileShare & verify readiness.")
    parser.add_argument("--plan-file", default="model.plan", help="Local model.plan file path")
    parser.add_argument("--config-file", default="config.pbtxt", help="Path to config.pbtxt")
    parser.add_argument("--model-name", default=os.getenv("RFDETR_MODEL_NAME", "tagdet_rt"), help="Triton model name (default: tagdet_rt)")
    parser.add_argument("--model-version", default=os.getenv("RFDETR_MODEL_VERSION", "2"), help="New model version number (e.g. 2)")
    parser.add_argument("--storage-account", default=os.getenv("AZURE_STORAGE_ACCOUNT", "<YOUR_STORAGE_ACCOUNT>"))
    parser.add_argument("--file-share", default=os.getenv("TRITON_FILE_SHARE", "aks-iras-t4-floor-models"), help="Azure File Share name (default: aks-iras-t4-floor-models)")
    parser.add_argument("--triton-url", default=os.getenv("TRITON_URL", "localhost:8001"), help="Triton gRPC endpoint")
    parser.add_argument("--skip-upload", action="store_true", help="Skip Azure upload, only verify Triton readiness")
    return parser.parse_args()

def upload_via_azure_cli(plan_path: Path, config_path: Path, account: str, share: str, model_name: str, version: str):
    import subprocess
    print("=" * 70)
    print(f" Uploading model version '{version}' to Azure File Share '{share}'")
    print("=" * 70)

    # 1. Ensure remote directories exist
    dest_dir = f"models/{model_name}/{version}"
    print(f"Creating remote directory structure: {dest_dir}...")
    subprocess.run([
        "az", "storage", "directory", "create",
        "--account-name", account,
        "--share-name", share,
        "--name", f"models/{model_name}"
    ], check=False)
    subprocess.run([
        "az", "storage", "directory", "create",
        "--account-name", account,
        "--share-name", share,
        "--name", dest_dir
    ], check=False)

    # 2. Upload config.pbtxt
    if config_path.exists():
        print(f"Uploading config: {config_path} -> models/{model_name}/config.pbtxt")
        subprocess.run([
            "az", "storage", "file", "upload",
            "--account-name", account,
            "--share-name", share,
            "--source", str(config_path),
            "--path", f"models/{model_name}/config.pbtxt"
        ], check=True)

    # 3. Upload model.plan
    print(f"Uploading model engine: {plan_path} -> {dest_dir}/model.plan")
    subprocess.run([
        "az", "storage", "file", "upload",
        "--account-name", account,
        "--share-name", share,
        "--source", str(plan_path),
        "--path", f"{dest_dir}/model.plan"
    ], check=True)

    print("✓ Upload completed successfully.")

def verify_triton_readiness(triton_url: str, model_name: str, version: str, timeout_sec: int = 120):
    print("=" * 70)
    print(f" Verifying is_model_ready('{model_name}', version={version}) on {triton_url}...")
    print("=" * 70)

    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print("Notice: 'tritonclient[all]' not installed locally. Verifying via HTTP REST fallback...")
        verify_http_readiness(triton_url, model_name, version, timeout_sec)
        return

    client = grpcclient.InferenceServerClient(url=triton_url)
    start_time = time.time()

    while time.time() - start_time < timeout_sec:
        try:
            is_server_live = client.is_server_live()
            is_ready = client.is_model_ready(model_name=model_name, model_version=str(version))
            if is_ready:
                print(f"\n=======================================================")
                print(f"✓ MODEL IS READY IN TRITON!")
                print(f"   - Model Name:    {model_name}")
                print(f"   - Version:       {version}")
                print(f"   - Triton Server: {triton_url}")
                print(f"=======================================================")
                return
        except Exception as e:
            pass

        print(".", end="", flush=True)
        time.sleep(5)

    print(f"\nTimed out waiting for {model_name} version {version} to be ready on {triton_url}.")
    print("Check Triton pod logs: kubectl logs -l app=triton-server -n <namespace>")

def verify_http_readiness(triton_url: str, model_name: str, version: str, timeout_sec: int = 120):
    import requests
    # If triton_url is 8001 (gRPC), HTTP is typically 8000
    host = triton_url.split(":")[0]
    http_url = f"http://{host}:8000/v2/models/{model_name}/versions/{version}/ready"
    start_time = time.time()

    while time.time() - start_time < timeout_sec:
        try:
            resp = requests.get(http_url, timeout=3)
            if resp.status_code == 200:
                print(f"\n✓ Triton HTTP confirms model {model_name} v{version} is READY!")
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(5)

    print(f"\nTimed out waiting for HTTP response from {http_url}")

def main():
    args = parse_args()
    plan_path = Path(args.plan_file)
    config_path = Path(args.config_file)

    if not args.skip_upload:
        if not plan_path.exists():
            print(f"ERROR: Model plan file '{plan_path}' does not exist.")
            sys.exit(1)
        upload_via_azure_cli(plan_path, config_path, args.storage_account, args.file_share, args.model_name, args.model_version)

    verify_triton_readiness(args.triton_url, args.model_name, args.model_version)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
deploy.py: Cross-platform deployment script for RF-DETR on AKS (Internal LoadBalancer - No Public IP).
Automates:
  1. Resource Group & ACR creation
  2. ACR Cloud Build (az acr build)
  3. AKS cluster provisioning with --attach-acr
  4. Kubernetes manifest deployment
  5. Internal VNet IP polling and port-forward instructions
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path

RESOURCE_GROUP = os.getenv("RESOURCE_GROUP", "rg-rfdetr-inference")
LOCATION = os.getenv("LOCATION", "eastus")
AKS_CLUSTER_NAME = os.getenv("AKS_CLUSTER_NAME", "aks-rfdetr-cluster")
IMAGE_NAME = os.getenv("IMAGE_NAME", "rfdetr-inference")
IMAGE_TAG = os.getenv("IMAGE_TAG", "v1")
NODE_COUNT = int(os.getenv("NODE_COUNT", "2"))
NODE_VM_SIZE = os.getenv("NODE_VM_SIZE", "Standard_D4s_v5")

BASE_DIR = Path(__file__).resolve().parent
SERVER_DIR = BASE_DIR / "server"
K8S_DIR = BASE_DIR / "k8s"
MODELS_DIR = BASE_DIR / "models"

def run_cmd(cmd_list, check=True):
    print(f"-> {' '.join(cmd_list)}")
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 and check:
        print(f"Error:\n{proc.stdout}")
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")
    return proc.returncode, proc.stdout.strip()

def main():
    print("==============================================================================")
    print(" RF-DETR AKS Deployment (Private Internal LoadBalancer - No Public IP)")
    print("==============================================================================")
    print(f" - Resource Group: {RESOURCE_GROUP}")
    print(f" - Location:       {LOCATION}")
    print(f" - Cluster:        {AKS_CLUSTER_NAME} ({NODE_VM_SIZE}, {NODE_COUNT} nodes)")
    print(f" - Server Dir:     {SERVER_DIR}")
    print("==============================================================================")

    # 1. Model Check
    model_dest = SERVER_DIR / "rfdetr_model.onnx"
    if not model_dest.exists():
        src_candidates = [
            MODELS_DIR / "rfdetr_model.onnx",
            BASE_DIR.parent / "deployment_artifacts" / "rfdetr_model.onnx"
        ]
        found = False
        for cand in src_candidates:
            if cand.exists():
                print(f"Copying {cand} -> {model_dest}")
                shutil.copy(cand, model_dest)
                found = True
                break
        if not found:
            print(f"ERROR: Model file not found at {model_dest}.")
            print("Please run 'convert_rfdetr_to_onnx.ipynb' first.")
            sys.exit(1)

    # 2. Check Azure CLI
    ret, out = run_cmd(["az", "account", "show"], check=False)
    if ret != 0:
        print("Not logged in to Azure CLI. Executing 'az login'...")
        run_cmd(["az", "login"])
    _, sub_name = run_cmd(["az", "account", "show", "--query", "name", "-o", "tsv"])
    print(f"✓ Azure Subscription: {sub_name}")

    # 3. Create Resource Group
    print(f"Ensuring Resource Group '{RESOURCE_GROUP}'...")
    run_cmd(["az", "group", "create", "--name", RESOURCE_GROUP, "--location", LOCATION])

    # 4. Create ACR
    acr_suffix = str(int(time.time()))[-6:]
    acr_name = os.getenv("ACR_NAME", f"acrrfdetr{acr_suffix}")
    print(f"Ensuring ACR '{acr_name}'...")
    run_cmd(["az", "acr", "create", "--resource-group", RESOURCE_GROUP, "--name", acr_name, "--sku", "Standard", "--location", LOCATION], check=False)
    _, login_server = run_cmd(["az", "acr", "show", "--name", acr_name, "--query", "loginServer", "-o", "tsv"])
    print(f"✓ ACR Login Server: {login_server}")

    # 5. Build Image in ACR
    full_image = f"{login_server}/{IMAGE_NAME}:{IMAGE_TAG}"
    print(f"Submitting cloud build to ACR: {full_image}...")
    run_cmd(["az", "acr", "build", "--registry", acr_name, "--image", f"{IMAGE_NAME}:{IMAGE_TAG}", str(SERVER_DIR)])

    # 6. Provision AKS Cluster
    print(f"Ensuring AKS Cluster '{AKS_CLUSTER_NAME}'...")
    ret, _ = run_cmd(["az", "aks", "show", "--resource-group", RESOURCE_GROUP, "--name", AKS_CLUSTER_NAME], check=False)
    if ret != 0:
        run_cmd([
            "az", "aks", "create",
            "--resource-group", RESOURCE_GROUP,
            "--name", AKS_CLUSTER_NAME,
            "--node-count", str(NODE_COUNT),
            "--node-vm-size", NODE_VM_SIZE,
            "--attach-acr", acr_name,
            "--enable-managed-identity",
            "--generate-ssh-keys"
        ])
    else:
        run_cmd(["az", "aks", "update", "--resource-group", RESOURCE_GROUP, "--name", AKS_CLUSTER_NAME, "--attach-acr", acr_name], check=False)

    # 7. Configure kubectl
    print("Configuring kubectl context...")
    run_cmd(["az", "aks", "get-credentials", "--resource-group", RESOURCE_GROUP, "--name", AKS_CLUSTER_NAME, "--overwrite-existing"])

    # 8. Render and Apply Manifests
    with open(K8S_DIR / "deployment.yaml") as f:
        tmpl = f.read()
    rendered = tmpl.replace("{{ACR_LOGIN_SERVER}}", login_server).replace("{{IMAGE_TAG}}", IMAGE_TAG)
    rendered_file = K8S_DIR / "deployment_rendered.yaml"
    with open(rendered_file, "w") as f:
        f.write(rendered)

    print("Applying Kubernetes manifests...")
    run_cmd(["kubectl", "apply", "-f", str(rendered_file)])
    run_cmd(["kubectl", "apply", "-f", str(K8S_DIR / "service.yaml")])
    run_cmd(["kubectl", "apply", "-f", str(K8S_DIR / "hpa.yaml")])

    # 9. Wait for Rollout
    print("Waiting for deployment rollout...")
    run_cmd(["kubectl", "rollout", "status", "deployment/rfdetr-inference-deployment", "--timeout=300s"])

    # 10. Check Private IP
    print("Querying Internal LoadBalancer Private VNet IP...")
    private_ip = None
    for attempt in range(24):
        ret, ip_out = run_cmd(["kubectl", "get", "service", "rfdetr-inference-service", "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"], check=False)
        if ret == 0 and ip_out and ip_out != "<pending>":
            private_ip = ip_out
            break
        time.sleep(5)

    print("\n==============================================================================")
    print(" DEPLOYMENT COMPLETE (NO PUBLIC IP)")
    print("==============================================================================")
    if private_ip:
        print(f" Private VNet IP:  http://{private_ip}")
        print(" Public Exposure:  NONE (Internal LoadBalancer only)")
    else:
        print(" Private VNet IP:  Pending allocation in VNet subnet")
    print("\nTo securely access and test the service from your workstation without VPN:")
    print("Run:")
    print("   ./port_forward.sh")
    print("Then query: http://127.0.0.1:8000/docs or run: python test_client.py")
    print("==============================================================================")

if __name__ == "__main__":
    main()

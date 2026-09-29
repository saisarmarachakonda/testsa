#!/usr/bin/env bash
# ==============================================================================
# port_forward.sh: Secure TLS port-forwarding tunnel for private AKS microservice
# Binds local port 8000 to internal Kubernetes Service port 80 (TargetPort 8000)
# ==============================================================================

set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-rfdetr-inference-service}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-80}"

echo "=============================================================================="
echo " Establishing Secure Port-Forward Tunnel to ${SERVICE_NAME}"
echo "=============================================================================="
echo " - Local Endpoint:   http://127.0.0.1:${LOCAL_PORT}"
echo " - Remote Service:   ${SERVICE_NAME}:${REMOTE_PORT}"
echo " - Public IP:        NONE (Traffic is encrypted over kubectl control plane)"
echo "=============================================================================="

# Terminate existing stale port-forward on same port if running
if pgrep -f "kubectl port-forward service/${SERVICE_NAME}" &>/dev/null; then
    echo "Terminating existing port-forward process..."
    pkill -f "kubectl port-forward service/${SERVICE_NAME}" || true
    sleep 1
fi

echo "Starting port-forwarding in foreground (Press Ctrl+C to terminate)..."
echo "Endpoints available once connected:"
echo " - Health Probe: http://127.0.0.1:${LOCAL_PORT}/healthz"
echo " - Swagger Docs: http://127.0.0.1:${LOCAL_PORT}/docs"
echo " - Prediction:   http://127.0.0.1:${LOCAL_PORT}/predict"
echo "------------------------------------------------------------------------------"

exec kubectl port-forward "service/${SERVICE_NAME}" "${LOCAL_PORT}:${REMOTE_PORT}"

#!/usr/bin/env bash
# cloud-train.sh — Deploy and manage Pylos AlphaZero training on GCP
#
# Usage:
#   cloud-train.sh [--dry-run] <command> [options]
#
# Commands:
#   deploy    Create GCE VM with GPU and deploy training container
#   start     Start training on the remote VM
#   stop      Stop training on the remote VM
#   status    Check VM and training status
#   teardown  Delete VM and clean up resources (use scripts/teardown.sh for full cleanup)
#
# Default mode is --dry-run (prints plan without executing).

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────

GCP_PROJECT="${GCP_PROJECT:-atomic-segment-456903-e6}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-pylos-train}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-50GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-ssd}"
IMAGE_NAME="${IMAGE_NAME:-pylos-train}"
GCR_IMAGE="gcr.io/${GCP_PROJECT}/${IMAGE_NAME}"
CONFIG_FILE="${CONFIG_FILE:-engine/config_v3.yaml}"
METRICS_PORT="${METRICS_PORT:-8080}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ───────────────────────────────────────────────────────

DRY_RUN=true
COMMAND=""
FORCE=false

# ── Cost estimates ─────────────────────────────────────────────────

declare -A COST_PER_HOUR=(
    ["n1-standard-4+nvidia-tesla-t4"]="0.35"
    ["n1-standard-8+nvidia-tesla-t4"]="0.48"
    ["n1-standard-4+nvidia-tesla-v100"]="2.48"
    ["n1-standard-8+nvidia-tesla-v100"]="2.61"
    ["n1-standard-4+nvidia-tesla-p100"]="1.46"
)

# ── Helpers ────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
dry()   { echo -e "${YELLOW}[DRY-RUN]${NC} Would execute: $*"; }

get_cost_estimate() {
    local key="${MACHINE_TYPE}+${GPU_TYPE}"
    local hourly="${COST_PER_HOUR[$key]:-unknown}"
    if [[ "$hourly" == "unknown" ]]; then
        echo "Cost estimate unavailable for ${key}"
    else
        local daily
        daily=$(echo "$hourly * 24" | bc)
        echo "\$${hourly}/hr (\$${daily}/day)"
    fi
}

check_prerequisites() {
    local missing=()
    command -v gcloud >/dev/null 2>&1 || missing+=("gcloud")
    command -v docker >/dev/null 2>&1 || missing+=("docker")

    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing[*]}"
        echo "  Install gcloud: https://cloud.google.com/sdk/docs/install"
        echo "  Install docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    # Verify gcloud is authenticated and project is set
    local current_project
    current_project=$(gcloud config get-value project 2>/dev/null || true)
    if [[ "$current_project" != "$GCP_PROJECT" ]]; then
        warn "Current gcloud project is '${current_project}', expected '${GCP_PROJECT}'"
        warn "Run: gcloud config set project ${GCP_PROJECT}"
    fi
}

confirm() {
    local prompt="$1"
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi
    echo -e "${YELLOW}${prompt}${NC}"
    read -r -p "Continue? [y/N] " response
    [[ "$response" =~ ^[Yy]$ ]]
}

# ── Commands ───────────────────────────────────────────────────────

cmd_deploy() {
    info "Deploy Plan:"
    echo ""
    echo "  GCP Project:   ${GCP_PROJECT}"
    echo "  Zone:          ${GCP_ZONE}"
    echo "  VM Name:       ${VM_NAME}"
    echo "  Machine Type:  ${MACHINE_TYPE}"
    echo "  GPU:           ${GPU_COUNT}x ${GPU_TYPE}"
    echo "  Boot Disk:     ${BOOT_DISK_SIZE} ${BOOT_DISK_TYPE}"
    echo "  Docker Image:  ${GCR_IMAGE}"
    echo "  Config:        ${CONFIG_FILE}"
    echo "  Metrics Port:  ${METRICS_PORT}"
    echo ""
    echo "  Estimated Cost: $(get_cost_estimate)"
    echo ""
    echo "  Mayor notification: REQUIRED before resource creation"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        dry "docker build -t ${GCR_IMAGE} ${PROJECT_DIR}"
        dry "docker push ${GCR_IMAGE}"
        dry "gcloud compute instances create ${VM_NAME} \\"
        echo "      --project=${GCP_PROJECT} \\"
        echo "      --zone=${GCP_ZONE} \\"
        echo "      --machine-type=${MACHINE_TYPE} \\"
        echo "      --accelerator=type=${GPU_TYPE},count=${GPU_COUNT} \\"
        echo "      --boot-disk-size=${BOOT_DISK_SIZE} \\"
        echo "      --boot-disk-type=${BOOT_DISK_TYPE} \\"
        echo "      --image-family=ubuntu-2204-lts \\"
        echo "      --image-project=ubuntu-os-cloud \\"
        echo "      --maintenance-policy=TERMINATE \\"
        echo "      --metadata=startup-script='#!/bin/bash"
        echo "        # Install NVIDIA drivers and Docker"
        echo "        curl -fsSL https://get.docker.com | sh"
        echo "        distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        echo "        curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | sed \"s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g\" | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        echo "        apt-get update && apt-get install -y nvidia-driver-535 nvidia-container-toolkit"
        echo "        systemctl restart docker"
        echo "        # Pull and run training container"
        echo "        gcloud auth configure-docker --quiet"
        echo "        docker pull ${GCR_IMAGE}"
        echo "      '"
        echo ""
        info "No resources created (dry-run mode)"
        info "To execute: cloud-train.sh --live deploy"
        return 0
    fi

    check_prerequisites

    # Notify mayor before creating resources
    info "Notifying mayor about resource creation..."
    if command -v gt >/dev/null 2>&1; then
        gt mail send mayor/ -s "GPU VM creation: ${VM_NAME}" \
            -m "Polecat creating GCE VM: ${MACHINE_TYPE} + ${GPU_TYPE} in ${GCP_ZONE}. Est. cost: $(get_cost_estimate)" \
            2>/dev/null || warn "Could not notify mayor (gt not available)"
    else
        warn "gt CLI not available - skipping mayor notification"
    fi

    if ! confirm "This will create a billable GCE VM. Cost: $(get_cost_estimate)"; then
        info "Aborted."
        return 1
    fi

    # Step 1: Build Docker image
    info "Building Docker image..."
    if [[ ! -f "${PROJECT_DIR}/Dockerfile" ]]; then
        error "No Dockerfile found in ${PROJECT_DIR}"
        error "Create one first (see py-wss bead) or use a pre-built image"
        exit 1
    fi
    docker build -t "${GCR_IMAGE}" "${PROJECT_DIR}"

    # Step 2: Push to GCR
    info "Pushing to Google Container Registry..."
    gcloud auth configure-docker --quiet
    docker push "${GCR_IMAGE}"

    # Step 3: Create VM with GPU
    info "Creating VM ${VM_NAME}..."
    gcloud compute instances create "${VM_NAME}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
        --boot-disk-size="${BOOT_DISK_SIZE}" \
        --boot-disk-type="${BOOT_DISK_TYPE}" \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --maintenance-policy=TERMINATE \
        --scopes=storage-ro \
        --metadata=startup-script='#!/bin/bash
set -e
# Install Docker
curl -fsSL https://get.docker.com | sh
# Install NVIDIA drivers
apt-get update && apt-get install -y ubuntu-drivers-common
ubuntu-drivers autoinstall
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
    | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker
# Auth and pull image
gcloud auth configure-docker --quiet
docker pull '"${GCR_IMAGE}"'
echo "SETUP_COMPLETE" > /tmp/vm-setup-status
'

    ok "VM ${VM_NAME} created. Waiting for startup script to complete..."
    info "Check status with: cloud-train.sh status"
    info "Set up tunnel with: scripts/setup-tunnel.sh"
}

cmd_start() {
    info "Starting training on ${VM_NAME}..."

    if [[ "$DRY_RUN" == "true" ]]; then
        dry "gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} --command='docker run -d --gpus all --name pylos-training -p ${METRICS_PORT}:${METRICS_PORT} ${GCR_IMAGE} python -m engine.train --config ${CONFIG_FILE}'"
        return 0
    fi

    check_prerequisites

    # Check if VM exists and is running
    local vm_status
    vm_status=$(gcloud compute instances describe "${VM_NAME}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

    if [[ "$vm_status" == "NOT_FOUND" ]]; then
        error "VM ${VM_NAME} not found. Run 'cloud-train.sh deploy' first."
        exit 1
    fi

    if [[ "$vm_status" == "TERMINATED" ]]; then
        info "VM is stopped. Starting it first..."
        gcloud compute instances start "${VM_NAME}" \
            --project="${GCP_PROJECT}" \
            --zone="${GCP_ZONE}"
        info "Waiting for VM to boot..."
        sleep 30
    fi

    info "Launching training container..."
    gcloud compute ssh "${VM_NAME}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --command="docker run -d \
            --gpus all \
            --name pylos-training \
            --restart=unless-stopped \
            -p ${METRICS_PORT}:${METRICS_PORT} \
            -v /home/\$(whoami)/checkpoints:/app/engine/checkpoints_v3 \
            ${GCR_IMAGE} \
            python -m engine.train --config ${CONFIG_FILE}"

    ok "Training started on ${VM_NAME}"
    info "View logs: cloud-train.sh status --logs"
    info "Set up tunnel: scripts/setup-tunnel.sh"
}

cmd_stop() {
    info "Stopping training on ${VM_NAME}..."

    if [[ "$DRY_RUN" == "true" ]]; then
        dry "gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} --command='docker stop pylos-training'"
        return 0
    fi

    check_prerequisites

    gcloud compute ssh "${VM_NAME}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --command="docker stop pylos-training 2>/dev/null || echo 'Container not running'"

    ok "Training stopped"
    info "VM is still running. To save costs:"
    info "  Stop VM:  gcloud compute instances stop ${VM_NAME} --zone=${GCP_ZONE}"
    info "  Teardown: scripts/teardown.sh"
}

cmd_status() {
    local show_logs=false
    for arg in "$@"; do
        [[ "$arg" == "--logs" ]] && show_logs=true
    done

    info "Checking status of ${VM_NAME}..."

    if [[ "$DRY_RUN" == "true" ]]; then
        dry "gcloud compute instances describe ${VM_NAME} --zone=${GCP_ZONE}"
        dry "gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} --command='docker ps; nvidia-smi'"
        return 0
    fi

    check_prerequisites

    # VM status
    local vm_status
    vm_status=$(gcloud compute instances describe "${VM_NAME}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

    echo ""
    echo "  VM Status:  ${vm_status}"

    if [[ "$vm_status" == "NOT_FOUND" ]]; then
        warn "VM ${VM_NAME} does not exist."
        return 0
    fi

    if [[ "$vm_status" != "RUNNING" ]]; then
        warn "VM is not running (status: ${vm_status})"
        return 0
    fi

    # VM creation time for cost estimate
    local created
    created=$(gcloud compute instances describe "${VM_NAME}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --format="value(creationTimestamp)" 2>/dev/null || echo "unknown")
    echo "  Created:    ${created}"
    echo "  Zone:       ${GCP_ZONE}"
    echo ""

    # Container and GPU status
    info "Checking container and GPU..."
    gcloud compute ssh "${VM_NAME}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --command="
echo '── Docker Containers ──'
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || echo 'Docker not available'
echo ''
echo '── GPU Status ──'
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo 'GPU not available'
echo ''
echo '── Training Progress ──'
docker exec pylos-training cat /app/engine/checkpoints_v3/training_progress.json 2>/dev/null || echo 'No training progress available'
" 2>/dev/null || warn "Could not connect to VM"

    if [[ "$show_logs" == "true" ]]; then
        echo ""
        info "Training logs (last 50 lines):"
        gcloud compute ssh "${VM_NAME}" \
            --project="${GCP_PROJECT}" \
            --zone="${GCP_ZONE}" \
            --command="docker logs --tail 50 pylos-training 2>&1" \
            2>/dev/null || warn "Could not fetch logs"
    fi
}

cmd_teardown() {
    info "Teardown plan for ${VM_NAME}:"
    echo ""
    echo "  1. Stop training container"
    echo "  2. Delete VM instance (${VM_NAME})"
    echo "  3. Delete GCR image (${GCR_IMAGE})"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        dry "gcloud compute instances delete ${VM_NAME} --zone=${GCP_ZONE}"
        dry "gcloud container images delete ${GCR_IMAGE}"
        info "No resources deleted (dry-run mode)"
        info "For full teardown: scripts/teardown.sh --live"
        return 0
    fi

    # Delegate to the dedicated teardown script
    exec "${SCRIPT_DIR}/teardown.sh" "$@"
}

# ── Argument Parsing ───────────────────────────────────────────────

usage() {
    echo "Usage: cloud-train.sh [options] <command>"
    echo ""
    echo "Commands:"
    echo "  deploy    Create GCE VM with GPU and deploy training container"
    echo "  start     Start training on the remote VM"
    echo "  stop      Stop training on the remote VM"
    echo "  status    Check VM and training status (--logs for container logs)"
    echo "  teardown  Delete VM and clean up resources"
    echo ""
    echo "Options:"
    echo "  --dry-run   Show what would happen without executing (default)"
    echo "  --live      Actually execute commands (disables dry-run)"
    echo "  --force     Skip confirmation prompts"
    echo ""
    echo "Environment Variables:"
    echo "  GCP_PROJECT    GCP project ID (default: atomic-segment-456903-e6)"
    echo "  GCP_ZONE       Compute zone (default: us-central1-a)"
    echo "  VM_NAME        VM instance name (default: pylos-train)"
    echo "  MACHINE_TYPE   Machine type (default: n1-standard-4)"
    echo "  GPU_TYPE       GPU type (default: nvidia-tesla-t4)"
    echo "  GPU_COUNT      Number of GPUs (default: 1)"
    echo "  CONFIG_FILE    Training config (default: engine/config_v3.yaml)"
    echo "  METRICS_PORT   Metrics/dashboard port (default: 8080)"
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --live)     DRY_RUN=false; shift ;;
        --force)    FORCE=true; shift ;;
        -h|--help)  usage; exit 0 ;;
        deploy|start|stop|status|teardown)
            COMMAND="$1"; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$COMMAND" ]]; then
    usage
    exit 1
fi

case "$COMMAND" in
    deploy)   cmd_deploy ;;
    start)    cmd_start ;;
    stop)     cmd_stop ;;
    status)   cmd_status "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" ;;
    teardown) cmd_teardown ;;
    *)        error "Unknown command: ${COMMAND}"; usage; exit 1 ;;
esac

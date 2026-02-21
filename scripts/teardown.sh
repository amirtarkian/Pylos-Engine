#!/usr/bin/env bash
# teardown.sh — Full teardown of Pylos cloud training resources
#
# Deletes the GCE VM, GCR Docker images, and cleans up local artifacts.
# Each destructive action requires confirmation unless --force is used.
#
# Usage:
#   teardown.sh [--dry-run] [--force] [--keep-local]

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────

GCP_PROJECT="${GCP_PROJECT:-atomic-segment-456903-e6}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-pylos-train}"
IMAGE_NAME="${IMAGE_NAME:-pylos-train}"
GCR_IMAGE="gcr.io/${GCP_PROJECT}/${IMAGE_NAME}"

DRY_RUN=false
FORCE=false
KEEP_LOCAL=false

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

confirm() {
    local prompt="$1"
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi
    echo -e "${YELLOW}${prompt}${NC}"
    read -r -p "Continue? [y/N] " response
    [[ "$response" =~ ^[Yy]$ ]]
}

# ── Teardown Steps ─────────────────────────────────────────────────

teardown_tunnel() {
    info "Step 1: Kill SSH tunnel (if running)..."
    local pid_file="/tmp/pylos-tunnel-${VM_NAME}.pid"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            if [[ "$DRY_RUN" == "true" ]]; then
                dry "kill ${pid}"
            else
                kill "$pid" 2>/dev/null || true
                ok "Tunnel killed (PID: ${pid})"
            fi
        fi
        [[ "$DRY_RUN" != "true" ]] && rm -f "$pid_file"
    else
        info "  No tunnel PID file found"
    fi
}

teardown_vm() {
    info "Step 2: Delete GCE VM instance..."

    local vm_status
    vm_status=$(gcloud compute instances describe "${VM_NAME}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

    if [[ "$vm_status" == "NOT_FOUND" ]]; then
        info "  VM ${VM_NAME} does not exist (already deleted or never created)"
        return 0
    fi

    echo "  VM Status: ${vm_status}"

    if [[ "$DRY_RUN" == "true" ]]; then
        dry "gcloud compute instances delete ${VM_NAME} --zone=${GCP_ZONE} --quiet"
        return 0
    fi

    if ! confirm "Delete VM '${VM_NAME}' in ${GCP_ZONE}? This is irreversible."; then
        warn "  Skipped VM deletion"
        return 0
    fi

    gcloud compute instances delete "${VM_NAME}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --quiet

    ok "VM ${VM_NAME} deleted"
}

teardown_gcr() {
    info "Step 3: Delete GCR Docker image..."

    # Check if any images exist
    local images
    images=$(gcloud container images list-tags "${GCR_IMAGE}" \
        --project="${GCP_PROJECT}" \
        --format="value(digest)" 2>/dev/null || true)

    if [[ -z "$images" ]]; then
        info "  No images found at ${GCR_IMAGE}"
        return 0
    fi

    local count
    count=$(echo "$images" | wc -l | tr -d ' ')
    echo "  Found ${count} image(s) at ${GCR_IMAGE}"

    if [[ "$DRY_RUN" == "true" ]]; then
        dry "gcloud container images delete ${GCR_IMAGE} --force-delete-tags --quiet"
        return 0
    fi

    if ! confirm "Delete all images at ${GCR_IMAGE}?"; then
        warn "  Skipped GCR cleanup"
        return 0
    fi

    # Delete all digests
    for digest in $images; do
        gcloud container images delete "${GCR_IMAGE}@${digest}" \
            --project="${GCP_PROJECT}" \
            --force-delete-tags \
            --quiet
    done

    ok "GCR images deleted"
}

teardown_local() {
    if [[ "$KEEP_LOCAL" == "true" ]]; then
        info "Step 4: Skipped local cleanup (--keep-local)"
        return 0
    fi

    info "Step 4: Clean up local Docker images..."

    local local_images
    local_images=$(docker images "${GCR_IMAGE}" --format "{{.ID}}" 2>/dev/null || true)

    if [[ -z "$local_images" ]]; then
        info "  No local images for ${GCR_IMAGE}"
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        dry "docker rmi ${GCR_IMAGE}"
        return 0
    fi

    if ! confirm "Remove local Docker image ${GCR_IMAGE}?"; then
        warn "  Skipped local cleanup"
        return 0
    fi

    docker rmi "${GCR_IMAGE}" 2>/dev/null || true
    ok "Local images cleaned up"
}

verify() {
    info "Verification..."
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        dry "gcloud compute instances list --project=${GCP_PROJECT}"
        return 0
    fi

    # Check no VM remains
    local remaining
    remaining=$(gcloud compute instances list \
        --project="${GCP_PROJECT}" \
        --filter="name=${VM_NAME}" \
        --format="value(name)" 2>/dev/null || true)

    if [[ -z "$remaining" ]]; then
        ok "No VM named '${VM_NAME}' found"
    else
        warn "VM '${VM_NAME}' still exists!"
    fi

    # Check no GCR images remain
    local gcr_remaining
    gcr_remaining=$(gcloud container images list-tags "${GCR_IMAGE}" \
        --project="${GCP_PROJECT}" \
        --format="value(digest)" 2>/dev/null || true)

    if [[ -z "$gcr_remaining" ]]; then
        ok "No GCR images at ${GCR_IMAGE}"
    else
        warn "GCR images still exist at ${GCR_IMAGE}"
    fi

    echo ""
    ok "Teardown complete"
}

# ── Argument Parsing ───────────────────────────────────────────────

usage() {
    echo "Usage: teardown.sh [options]"
    echo ""
    echo "Options:"
    echo "  --dry-run       Show what would be deleted (default: false)"
    echo "  --force         Skip confirmation prompts"
    echo "  --keep-local    Don't remove local Docker images"
    echo "  -h, --help      Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  GCP_PROJECT    GCP project (default: atomic-segment-456903-e6)"
    echo "  GCP_ZONE       Zone (default: us-central1-a)"
    echo "  VM_NAME        VM name (default: pylos-train)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --live)         DRY_RUN=false; shift ;;
        --force)        FORCE=true; shift ;;
        --keep-local)   KEEP_LOCAL=true; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ── Main ───────────────────────────────────────────────────────────

echo ""
info "Pylos Cloud Training Teardown"
echo "  Project: ${GCP_PROJECT}"
echo "  VM:      ${VM_NAME}"
echo "  Zone:    ${GCP_ZONE}"
echo "  Image:   ${GCR_IMAGE}"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "  Mode:    DRY-RUN (no resources will be deleted)"
fi
echo ""

teardown_tunnel
teardown_vm
teardown_gcr
teardown_local
verify

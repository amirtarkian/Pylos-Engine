#!/usr/bin/env bash
# setup-tunnel.sh — Set up SSH tunnel for Pylos training metrics
#
# Creates an SSH tunnel to forward the remote training metrics port to localhost,
# so you can view the training dashboard locally.
#
# Usage:
#   setup-tunnel.sh [options]
#
# Options:
#   --local-port PORT    Local port to bind (default: 8080)
#   --remote-port PORT   Remote port on VM (default: 8080)
#   --reverse            Set up reverse tunnel (local -> remote) for pushing metrics
#   --kill               Kill any existing tunnel for this VM
#   --status             Show tunnel status

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────

GCP_PROJECT="${GCP_PROJECT:-atomic-segment-456903-e6}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-pylos-train}"
LOCAL_PORT="${LOCAL_PORT:-8080}"
REMOTE_PORT="${REMOTE_PORT:-8080}"
REVERSE=false
PID_FILE="/tmp/pylos-tunnel-${VM_NAME}.pid"

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

check_prerequisites() {
    if ! command -v gcloud >/dev/null 2>&1; then
        error "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
}

get_tunnel_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
        rm -f "$PID_FILE"
    fi
    return 1
}

# ── Commands ───────────────────────────────────────────────────────

cmd_start() {
    check_prerequisites

    # Check if tunnel already running
    if pid=$(get_tunnel_pid); then
        warn "Tunnel already running (PID: ${pid})"
        info "Kill it first: setup-tunnel.sh --kill"
        return 1
    fi

    # Verify VM is running
    local vm_status
    vm_status=$(gcloud compute instances describe "${VM_NAME}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

    if [[ "$vm_status" != "RUNNING" ]]; then
        error "VM ${VM_NAME} is not running (status: ${vm_status})"
        exit 1
    fi

    if [[ "$REVERSE" == "true" ]]; then
        # Reverse tunnel: remote can reach localhost:LOCAL_PORT
        info "Setting up reverse tunnel: VM:${REMOTE_PORT} -> localhost:${LOCAL_PORT}"
        gcloud compute ssh "${VM_NAME}" \
            --project="${GCP_PROJECT}" \
            --zone="${GCP_ZONE}" \
            -- -R "${REMOTE_PORT}:localhost:${LOCAL_PORT}" \
               -N -f -o "ExitOnForwardFailure=yes" \
               -o "ServerAliveInterval=60" \
               -o "ServerAliveCountMax=3"
    else
        # Forward tunnel: localhost:LOCAL_PORT -> remote:REMOTE_PORT
        info "Setting up forward tunnel: localhost:${LOCAL_PORT} -> VM:${REMOTE_PORT}"
        gcloud compute ssh "${VM_NAME}" \
            --project="${GCP_PROJECT}" \
            --zone="${GCP_ZONE}" \
            -- -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
               -N -f -o "ExitOnForwardFailure=yes" \
               -o "ServerAliveInterval=60" \
               -o "ServerAliveCountMax=3"
    fi

    # Find the SSH process PID
    local tunnel_pid
    tunnel_pid=$(pgrep -f "ssh.*${VM_NAME}.*-[LR].*${LOCAL_PORT}" | tail -1 || true)

    if [[ -n "$tunnel_pid" ]]; then
        echo "$tunnel_pid" > "$PID_FILE"
        ok "Tunnel established (PID: ${tunnel_pid})"
        if [[ "$REVERSE" == "true" ]]; then
            info "Remote VM can now reach localhost:${LOCAL_PORT} via port ${REMOTE_PORT}"
        else
            info "Dashboard available at: http://localhost:${LOCAL_PORT}"
        fi
        info "Kill with: setup-tunnel.sh --kill"
    else
        warn "Tunnel may have started but PID not found. Check with: setup-tunnel.sh --status"
    fi
}

cmd_kill() {
    if pid=$(get_tunnel_pid); then
        kill "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
        ok "Tunnel killed (PID: ${pid})"
    else
        info "No tunnel running for ${VM_NAME}"
        # Also try to find and kill any orphaned tunnels
        local orphans
        orphans=$(pgrep -f "ssh.*${VM_NAME}.*-[LR]" 2>/dev/null || true)
        if [[ -n "$orphans" ]]; then
            warn "Found orphaned SSH tunnel processes: ${orphans}"
            echo "  Kill manually: kill ${orphans}"
        fi
    fi
}

cmd_status() {
    if pid=$(get_tunnel_pid); then
        ok "Tunnel is running (PID: ${pid})"
        echo "  VM: ${VM_NAME}"
        echo "  Local port: ${LOCAL_PORT}"
        echo "  Remote port: ${REMOTE_PORT}"
        echo "  PID file: ${PID_FILE}"
    else
        info "No tunnel running for ${VM_NAME}"
    fi
}

# ── Argument Parsing ───────────────────────────────────────────────

usage() {
    echo "Usage: setup-tunnel.sh [options]"
    echo ""
    echo "Options:"
    echo "  --local-port PORT    Local port (default: 8080)"
    echo "  --remote-port PORT   Remote port on VM (default: 8080)"
    echo "  --reverse            Reverse tunnel (local -> remote)"
    echo "  --kill               Kill existing tunnel"
    echo "  --status             Show tunnel status"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  GCP_PROJECT    GCP project (default: atomic-segment-456903-e6)"
    echo "  GCP_ZONE       Zone (default: us-central1-a)"
    echo "  VM_NAME        VM name (default: pylos-train)"
}

ACTION="start"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-port)   LOCAL_PORT="$2"; shift 2 ;;
        --remote-port)  REMOTE_PORT="$2"; shift 2 ;;
        --reverse)      REVERSE=true; shift ;;
        --kill)         ACTION="kill"; shift ;;
        --status)       ACTION="status"; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

case "$ACTION" in
    start)  cmd_start ;;
    kill)   cmd_kill ;;
    status) cmd_status ;;
esac

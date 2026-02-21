# Cloud Training Guide

Deploy Pylos AlphaZero training to a GCP VM with GPU acceleration.

## Prerequisites

1. **Google Cloud SDK** (`gcloud` CLI)
   ```bash
   # macOS
   brew install google-cloud-sdk

   # Or: https://cloud.google.com/sdk/docs/install
   ```

2. **Docker** (for building/pushing the training image)
   ```bash
   # macOS
   brew install docker
   ```

3. **GCP Project Access**
   ```bash
   gcloud auth login
   gcloud config set project atomic-segment-456903-e6
   ```

4. **Enable Required APIs**
   ```bash
   gcloud services enable compute.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

5. **GPU Quota** — Request GPU quota in your target zone if not already available:
   - Go to: IAM & Admin > Quotas
   - Filter: `NVIDIA T4 GPUs` in `us-central1`
   - Request increase to at least 1

## Cost Reference

| Configuration | GPU | $/hr | $/day | Notes |
|---|---|---:|---:|---|
| n1-standard-4 + T4 | NVIDIA T4 (16GB) | $0.35 | $8.40 | Cheapest GPU option |
| n1-standard-8 + T4 | NVIDIA T4 (16GB) | $0.48 | $11.52 | More CPU/RAM |
| n1-standard-4 + P100 | NVIDIA P100 (16GB) | $1.46 | $35.04 | Better FP32 |
| n1-standard-4 + V100 | NVIDIA V100 (16GB) | $2.48 | $59.52 | Best performance |
| Boot disk (50GB SSD) | — | — | $0.28 | ~$8.50/month |
| Network egress | — | — | — | ~$0.12/GB (minimal) |

**Recommendation**: Start with **n1-standard-4 + T4** (~$0.35/hr). It provides 16GB GPU
memory, which is sufficient for the Pylos model. Upgrade to V100 only if training is
too slow.

## Quick Start

### 1. Preview the deployment plan (dry-run)

```bash
bash scripts/cloud-train.sh --dry-run deploy
```

This shows what would be created without executing anything.

### 2. Deploy to GCP

```bash
bash scripts/cloud-train.sh --live deploy
```

This will:
- Build the Docker image locally
- Push it to Google Container Registry
- Create a GCE VM with GPU
- Install NVIDIA drivers and Docker on the VM
- Pull the training image

### 3. Start training

```bash
bash scripts/cloud-train.sh --live start
```

### 4. Set up the metrics tunnel

```bash
bash scripts/setup-tunnel.sh
```

This forwards the remote metrics port to `localhost:8080` so you can view the
training dashboard locally.

### 5. Monitor training

```bash
# Check VM and training status
bash scripts/cloud-train.sh --live status

# Include container logs
bash scripts/cloud-train.sh --live status --logs
```

### 6. Teardown when done

```bash
# Preview what will be deleted
bash scripts/teardown.sh --dry-run

# Execute teardown
bash scripts/teardown.sh
```

## Script Reference

### cloud-train.sh

Main orchestration script. **Default mode is `--dry-run`** (safe).

```bash
# Commands
cloud-train.sh [--dry-run|--live] deploy     # Create VM + deploy image
cloud-train.sh [--dry-run|--live] start      # Start training container
cloud-train.sh [--dry-run|--live] stop       # Stop training container
cloud-train.sh [--dry-run|--live] status     # Check VM/training status
cloud-train.sh [--dry-run|--live] teardown   # Delete all resources

# Options
--dry-run    Show plan without executing (default)
--live       Execute commands for real
--force      Skip confirmation prompts
```

### setup-tunnel.sh

SSH tunnel management for metrics/dashboard access.

```bash
setup-tunnel.sh                    # Create forward tunnel (remote -> local)
setup-tunnel.sh --reverse          # Create reverse tunnel (local -> remote)
setup-tunnel.sh --local-port 9090  # Use different local port
setup-tunnel.sh --status           # Check if tunnel is running
setup-tunnel.sh --kill             # Kill existing tunnel
```

### teardown.sh

Full resource cleanup with per-step confirmation.

```bash
teardown.sh              # Interactive teardown (confirms each step)
teardown.sh --dry-run    # Preview what would be deleted
teardown.sh --force      # Skip confirmations
teardown.sh --keep-local # Don't remove local Docker images
```

## Configuration

All scripts read configuration from environment variables:

| Variable | Default | Description |
|---|---|---|
| `GCP_PROJECT` | `atomic-segment-456903-e6` | GCP project ID |
| `GCP_ZONE` | `us-central1-a` | Compute zone |
| `VM_NAME` | `pylos-train` | VM instance name |
| `MACHINE_TYPE` | `n1-standard-4` | GCE machine type |
| `GPU_TYPE` | `nvidia-tesla-t4` | GPU accelerator type |
| `GPU_COUNT` | `1` | Number of GPUs |
| `CONFIG_FILE` | `engine/config_v3.yaml` | Training config path |
| `METRICS_PORT` | `8080` | Port for metrics/dashboard |

Example with custom config:

```bash
MACHINE_TYPE=n1-standard-8 GPU_TYPE=nvidia-tesla-v100 \
  bash scripts/cloud-train.sh --live deploy
```

## Training Configuration

The training config (`engine/config_v3.yaml`) controls the AlphaZero training loop.
Key settings for cloud training:

```yaml
training:
  selfplay_games: 1000000    # Total games to play
  search_iterations: 64      # MCTS depth per move
  batch_size: 128            # Training batch size
  selfplay_batch_size: 0     # 0=CPU workers, >0=GPU batched MCTS

checkpoints:
  save_every: 1000           # Checkpoint frequency
  eval_games: 50             # Games per checkpoint evaluation
  dir: engine/checkpoints_v3
```

For GPU training, consider setting `selfplay_batch_size` to a value like 32 or 64
to use GPU-batched MCTS instead of CPU workers.

## Troubleshooting

### VM won't start / GPU not available

```bash
# Check GPU quota
gcloud compute regions describe us-central1 \
  --format="table(quotas.filter(metric=NVIDIA_T4_GPUS))"

# Try a different zone
GCP_ZONE=us-central1-b bash scripts/cloud-train.sh --live deploy
```

### Can't SSH to VM

```bash
# Ensure your SSH key is registered
gcloud compute config-ssh

# Try with explicit project
gcloud compute ssh pylos-train --project=atomic-segment-456903-e6 --zone=us-central1-a
```

### Docker image push fails

```bash
# Configure Docker for GCR auth
gcloud auth configure-docker

# Verify you have push access
gcloud projects get-iam-policy atomic-segment-456903-e6 \
  --flatten="bindings[].members" --filter="bindings.role:roles/storage.admin"
```

### Training container crashes

```bash
# Check container logs
bash scripts/cloud-train.sh --live status --logs

# SSH in and check manually
gcloud compute ssh pylos-train --zone=us-central1-a
docker logs pylos-training
nvidia-smi  # Verify GPU is visible
```

### Tunnel won't connect

```bash
# Kill any stale tunnels
bash scripts/setup-tunnel.sh --kill

# Check VM is running
bash scripts/cloud-train.sh --live status

# Try a different local port
bash scripts/setup-tunnel.sh --local-port 9090
```

## Manual Operations

If the scripts fail, these are the manual equivalents:

```bash
# Create VM
gcloud compute instances create pylos-train \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --boot-disk-size=50GB \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE

# Delete VM
gcloud compute instances delete pylos-train --zone=us-central1-a

# Delete GCR image
gcloud container images delete gcr.io/atomic-segment-456903-e6/pylos-train

# Verify cleanup
gcloud compute instances list
```

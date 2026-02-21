# Pylos AlphaZero Training — GPU Container
# Base: CUDA 12.1 runtime + Python 3.11
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy engine source
COPY engine/ engine/

# Checkpoint volume mount point
VOLUME /app/checkpoints

ENTRYPOINT ["python", "engine/train.py", "--config", "engine/config_cloud.yaml"]

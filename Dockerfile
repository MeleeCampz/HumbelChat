# ──────────────────────────────────────────────────────────────
# HumbelChat Discord bot — production image (see docs/docker.md)
#
# Build:  docker compose up -d --build     (from the repo root)
# The image contains only code + deps. All runtime state (KB, recordings,
# history, logs, model cache) lives in bind-mounted host directories.
# ──────────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

# libopus0 — discord.py voice loads it via ctypes at runtime (DAVE E2EE).
# curl — reachability probes for INFER_URL during setup.
# git — Obsidian vault sync (clone/pull/commit/push of KB vaults, #19).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopus0 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first: keeps the image small (skips multi-GB CUDA wheels) and
# satisfies sentence-transformers' torch dependency. After building, verify
# `docker compose run --rm bot python -c "import torch; print(torch.__version__)"`
# shows a +cpu build. If pip resolves a conflicting version from the default
# index instead, drop this line and accept the default torch.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# NOTE: intentionally runs as root. The bind-mounted host dirs (./data, ./logs,
# ./hf-cache) are owned by uid 0 inside the container on Docker Desktop, so a
# non-root user could not write logs/history/vector index. Single-host, trusted
# image — accepted trade-off for a local deployment.

CMD ["python", "-u", "main.py"]

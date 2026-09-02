FROM python:3.12-slim

# System deps: libopus for Discord voice transport
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopus0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast, cache-friendly installs
COPY --from=ghcr.io/astral-sh/uv:0.7.21 /uv /uvx /usr/local/bin/

# Pin deps and install (cache-friendly layer)
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENTRYPOINT ["python", "-u", "main.py"]

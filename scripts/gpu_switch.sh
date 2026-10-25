#!/usr/bin/env bash
# gpu_switch.sh — swap GPU-1 (RTX 5070 Ti) between RAG serving and image generation.
#
# Background (2026-10-25, see HANDOFF_RERANKER_LLAMA.md "GPU accounting"):
#   The reranker + embedder llama.cpp servers pin ~6.4 GB of GPU-1 VRAM
#   (reranker 2,742 MiB, embedder 3,612 MiB — measured from the servers' own
#   startup "memory breakdown" logs; nvidia-smi numbers are unreliable under
#   WDDM retention and must not be used for this decision). The chat model
#   (Unsloth Studio) keeps a ~4-6 GB tensor-split share on the same card.
#   Stopping both RAG servers frees ~6.4 GB for an image model; the bot
#   degrades gracefully while they are down:
#     - retrieval falls back to dense + BM25 (RRF) ranking — the measured
#       pre-reranker baseline, nothing crashes
#     - the rerank attach floor no-ops (gate scores absent)
#     - the embedder is only needed when INDEXING new chunks; plain search
#       reads precomputed vectors and never touches it
#
# Usage (from the repo root):
#   ./scripts/gpu_switch.sh status    # show service states + what is resident
#   ./scripts/gpu_switch.sh image     # stop reranker + embedder (image-gen mode)
#   ./scripts/gpu_switch.sh rag       # start both and wait until healthy
set -euo pipefail

cd "$(dirname "$0")/.."

RERANK_PORT=8090
EMBED_PORT=8091

wait_healthy() {
  local name=$1 port=$2 tries=${3:-60}
  for i in $(seq 1 "$tries"); do
    if curl -sf --max-time 3 "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; then
      echo "  ${name}: healthy after ~$((i * 2))s"
      return 0
    fi
    sleep 2
  done
  echo "  ${name}: NOT healthy after $((tries * 2))s — check: docker logs humbelchat-${name}-llama" >&2
  return 1
}

state_of() {
  docker inspect --format '{{if .State.Running}}running{{else}}stopped{{end}}' "humbelchat-$1-llama" 2>/dev/null || echo "missing"
}

case "${1:-status}" in
  status)
    echo "GPU-1 RAG servers:"
    echo "  reranker-llama: $(state_of reranker)"
    echo "  embedder-llama:  $(state_of embedder)"
    echo
    echo "Resident VRAM on GPU 1 (measured, not nvidia-smi):"
    echo "  chat model (Studio, tensor-split 50,16): ~4-6 GB while loaded"
    [[ "$(state_of reranker)" == running ]] && echo "  reranker: 2,742 MiB"
    [[ "$(state_of embedder)" == running ]] && echo "  embedder: 3,612 MiB"
    echo
    echo "nvidia-smi (WDDM-retained, informational only):"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed 's/^/  /'
    ;;
  image)
    echo "Stopping RAG servers (bot degrades to dense+BM25; embedder only needed for reindexing)..."
    docker compose stop reranker-llama embedder-llama
    echo "Done. ~6.4 GB freed on GPU 1 for an image model."
    echo "Restore with: ./scripts/gpu_switch.sh rag"
    ;;
  rag)
    echo "Starting RAG servers..."
    docker compose up -d reranker-llama embedder-llama
    wait_healthy reranker "$RERANK_PORT"
    wait_healthy embedder "$EMBED_PORT"
    echo "RAG stack back to full quality."
    ;;
  *)
    echo "Usage: $0 {status|image|rag}" >&2
    exit 1
    ;;
esac

#!/usr/bin/env bash
# gpu_switch.sh — switch the 4070 Ti SUPER (GPU 0) between RAG serving and
# image generation (GPU-split 2026-11).
#
# Topology:
#   GPU 0 (4070 Ti SUPER): ONE mode at a time —
#     "rag"   reranker-llama + embedder-llama (default steady state)
#     "image" those stopped + the Studio diffusion model loaded (~12.5 GiB
#             measured at Q4_K_M; loads via Studio's management API)
#   GPU 1 (5070 Ti): chat (Strata) + tts-ollama — NEVER touched here.
#
# The bot switches modes automatically around generations via the gpu-switch
# helper service (compose service "gpu-switch", scripts/gpu_switch_helper.py).
# This script is the MANUAL twin: it prefers the helper (single source of
# truth for the mode) and falls back to raw docker compose commands when the
# helper is unreachable (helper down / running before compose was updated).
#
# While in image mode the bot degrades gracefully:
#   - retrieval falls back to dense + BM25 (RRF) ranking — the measured
#     pre-reranker baseline, nothing crashes
#   - the rerank attach floor no-ops (gate scores absent)
#   - the embedder is only needed when INDEXING new chunks
#
# Usage (from the repo root):
#   ./scripts/gpu_switch.sh status    # helper state (+ docker states)
#   ./scripts/gpu_switch.sh image     # -> image mode (stop RAG, load diffusion)
#   ./scripts/gpu_switch.sh rag       # -> rag mode (unload diffusion, start RAG)
set -euo pipefail

cd "$(dirname "$0")/.."

# The helper's host-visible port (compose maps 8080 -> HOST_GPU_SWITCH_PORT).
HELPER="http://127.0.0.1:${HOST_GPU_SWITCH_PORT:-8082}"
RERANK_PORT=8090
EMBED_PORT=8091

helper_post() {  # $1 = rag|image
  # returns: 0 = ok, 1 = helper unreachable, 2 = helper REFUSED the switch
  local res
  res=$(curl -sf --max-time 330 -X POST -H 'Content-Type: application/json' \
        -d "{\"mode\":\"$1\"}" "$HELPER/mode" 2>/dev/null) || return 1
  echo "  helper: $res"
  if printf '%s' "$res" | grep -q '"refused'; then
    return 2
  fi
  return 0
}

helper_get() {
  curl -sf --max-time 5 "$HELPER/mode" 2>/dev/null || true
}

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
    echo "GPU mode (helper): ${HELPER}"
    s=$(helper_get)
    if [[ -n "$s" ]]; then
      echo "  $s"
    else
      echo "  helper unreachable — falling back to docker states"
    fi
    echo "GPU-0 RAG servers (docker):"
    echo "  reranker-llama: $(state_of reranker)"
    echo "  embedder-llama:  $(state_of embedder)"
    echo
    echo "Measured working sets (nvidia-smi under WDDM is informational only):"
    echo "  reranker 2,742 MiB / embedder 3,612 MiB (startup breakdown logs)"
    echo "  image model ~12.5 GiB (Studio resident_mib)"
    echo "  chat model on GPU 1 (Strata) — not managed by this script"
    echo
    echo "nvidia-smi (informational only):"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | sed 's/^/  /'
    ;;
  image)
    rc=0
    helper_post image || rc=$?
    if [[ $rc == 0 ]]; then
      echo "Done: 4070 in IMAGE mode. Restore with: ./scripts/gpu_switch.sh rag"
    elif [[ $rc == 2 ]]; then
      # The helper refused deliberately (image backend unreachable — it
      # refuses to tear RAG down). Do NOT raw-fallback: that is exactly the
      # half-switched state the refusal exists to prevent.
      echo "REFUSED: image backend unreachable — the 4070 stays in rag mode." >&2
      echo "Fix Studio, then retry: ./scripts/gpu_switch.sh image" >&2
      exit 2
    else
      echo "helper unreachable — raw fallback: stopping RAG servers only" >&2
      echo "  (the diffusion model will NOT be loaded — trigger a generation" >&2
      echo "   or load it manually in Studio)" >&2
      docker compose stop reranker-llama embedder-llama
      echo "Done: 4070 RAG servers stopped. Restore with: ./scripts/gpu_switch.sh rag"
    fi
    ;;
  rag)
    if helper_post rag; then :; else
      echo "helper unreachable — raw fallback: starting RAG servers" >&2
      docker compose up -d reranker-llama embedder-llama
    fi
    wait_healthy reranker "$RERANK_PORT"
    wait_healthy embedder "$EMBED_PORT"
    echo "RAG stack back to full quality."
    ;;
  *)
    echo "Usage: $0 {status|image|rag}" >&2
    exit 1
    ;;
esac

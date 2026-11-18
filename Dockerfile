# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────
# HumbelChat Discord bot — production image (see docs/docker.md)
#
# Build:  docker compose up -d --build     (from the repo root)
# The image contains only code + deps. All runtime state (KB, history,
# logs, model cache) lives in bind-mounted host directories.
#
# Build speed: pip downloads (notably the ~900MB CPU torch wheel) are cached
# on the daemon via BuildKit cache mounts, so changing requirements.txt never
# re-downloads wheels. Requires BuildKit (default on Docker Desktop /
# docker 23+). The .dockerignore keeps the uploaded build context small —
# keep it in sync when adding big dirs to the repo.
# ──────────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

# curl — reachability probes for INFER_URL during setup.
# git — Obsidian vault sync (clone/pull/commit/push of KB vaults, #19).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first: keeps the image small (skips multi-GB CUDA wheels) and
# satisfies sentence-transformers' torch dependency. After building, verify
# `docker compose run --rm bot python -c "import torch; print(torch.__version__)"`
# shows a +cpu build. If pip resolves a conflicting version from the default
# index instead, drop this line and accept the default torch.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch

WORKDIR /app

# Build stamp: which commit this image was built from. Derived AT BUILD TIME from
# the repo's own .git metadata (see the RUN right after COPY . .), so a plain
# `docker compose up -d --build` stamps itself — no --build-arg, no env var, no
# chance of a stale value being typed by hand. The bot logs it at startup and
# deploys verify the running container against the repo tip — a stale cached
# build once served pre-fix code while looking healthy (#68 follow-up, 2026-10-28).
# Override only for contexts that have no .git at all (CI from a tarball, a
# copied source tree): docker compose build --build-arg GIT_SHA=<full sha>
# (/app/.build_sha is the single source of truth — nothing reads an env var).
ARG GIT_SHA=

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .
# Resolve HEAD -> commit sha inside the build. The context mount carries only
# what .dockerignore allows (HEAD, packed-refs, refs/**), which is enough:
# loose ref file first, packed-refs as the fallback (git gc packs refs), a bare
# sha in HEAD covers detached-HEAD builds. Anything unresolvable stays "unknown"
# — a visible unknown beats a confidently wrong sha. The whole-context mount
# (instead of source=.git) is deliberate: a context with no .git at all must
# still build.
RUN --mount=type=bind,source=.,target=/tmp/ctx \
    set -eu; \
    sha="${GIT_SHA}"; \
    gm=/tmp/ctx/.git; \
    if [ -z "$sha" ] && [ -f "$gm/HEAD" ]; then \
      head_content=$(tr -d ' \t\r\n' < "$gm/HEAD"); \
      case "$head_content" in \
        ref:*) ref="${head_content#ref:}"; \
               if [ -f "$gm/$ref" ]; then \
                 sha=$(tr -d ' \t\r\n' < "$gm/$ref"); \
               elif [ -f "$gm/packed-refs" ]; then \
                 sha=$(awk -v r="$ref" '$2 == r && length($1) == 40 { print $1; exit }' "$gm/packed-refs"); \
               fi; \
               ;; \
        *) sha="$head_content" ;; \
      esac; \
    fi; \
    case "$sha" in *[!0-9a-f]*) sha=unknown;; esac; \
    [ "${#sha}" -ge 7 ] || sha=unknown; \
    printf '%s' "$sha" > /app/.build_sha; \
    rm -rf /app/.git; \
    echo "build stamp -> $sha"

# NOTE: intentionally runs as root. The bind-mounted host dirs (./data, ./logs,
# ./hf-cache) are owned by uid 0 inside the container on Docker Desktop, so a
# non-root user could not write logs/history/vector index. Single-host, trusted
# image — accepted trade-off for a local deployment.

CMD ["python", "-u", "main.py"]

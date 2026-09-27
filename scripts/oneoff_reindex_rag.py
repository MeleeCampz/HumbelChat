"""One-off: force-rebuild the vector index with the new plain-text chunking.

Run while the bot container is STOPPED (sole-writer rule on vector_index.db):
    docker compose exec -T bot python - < scripts/oneoff_reindex_rag.py
"""
from __future__ import annotations

import asyncio
import sys
import time

sys.path.insert(0, "/app")

from kb.index import KBIndexStore  # noqa: E402


async def main():
    store = KBIndexStore("/app/data/knowledge")
    t0 = time.monotonic()
    idx = await store.rebuild()
    print(f"Rebuilt index: {idx.count()} chunks in {time.monotonic() - t0:.1f}s")
    await store.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

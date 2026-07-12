# HumbleChat — Pending Work

## User-side pending actions (non-code)

- Verify `/shared-knowledge` Docker volume mount reflects OWUI KB data on your host
- Restart bot container after mount is active
- Test `/ai "Who is Alderheart?"` and verify lore content injection from HumbleWood KB

## Notes for future work

- The filesystem RAG reader is implemented but depends on the shared volume being correctly mounted — testing awaits container restart
- Both `humblechatsystem` and `trixysmoldersome` configs reference HumbleWood, but models live in separate containers and do not share the Ollama host
- OpenWebUI uses tool calling (not vector search) for RAG — `extra_body["retrieval"]` does nothing from the bot side; filesystem reads are the correct approach here

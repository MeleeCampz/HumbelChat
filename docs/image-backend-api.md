# Talking to the image generation backend

A standalone note on driving this image backend (Unsloth Studio, OpenAI-compatible) from any
program. Nothing else in this repository is required.

## Two faces of the same server

| | base | what it is for |
|---|---|---|
| OpenAI-compatible API | `http://<host>:<port>/v1` | generating images |
| Studio management API | `http://<host>:<port>` (same host, no `/v1`) | loading/unloading the diffusion model |

Auth on both: `Authorization: Bearer <key>`. A key is genuinely required — a request with no
header and a request with a made-up key both answer `401`, so `401` means "wrong key", not
"endpoint missing".

## 1. Make sure a model is resident

One diffusion model lives in VRAM at a time, and it is NOT loaded automatically. Ask first:

```
GET /api/inference/images/status
→ { "loaded": false, "resident_mib": null, "family": null, "engine": "diffusers",
    "supports_negative_prompt": true, "supports_controlnet": false, … }
```

`loaded: true` means ready to paint; `resident_mib` says how much VRAM it holds. When it is
false, load it (this is a 10–25 s operation for a ~12 GB GGUF):

```
POST /api/inference/images/load
{ "model_path": "unsloth/Qwen-Image-2.1-Turbo-GGUF",   # an id from GET /v1/models
  "model_kind": "gguf",
  "gguf_filename": "qwen-image-2.1-turbo-Q4_K_M.gguf",
  "gpu_ids": [0] }
```

- `2xx` → then **poll** `GET /api/inference/images/status` every ~10 s until `loaded: true`.
- `409` → a load is already in progress (another caller got there first): poll, do not resend.
- Still not loaded after ~300 s → `POST /api/inference/images/unload` (clears the stuck slot),
  wait ~15 s, load once more. A load can stall without ever reporting ready; this is the only
  way out that was observed to work.

Model ids come from `GET /v1/models` (some carry a `:QUANT` suffix, e.g.
`unsloth/Qwen-Image-2.1-Turbo-GGUF`). The id you pass to the generate call must match the one
that is loaded.

## 2. Generate

```
POST /v1/images/generations
{ "model": "unsloth/Qwen-Image-2.1-Turbo-GGUF",
  "prompt": "…",
  "n": 1,
  "size": "1024x1024",              # maps to width/height; square
  "response_format": "b64_json",
  "negative_prompt": "…" }
```

Reply:

```json
{ "created": 1795623611, "data": [ { "b64_json": "<base64 PNG>" } ] }
```

`data[0].b64_json` is the only field worth depending on — decode it and write the bytes; the
result is a PNG.

Notes that matter in practice:

- **`negative_prompt` is accepted here** even though some OpenAI SDKs have no kwarg for it —
  pass it in the raw body (with the `openai` SDK: `extra_body={"negative_prompt": …}`).
- **Long prompts are fine.** 3 000+ characters were sent routinely; this backend's text encoder
  is a Qwen3-VL model, so the old 77-token / 1 000-character image-prompt limits do not apply.
- **Size:** 1024 is the comfortable default (~12 s per image once the model is resident, and the
  model itself holds ~12.5 GiB). 2048 works but needs the card to itself and is slower.
- **Supported and unverified here:** `seed` / `seeds[]`, `steps` (default 8), `guidance`
  (default 1.0), multiple `prompts[]` in one call, reference images / edit workflows, LoRAs.
  Not supported for this model family: ControlNet, hires-fix, prompt upsampler.

## 3. Errors, timeouts, retries

| symptom | what it means | what to do |
|---|---|---|
| ~12 s per image at 1024 | normal, after the model is loaded | budget 600 s wall clock anyway |
| `HTTP 500` with "Image generation failed" | transient backend hiccup mid-run | retry once after ~5 s; it recovered every time it was observed |
| `HTTP 500` **and** `status.loaded == false` | the diffusion model was dropped/unloaded | re-run step 1, then retry the generation |
| `HTTP 4xx` | bad request (bad model id, bad size, oversized body) | fix the request — retrying will not help |
| `HTTP 401` on any endpoint | missing or wrong key (also for `GET /v1/models`) | fix the key; nothing else to probe |
| connection refused / reset | server down or port wrong | treat as unavailable, do not retry in a loop |

One generation per HTTP call. If you want N images, loop the call while the model stays loaded —
it does not need re-loading between images.

## 4. Shortest working client

```python
import base64, httpx, time

BASE, KEY = "http://127.0.0.1:8888", "<key>"
MODEL, GGUF = "unsloth/Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-Q4_K_M.gguf"
H = {"Authorization": f"Bearer {KEY}"}

with httpx.Client(timeout=600.0) as c:
    if not c.get(f"{BASE}/api/inference/images/status", headers=H).json().get("loaded"):
        c.post(f"{BASE}/api/inference/images/load", headers=H, json={
            "model_path": MODEL, "model_kind": "gguf",
            "gguf_filename": GGUF, "gpu_ids": [0]})
        while not c.get(f"{BASE}/api/inference/images/status", headers=H).json().get("loaded"):
            time.sleep(10)

    r = c.post(f"{BASE}/v1/images/generations", headers=H, json={
        "model": MODEL, "prompt": "a fox-folk ranger drawing a bow on an old stone bridge",
        "n": 1, "size": "1024x1024", "response_format": "b64_json",
        "negative_prompt": "text, watermark"}).raise_for_status()
    open("out.png", "wb").write(base64.b64decode(r.json()["data"][0]["b64_json"]))
```

curl equivalent, once a model is loaded:

```bash
curl -s -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"unsloth/Qwen-Image-2.1-Turbo-GGUF","prompt":"…","n":1,
       "size":"1024x1024","response_format":"b64_json"}' \
  http://127.0.0.1:8888/v1/images/generations \
| python3 -c 'import sys,base64,json;open("out.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

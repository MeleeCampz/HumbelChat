#!/usr/bin/env python3
"""Diagnostic script for OpenWebUI at http://192.168.X.X:3000/api/v1/

Usage:
    python check_backend.py                        # uses API_KEY from .env
    OPENWEBUI_API_KEY=<key> python check_backend.py  # overrides env var
"""

import urllib.request
import json
import os

from dotenv import load_dotenv
load_dotenv()

API_BASE = os.getenv("OPENAI_API_URL", "http://192.168.X.X:3000/api/v1").rstrip("/")
API_KEY  = os.getenv("OPENWEBUI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")


def _fetch(path: str) -> bytes | None:
    """GET a path on the OpenWebUI API, return raw bytes or None."""
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url)

    # OpenWebUI requires X-API-Key header for all /api/v1/* endpoints
    if API_KEY:
        req.add_header("X-API-Key", API_KEY)

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        print(f"\n  ⚠️ HTTP {e.code} on {url}")
        try:
            body = e.read().decode(errors="replace")
            if body.strip():
                print(f"     Response: {body[:300]}")
        except Exception:
            pass
        return None


def check_backend():
    print(f"Target OpenWebUI API base:\n  {API_BASE}\n")
    print(f"OpenWebUI API Key configured: {'YES' if API_KEY else 'NO'}\n")

    # ── Check models list ───────────────────────────────────────
    data = _fetch("/models")
    if data is None:
        print("\n❌ Could not fetch /models.\n")
        print("Troubleshooting:")
        print("  1. Go to http://192.168.X.X:3000 in your browser")
        print("  2. Settings → Keys (bottom-left)")
        print("  3. Click 'Generate Key' and paste it into OPENWEBUI_API_KEY in .env")
        return

    try:
        response = json.loads(data)
    except json.JSONDecodeError:
        # Some OpenWebUI versions use a different format — dump raw first 400 chars
        print("\n⚠️ Response is not valid JSON. Raw bytes (first 400):")
        print(data[:400])
        return

    models = response.get("data", []) if isinstance(response, dict) else []
    if not models:
        # Try alternative key or structure
        models = response if isinstance(response, list) else []
        # Also try nested under 'models' key
        if not models:
            models = response.get("models", [])

    if not models:
        print("\n⚠️ /models returned empty (non-list/dict).")
        return

    print(f"✅ Connected! Found {len(models)} model(s):\n")
    for m in models:
        name = None
        if isinstance(m, dict):
            name = m.get("id", m.get("model", ""))
        elif isinstance(m, str):
            name = m
        tag = " ← TARGET MODEL" if (name and name == MODEL_NAME) else ""
        print(f"  - {name}{tag}")

    # ── Quick chat test (optional — only if we have a model) ────
    if models and API_KEY:
        sample = models[0] if isinstance(models[0], str) else models[0].get("id")
        if sample:
            print(f"\n💡 Try this chat completion with model '{sample}':")
            url = f"{API_BASE}/chat/completions"
            payload = json.dumps({
                "model": sample,
                "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
                "max_tokens": 20,
            }).encode()
            req = urllib.request.Request(url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            if API_KEY:
                req.add_header("X-API-Key", API_KEY)
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    result = json.loads(resp.read().decode())
                    reply = (result.get("choices", [{}])[0]
                                 .get("message", {})
                                 .get("content", "(no content)"))
                    print(f"\n  User: hello\n  Model: {reply}\n")
            except Exception as e:
                print(f"\n  ⚠️ Chat test failed: {e}")


if __name__ == "__main__":
    check_backend()

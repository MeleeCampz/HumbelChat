"""Phonon TTS live test — synthesize arbitrary text via the local Ollama container.

Usage (from anywhere; stdlib only, any Python 3.8+):

    python scripts/phonon_test.py "Willkommen zur Session!"
    python scripts/phonon_test.py --model phonon:multilingual "Hello / Bonjour"
    echo "Text hier" | python scripts/phonon_test.py          # stdin
    python scripts/phonon_test.py "..." --no-play            # don't auto-open

Requires the Ollama container to be up (loopback :11435):

    docker compose up -d tts-ollama     # always-on service (renamed from embedder-qwen)

First synthesis after idle pays a ~6 s model load (Ollama auto-unloads
after 5 min). Output WAV lands in data/tts_voice_candidates/ by default and
opens in the system player on Windows.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

OLLAMA = os.environ.get("PHONON_OLLAMA_URL", "http://127.0.0.1:11435")


def synthesize(text: str, model: str) -> tuple[bytes, float]:
    body = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/v1/audio/speech", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        audio = r.read()
    return audio, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text", nargs="?", help="text to speak (or pipe via stdin)")
    ap.add_argument("--model", default="phonon:de",
                    help="phonon tag (default phonon:de; try :multilingual, :en, …)")
    ap.add_argument("--out", default=None, help="output path (default: auto in data/tts_voice_candidates/)")
    ap.add_argument("--no-play", action="store_true", help="don't open the result in the system player")
    args = ap.parse_args()

    text = args.text if args.text else sys.stdin.read().strip()
    if not text:
        print("no input — pass text as an argument or via stdin", file=sys.stderr)
        return 1

    try:
        audio, dt = synthesize(text, args.model)
    except Exception as e:  # connection refused → container down
        print(f"synthesis failed: {e}\n"
              f"(is the Ollama container up? docker compose up -d tts-ollama)",
              file=sys.stderr)
        return 1

    out = args.out or os.path.join(
        "data", "tts_voice_candidates",
        f"phonon_test_{time.strftime('%Y%m%d_%H%M%S')}.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(audio)

    # 24 kHz mono 16-bit → bytes / 48000 = seconds
    secs = len(audio) / 48000
    print(f"{args.model}: {len(text)} chars -> {secs:.1f} s audio in {dt:.2f} s "
          f"({'COLD load included' if dt > 3 else 'warm'}) -> {out}")

    if not args.no_play and sys.platform == "win32":
        subprocess.Popen(["cmd", "/c", "start", "", out], shell=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

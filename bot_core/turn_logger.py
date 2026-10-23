"""Per-turn AI records for offline debugging — one JSON file per turn.

Every ``ask_ai`` / ``ask_ai_stream`` call writes a self-contained record to
``logs/ai_turns/<timestamp>_<query-slug>.json`` covering the turn start to
finish: the complete final prompt (system + history block + RAG context +
question), retrieval details (attached files, rerank/dense top scores), agent
mode telemetry (trigger, iterations, tool calls with results via
``gathered_context``, fallback reason), the full response text, and timings.

Design rules:
- **Never breaks a turn.** Every public function is total — any internal
  failure (disk full, bad path, serialization error) is swallowed after a
  single debug log. The logger is observability, not a dependency.
- **Toggleable + bounded.** ``AI_TURN_LOG_ENABLED=0`` disables it entirely
  (zero overhead: record creation short-circuits). ``AI_TURN_LOG_KEEP``
  rotates the folder down to N most-recent files after each write, so the
  folder stays bounded on a long-lived bot.
- **Serialized by construction.** Turns run under the process-wide AI slot,
  so there is no concurrent write/rotation race to guard against.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import (
    AI_TURN_LOG_DIR,
    AI_TURN_LOG_ENABLED,
    AI_TURN_LOG_KEEP,
)

log = logging.getLogger("bot.turn_logger")

_RECORD_VERSION = 1


def _slugify(text: str, max_len: int = 40) -> str:
    """Filesystem-safe slug of the user message (German umlauts included)."""
    norm = unicodedata.normalize("NFKD", text or "")
    ascii_ = "".join(c for c in norm if not unicodedata.combining(c))
    slug = re.sub(r"[^0-9A-Za-z]+", "_", ascii_).strip("_").lower()
    return slug[:max_len].rstrip("_") or "turn"


def new_turn(
    *,
    channel_id: int | None,
    username: str,
    char_key: str | None,
    model: str,
    user_message: str,
) -> dict[str, Any] | None:
    """Start a turn record (or ``None`` when logging is disabled)."""
    if not AI_TURN_LOG_ENABLED:
        return None
    try:
        return {
            "v": _RECORD_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "channel_id": channel_id,
            "username": username,
            "char_key": char_key,
            "model": model,
            "user_message": user_message,
            "retrieval": None,
            "prompt": None,
            "agent": None,
            "response": None,
            "timing_ms": {},
            "status": "pending",
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 — never break the turn
        log.debug("turn record init failed: %s", exc)
        return None


def set_retrieval(
    record: dict[str, Any] | None,
    *,
    attached_files: list[str],
    rag_chars: int,
    retrieval_meta: dict[str, Any] | None = None,
) -> None:
    """Attach the retrieval outcome (what got into the prompt + why)."""
    if record is None:
        return
    try:
        record["retrieval"] = {
            "attached_files": list(attached_files),
            "rag_chars": rag_chars,
            **(dict(retrieval_meta) if retrieval_meta else {}),
        }
    except Exception as exc:  # noqa: BLE001 — never break the turn
        log.debug("turn record retrieval failed: %s", exc)


def set_prompt(record: dict[str, Any] | None, *, messages: list[Any], total_chars: int) -> None:
    """Attach the COMPLETE final prompt exactly as sent to the model."""
    if record is None:
        return
    try:
        # Message params are plain dicts (role/content/...) — JSON-serializable
        # as-is; coerce anything exotic to str so serialization can't fail.
        serializable = [
            {k: v for k, v in (m.items() if isinstance(m, dict) else [("content", str(m))])}
            for m in messages
        ]
        record["prompt"] = {
            "messages": serializable,
            "total_chars": total_chars,
            "estimated_tokens": max(1, total_chars // 4),
        }
    except Exception as exc:  # noqa: BLE001 — never break the turn
        log.debug("turn record prompt failed: %s", exc)


def set_agent(record: dict[str, Any] | None, *, trigger: str | None, meta: Any = None) -> None:
    """Attach agent-mode telemetry (``trigger=None`` → fast path this turn)."""
    if record is None:
        return
    try:
        if trigger is None:
            record["agent"] = {"mode": "off"}
            return
        record["agent"] = {
            "mode": "auto",
            "trigger": trigger,
            "iterations": getattr(meta, "iterations", 0),
            "tool_calls": list(getattr(meta, "tool_calls", [])),
            "fallback": bool(getattr(meta, "fallback", False)),
            "fallback_reason": getattr(meta, "fallback_reason", ""),
            # Every "[tool args]" line + its result — the loop's transcript.
            "gathered_context": getattr(meta, "gathered_context", ""),
        }
    except Exception as exc:  # noqa: BLE001 — never break the turn
        log.debug("turn record agent failed: %s", exc)


def finish_turn(
    record: dict[str, Any] | None,
    *,
    response_text: str = "",
    status: str = "ok",
    error: str = "",
    started_monotonic: float | None = None,
) -> None:
    """Finalize + write the turn record, then rotate the folder.

    ``status`` is ``"ok"``, ``"error"`` (classified failure surfaced to the
    user) or ``"cancelled"`` (/ai stop mid-stream). A pending record that is
    never finished (e.g. validation raised before retrieval) simply isn't
    written — nothing is lost worth persisting.
    """
    if record is None:
        return
    try:
        record["status"] = status
        record["error"] = error
        record["response"] = {
            "text": response_text,
            "chars": len(response_text),
            "approx_tokens": max(1, len(response_text) // 4) if response_text else 0,
        }
        if started_monotonic is not None:
            record["timing_ms"]["total"] = int((time.monotonic() - started_monotonic) * 1000)

        out_dir = Path(AI_TURN_LOG_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_compact = (record.get("ts") or "").replace("-", "").replace(":", "")[:15]
        path = out_dir / f"{ts_compact}_{_slugify(record.get('user_message', ''))}.json"
        # Atomic-ish: write .tmp then replace, so a reader never sees partial JSON.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

        _rotate(out_dir)
    except Exception as exc:  # noqa: BLE001 — never break the turn
        log.debug("turn record write failed: %s", exc)


def _rotate(out_dir: Path) -> None:
    """Keep at most AI_TURN_LOG_KEEP most-recent records in the folder."""
    try:
        if AI_TURN_LOG_KEEP <= 0:
            return
        files = sorted(
            (p for p in out_dir.glob("*.json") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        for stale in files[: max(0, len(files) - AI_TURN_LOG_KEEP)]:
            try:
                stale.unlink()
            except OSError:
                pass  # best-effort cleanup; a stuck file is not an error
    except Exception as exc:  # noqa: BLE001 — never break the turn
        log.debug("turn record rotation failed: %s", exc)


def turns_dir() -> Path:
    """The folder holding per-turn records (for tooling/tests)."""
    return Path(os.getenv("AI_TURN_LOG_DIR") or AI_TURN_LOG_DIR)

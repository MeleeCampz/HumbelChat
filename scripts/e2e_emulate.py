#!/usr/bin/env python3
"""E2E emulation harness — run INSIDE the running bot container.

Invokes a REAL registered slash command (the exact callback wired up in
``main.py``) with a mock interaction, so the full production pipeline
(command dispatch → RAG/embeddings → rerank → LLM → reply formatting) runs
against live config, KB and backends — without any Discord traffic.

Usage (from the repo root on the host):

    docker compose exec -T bot python scripts/e2e_emulate.py /ai message="ping"
    docker compose exec -T bot python scripts/e2e_emulate.py /item_stats
    docker compose exec -T bot python scripts/e2e_emulate.py /character action=list

Options are passed as ``key=value`` pairs matching the command's parameter
names. Supported value types: str, int, float, bool, and Choice[str]
(pass a plain string; use the literal ``None`` for optional params).

Environment:
    E2E_GUILD_ID     fake guild id used in the mock interaction (default 999000001)
    E2E_CHANNEL_ID   fake channel id (default 999000001) — history entries are
                     stored under this key, i.e. test artifacts; clear with /clear_history
    E2E_TIMEOUT      seconds before the run is aborted (default 300)

Exit code: 0 = command completed and produced at least one output,
1 = no output captured, 2 = error/timeout.

NOTE: commands whose delivery path goes through ``bot_core.channel_delivery``
REST fallbacks (e.g. /remind posting later) need a logged-in bot and will fail
loudly here — this harness is for turn-based slash commands.

WARNING: outside the container, if you override EMBED_BACKEND=local while the
stored vector index was built with a different model, the mismatch guard may
attempt to REBUILD the live index DB. Back up
data/knowledge/.vector_index_cache/vector_index.db first (see docs/e2e-testing.md).
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import pathlib
import sys
import time
import types as _types
import typing
from typing import Any, Union, get_args, get_origin

# Make the repo root importable regardless of cwd (script lives in scripts/).
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Importing main builds the bot object and registers every tree command.
# It does NOT connect to Discord (bot.run() only happens under __main__).
import main  # noqa: E402

log = logging.getLogger("bot.e2e_emulate")


class _User:
    def __init__(self) -> None:
        self.id = 999000001
        self.display_name = "e2e-harness"
        self.name = "e2e_harness"

    def __str__(self) -> str:  # handlers log interaction.user directly
        return f"e2e-harness#{self.id}"

    __repr__ = __str__


def _make_entry(kind: str, content: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize a send() call into a recordable entry (embed= and embeds=)."""
    entry: dict[str, Any] = {
        "kind": kind,
        # handlers often pass content=None with an embed — keep it falsy
        "content": "" if content is None else str(content),
        "ephemeral": kwargs.get("ephemeral", False),
    }
    embeds = list(kwargs.get("embeds") or [])
    if kwargs.get("embed") is not None:  # singular embed= kwarg
        embeds.insert(0, kwargs["embed"])
    if embeds:
        entry["embeds"] = [getattr(e, "to_dict", lambda: {"repr": repr(e)})() for e in embeds]
    return entry


class _Response:
    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink
        self._done = False

    def is_done(self) -> bool:
        return self._done

    async def defer(self, **kwargs: Any) -> None:
        self._done = True
        self._sink.append({"kind": "deferred", "ephemeral": kwargs.get("ephemeral", False)})

    async def send_message(self, content: str = "", **kwargs: Any) -> None:
        self._done = True
        self._sink.append(_make_entry("response", content, kwargs))


class _Followup:
    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink

    async def send(self, content: str = "", **kwargs: Any) -> None:
        self._sink.append(_make_entry("followup", content, kwargs))


class MockInteraction:
    """Just enough of discord.Interaction for the command handlers."""

    def __init__(self, client: Any, guild_id: int, channel_id: int) -> None:
        self.client = client
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.user = _User()
        self.channel = None  # no cached TextChannel; typing loop degrades gracefully
        self.guild = None
        self._sink: list[dict[str, Any]] = []
        self.response = _Response(self._sink)
        self.followup = _Followup(self._sink)

    @property
    def sent(self) -> list[dict[str, Any]]:
        return self._sink


def _unwrap_optional(annotation: Any) -> Any:
    """Optional[X] / X | None → X."""
    if get_origin(annotation) in (Union, _types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _coerce(value: str, annotation: Any) -> Any:
    """Convert a CLI string to the parameter's declared type."""
    if value == "None":
        return None
    annotation = _unwrap_optional(annotation)
    import discord.app_commands as app_commands

    if annotation is app_commands.Choice[str]:
        return app_commands.Choice(name=value, value=value)
    if annotation is str:
        return value
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is bool:
        return value.lower() in ("1", "true", "yes")
    # Fallback: pass through as string
    return value


def _build_kwargs(callback: Any, opts: dict[str, str], ix: MockInteraction) -> dict[str, Any]:
    sig = inspect.signature(callback)
    # main.py uses `from __future__ import annotations` — raw signatures carry
    # STRING annotations; get_type_hints evaluates them against the module.
    try:
        hints = typing.get_type_hints(callback)
    except Exception:  # pragma: no cover - exotic annotations fall back to str
        hints = {}
    kwargs: dict[str, Any] = {"interaction": ix}
    for name, param in sig.parameters.items():
        if name == "interaction":
            continue
        if name in opts:
            kwargs[name] = _coerce(opts[name], hints.get(name, str))
        else:
            if param.default is inspect.Parameter.empty:
                print(f"ERROR: missing required option: {name}", file=sys.stderr)
                raise SystemExit(2)
    return kwargs


async def _run(command_name: str, opts: dict[str, str]) -> int:
    name = command_name.lstrip("/")
    cmd = main.bot.tree.get_command(name)
    if cmd is None:
        available = ", ".join(sorted(c.name for c in main.bot.tree.get_commands()))
        print(f"ERROR: unknown command /{name}. Available: {available}", file=sys.stderr)
        return 2

    guild_id = int(os.environ.get("E2E_GUILD_ID", "999000001"))
    channel_id = int(os.environ.get("E2E_CHANNEL_ID", "999000001"))
    timeout = float(os.environ.get("E2E_TIMEOUT", "300"))

    ix = MockInteraction(main.bot, guild_id, channel_id)
    kwargs = _build_kwargs(cmd.callback, opts, ix)

    print(f"== e2e_emulate: invoking /{name} with {opts or '{}'}")
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(cmd.callback(**kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"ERROR: timed out after {timeout:.0f}s", file=sys.stderr)
        return 2
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 — report any failure loudly
        log.exception("e2e_emulate: /%s failed", name)
        print(f"ERROR: /{name} raised {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    elapsed = time.monotonic() - t0

    print(f"-- /{name} completed in {elapsed:.1f}s; captured {len(ix.sent)} output(s):")
    for i, entry in enumerate(ix.sent, 1):
        kind = entry["kind"]
        content = entry.get("content", "")
        if len(content) > 2000:
            content = content[:2000] + f" …[truncated {len(content)} chars total]"
        print(f"  [{i}] {kind} (ephemeral={entry.get('ephemeral', False)}): {content!r}")
        for j, emb in enumerate(entry.get("embeds", []), 1):
            title = emb.get("title") or ""
            desc = str(emb.get("description") or "")[:500]
            print(f"      embed[{j}] title={title!r} description={desc!r}")

    if not ix.sent:
        print("WARNING: command completed but produced no captured output", file=sys.stderr)
        return 1
    return 0


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", help='slash command name, e.g. "/ai" or "item_stats"')
    parser.add_argument(
        "options",
        nargs="*",
        metavar="key=value",
        help="command options matching parameter names, e.g. message=\"ping\"",
    )
    args = parser.parse_args()

    opts: dict[str, str] = {}
    for item in args.options:
        if "=" not in item:
            parser.error(f"option must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        opts[k.strip()] = v

    sys.exit(asyncio.run(_run(args.command, opts)))


if __name__ == "__main__":
    # NB: named `cli`, not `main` — a top-level function called `main` would
    # shadow the imported bot module (`import main`) inside this file.
    cli()

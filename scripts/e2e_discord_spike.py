#!/usr/bin/env python3
"""E2E Discord driver — send REAL traffic at the live bot and verify its reply.

Runs from anywhere that can reach discord.com (e.g. this sandbox). Two modes:

  --mode prefix   (default) Send a plain ``!ai <prompt>`` message via REST.
      The running bot's on_message handler picks it up and runs the FULL
      production pipeline (persona → RAG → rerank → LLM → reply in channel).
      IMPORTANT: the bot ignores messages authored by itself, so use a
      separate "driver" bot token for this mode (E2E_DRIVER_TOKEN) — see
      docs/e2e-testing.md for the 2-minute setup.

  --mode type2    Create a type-2 (application command) message. DOCUMENTED
      DEAD END with a bot token: Discord accepts it but stores it as a plain
      empty type-0 message and dispatches NO interaction to the bot
      (verified live 2026-10-17 — see docs/e2e-testing.md). Kept for reference.

Usage:
    python scripts/e2e_discord_spike.py --mode prefix --text "ping"
    E2E_DRIVER_TOKEN=... python scripts/e2e_discord_spike.py --text "who am I?"
    python scripts/e2e_discord_spike.py --mode type2 --command help

Environment (tokens come from .env automatically):
    E2E_DRIVER_TOKEN   token of a SECOND bot account for prefix mode
                       (falls back to DISCORD_BOT_TOKEN — which the target
                       bot will ignore, so you'll get "no reply")
    E2E_CHANNEL_ID     target channel id (overrides name discovery)
    E2E_CHANNEL_NAME   channel to discover by name (default: "bottesting")
    E2E_TIMEOUT        seconds to wait for the bot's reply (default 120)

Exit codes: 0 = bot replied, 1 = no reply within timeout, 2 = setup/HTTP error.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()
if not os.environ.get("DISCORD_BOT_TOKEN"):
    # not run from the repo root — try <repo>/.env next to scripts/
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")

API = "https://discord.com/api/v10"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bot {token}", "Content-Type": "application/json"}


class Client:
    def __init__(self, token: str) -> None:
        self.token = token
        self.http = httpx.Client(base_url=API, headers=_headers(token), timeout=30)

    def get(self, path: str, **params):
        r = self.http.get(path, params=params or None)
        return self._check(r)

    def post(self, path: str, payload: dict):
        r = self.http.post(path, json=payload)
        return self._check(r)

    @staticmethod
    def _check(r: httpx.Response):
        if r.status_code >= 400:
            raise SystemExit(
                f"HTTP {r.status_code} on {r.request.method} {r.url.path}: "
                f"{r.text[:500]}"
            )
        return r.json() if r.text else None


def resolve_channel(cli: Client, guild_id: int) -> tuple[int, str]:
    """Return (channel_id, channel_name) for the E2E target channel."""
    env_id = os.environ.get("E2E_CHANNEL_ID")
    if env_id:
        return int(env_id), f"(id {env_id})"
    want = os.environ.get("E2E_CHANNEL_NAME", "bottesting").lower()
    channels = cli.get(f"/guilds/{guild_id}/channels")
    matches = [c for c in channels if str(c["type"]) == "0" and c["name"].lower() == want]
    if not matches:
        names = sorted(c["name"] for c in channels if str(c["type"]) == "0")
        raise SystemExit(
            f"no text channel named {want!r} in guild {guild_id}. "
            f"Available: {', '.join(names) or '(none)'} — set E2E_CHANNEL_ID or create the channel."
        )
    ch = matches[0]
    return ch["id"], ch["name"]


def command_ids(cli: Client, app_id: str, guild_id: int) -> dict[str, str]:
    """Map command name → application command id (guild first, then global)."""
    ids: dict[str, str] = {}
    try:
        for entry in cli.get(f"/applications/{app_id}/commands"):
            ids[entry["name"]] = entry["id"]
    except SystemExit:
        pass  # no global commands
    try:
        for entry in cli.get(f"/applications/{app_id}/commands/guilds/{guild_id}"):
            ids[entry["name"]] = entry["id"]  # guild commands win
    except SystemExit:
        pass  # no guild commands — globals only
    return ids


def send_command_message(
    cli: Client, channel_id: int, command_name: str, command_id: str,
    options: list[dict] | None,
) -> dict:
    """Create the type-2 message. Returns the created message."""
    payload: dict = {
        "type": 2,
        "components": [
            {
                "type": 1,
                "components": [
                    {"type": 2, "style": 2, "label": f"/{command_name}", "custom_id": command_id}
                ],
            }
        ],
    }
    if options:
        payload["options"] = options  # spike Q2 — Discord may ignore/reject this
    return cli.post(f"/channels/{channel_id}/messages", payload)


def wait_for_reply(
    cli: Client, channel_id: int, bot_id: str, after_id: str, timeout: float
) -> dict | None:
    """Poll for a message authored by the target bot after `after_id`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msgs = cli.get(f"/channels/{channel_id}/messages", after=after_id, limit=50)
        for m in msgs:
            if m.get("author", {}).get("id") == bot_id and m["id"] != after_id:
                return m
        time.sleep(2)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["prefix", "type2"], default="prefix")
    parser.add_argument("--text", default=None, help='prompt for prefix mode, e.g. "ping"')
    parser.add_argument("--command", default=None, help="command name for type2 mode, e.g. 'help'")
    parser.add_argument(
        "--options", default=None,
        help='JSON list of options for type2 mode (likely ignored by Discord)',
    )
    args = parser.parse_args()

    target_token = os.environ.get("DISCORD_BOT_TOKEN")
    if not target_token:
        raise SystemExit("DISCORD_BOT_TOKEN not set (checked .env)")

    cli = Client(target_token)  # always the TARGET bot's token for polling/replies
    me = cli.get("/users/@me")
    bot_id = me["id"]
    guilds = cli.get("/users/@me/guilds")
    if not guilds:
        raise SystemExit("bot is a member of no guilds")
    guild_id = int(guilds[0]["id"])
    if len(guilds) > 1:
        print(f"note: bot in {len(guilds)} guilds — using first: {guilds[0]['name']} "
              "(set via E2E_CHANNEL_ID to target a specific one)")

    channel_id, channel_name = resolve_channel(cli, guild_id)
    print(f"target bot={me.get('global_name') or me['username']} ({bot_id})  "
          f"guild={guild_id}  channel={channel_name} ({channel_id})")

    t0 = time.monotonic()
    if args.mode == "prefix":
        if not args.text:
            raise SystemExit("--text is required in prefix mode")
        driver_token = os.environ.get("E2E_DRIVER_TOKEN", target_token)
        if driver_token == target_token:
            print("WARNING: no E2E_DRIVER_TOKEN set — sending with the target bot's own "
                  "token, and the bot IGNORES its own messages (expect 'no reply')")
        driver = Client(driver_token)
        prefix = os.environ.get("BOT_PREFIX", "!ai")
        sent = driver.post(f"/channels/{channel_id}/messages", {"content": f"{prefix} {args.text}"})
        print(f"prefix message '{prefix} {args.text}' created: id={sent['id']}")
    else:
        if not args.command:
            raise SystemExit("--command is required in type2 mode")
        options = None
        if args.options:
            try:
                options = json.loads(args.options)
            except json.JSONDecodeError as e:
                raise SystemExit(f"--options is not valid JSON: {e}")
        ids = command_ids(cli, bot_id, guild_id)
        if args.command not in ids:
            raise SystemExit(f"command {args.command!r} not registered. Known: {', '.join(sorted(ids))}")
        sent = send_command_message(cli, channel_id, args.command, ids[args.command], options)
        print(f"type-2 message created: id={sent['id']} (accepted by Discord)")

    reply = wait_for_reply(cli, channel_id, bot_id, sent["id"], float(os.environ.get("E2E_TIMEOUT", "120")))
    if reply is None:
        print(
            f"NO REPLY within {os.environ.get('E2E_TIMEOUT', '120')}s.\n"
            "Interpretation: the bot didn't answer — it may be down, the backend may be\n"
            "down (check logs for health lines), or (prefix mode without E2E_DRIVER_TOKEN)\n"
            "the message was authored by the bot itself and ignored. Check\n"
            "`docker compose logs bot` on the host.",
            file=sys.stderr,
        )
        sys.exit(1)

    content = reply.get("content") or ""
    embeds = [e.get("description", "")[:300] for e in reply.get("embeds", [])]
    print(f"REPLY after {time.monotonic() - t0:.1f}s (message id {reply['id']}):")
    if content:
        print(f"  content: {content[:1000]}")
    for e in embeds:
        print(f"  embed: {e}")
    if not content and not embeds:
        print("  (placeholder/deferred message — the real answer may arrive as an EDIT;")
        print("   watch the channel or re-run with a longer E2E_TIMEOUT)")


if __name__ == "__main__":
    main()

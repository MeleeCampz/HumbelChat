import asyncio
import json
import os
import pathlib
import re
import sys
import uuid
import logging
from dotenv import load_dotenv

load_dotenv()  # Load .env variables into os.environ

import discord
from discord.ext import commands
import discord.app_commands as app_commands
from openai import AsyncOpenAI
import httpx

# ─────────────────────────── Configuration ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("bot")

INTENTS = discord.Intents.default()
INTENTS.messages = True
INTENTS.guilds = True
INTENTS.guild_messages = True
INTENTS.message_content = True

DISCORD_TOKEN  = os.getenv("DISCORD_BOT_TOKEN", "")

# API Key — local endpoints typically ignore it, but we pass something non-empty
OPENAI_API_KEY = (os.getenv("OPENWEBUI_API_KEY") or
                  os.getenv("OPENAI_API_KEY", "local-model-key"))

# Base URL for the local AI backend (OpenWebUI by default):
#   http://localhost:8080/v1
API_BASE_URL = os.getenv("OPENAI_API_URL", "http://localhost:8080/v1")

# Fallback values used when no characters are configured.
DEFAULT_MODEL      = os.getenv("MODEL_NAME", "default-model-name")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant embedded in a Discord bot.",
)

CONTEXT_WINDOW: int = 10
prefix         = os.getenv("BOT_PREFIX", "!ai")


# ─────────────────────────── Reminders ──────────────────────────────

async def _send_reminder(channel_id: int, message: str) -> None:
    """Send a reminder DM or channel message."""
    try:
        chan = bot.get_channel(channel_id)
        if chan:
            await chan.send(f"⏰ **Reminder:** {message}")
    except Exception as e:
        log.error(f"Failed to send reminder: {e}")


async def _reminder_handler(
    channel_id: int,
    delay_seconds: int,
    message: str,
) -> None:
    """Wait *delay_seconds*, then deliver the reminder."""
    await asyncio.sleep(delay_seconds)
    log.info("Delivering reminder for channel %s: %s", channel_id, message)
    await _send_reminder(channel_id, message)


# ─────────────────────────── Characters ──────────────────────────────
CHARACTERS_FILE = pathlib.Path(__file__).parent / "characters.json"

try:
    with open(CHARACTERS_FILE, encoding="utf-8") as fh:
        _char_cfg = json.load(fh)
except FileNotFoundError:
    log.warning("characters.json not found — using single default character.")
    _char_cfg = {
        "default": "Default",
        "characters": {
            "Default": {"model": DEFAULT_MODEL},
        },
    }

CHARACTERS: dict       = _char_cfg.get("characters", {})
DEFAULT_CHARACTER     = _char_cfg.get("default", list(CHARACTERS)[0]) if CHARACTERS else "Default"

# Pre-built Discord slash-command choices (dropdown) — populated at startup from characters.json
_CHAR_CHOICES = [
    discord.app_commands.Choice(name=name, value=name)
    for name in CHARACTERS.keys()
]

# Per-guild channel active character  (key = (guild_id, channel_id))
_active_characters: dict[tuple[int, int], str] = {}


def _get_char_model(gid: int | None, cid: int) -> str:
    """Return the model slug for the current active character."""
    if gid is not None and (gid, cid) in _active_characters:
        name = _active_characters[(gid, cid)]
    else:
        name = DEFAULT_CHARACTER
    char_data = CHARACTERS.get(name, {})
    return char_data.get("model", DEFAULT_MODEL)


def _switch_character(gid: int | None, cid: int, name: str) -> tuple[bool, str]:
    """Set active character. Returns (ok, message)."""
    if name not in CHARACTERS:
        avail = ", ".join(f"`{k}`" for k in CHARACTERS)
        return False, f"Unknown character '{name}'. Available: {avail}"
    if gid is not None:
        _active_characters[(gid, cid)] = name
    entry = CHARACTERS[name].get("model", "")
    return True, f"Switched to **{name}** ({entry})"


# ─────────────────────────── Bot Setup ───────────────────────────────
bot = commands.Bot(command_prefix=prefix, intents=INTENTS)

# Per-channel conversation history: guild_id -> channel_id -> [messages]
chat_histories: dict[int, dict[int, list[dict]]] = {}


def _ensure_history(guild_id: int, channel_id: int) -> None:
    if guild_id not in chat_histories:
        chat_histories[guild_id] = {}
    if channel_id not in chat_histories[guild_id]:
        chat_histories[guild_id][channel_id] = []


def _clear_history(guild_id: int, channel_id: int) -> None:
    chat_histories.get(guild_id, {}).pop(channel_id, None)


# ─────────────────────────── Helpers ─────────────────────────────────


async def ask_ai_with_model(
    user_message: str,
    model: str,
    guild_id: int,
    channel_id: int,
    username: str = "",
) -> str:
    """Send *user_message* to the AI using a specific model slug and return the reply text.

    If *username* is provided, it is prepended to every user message so the AI can
    distinguish between different speakers.
    """
    global KB_KB_ID

    timeout_sec = int(os.getenv("AI_REQUEST_TIMEOUT", "120"))
    system_p = DEFAULT_SYSTEM_PROMPT

    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

    _ensure_history(guild_id, channel_id)
    history = chat_histories[guild_id][channel_id]

    messages: list[dict] = [{"role": "system", "content": system_p}]
    messages += history
    if username:
        user_content = f"**{username}:** {user_message}"
    else:
        user_content = user_message
    messages.append({"role": "user", "content": user_content})

    # Resolve KB ID lazily (only when needed by /ai command)
    kb_id_to_use = KB_KB_ID
    if kb_id_to_use is None:
        log.warning("KB ID not resolved at startup — attempting lazy resolution")
        kb_id_to_use, err = await _resolve_kb_id(KB_KB_NAME)
        if kb_id_to_use:
            KB_KB_ID = kb_id_to_use  # cache
        else:
            log.error("Knowledge base ID resolution failed: %s", err)
    
    extra_body: dict = {}
    if kb_id_to_use:
        extra_body["retrieval"]       = True
        extra_body["knowledge_bases"] = [kb_id_to_use]
    else:
        log.warning("No knowledge base ID available — RAG disabled for this request")
    
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        stream=False,
        timeout=timeout_sec,
        extra_body=extra_body if extra_body else None,
    )

    reply = resp.choices[0].message.content or "(empty response)"

    history.append({"role": "user", "content": f"**{username}:** {user_message}" if username else user_message})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 2 * CONTEXT_WINDOW:
        chat_histories[guild_id][channel_id] = history[-(2 * CONTEXT_WINDOW):]

    return reply


async def _typing_loop(channel, duration_sec: int = 30):
    """Send typing indicators every 10s for up to duration_sec seconds."""
    end_time = asyncio.get_event_loop().time() + duration_sec
    while asyncio.get_event_loop().time() < end_time:
        try:
            await channel.typing()
        except discord.Forbidden:
            break
        except Exception:
            pass
        await asyncio.sleep(10)


async def _send_long_response(source, reply: str, char_name: str = "") -> None:
    """Send `reply`, chunking it into multiple messages if > 1900 chars.

    Paragraph-aware splitting keeps code blocks and lists intact.
    Works for both Slash Commands (followup.send) and prefix commands (reply).
    """
    MAX_LEN = 1900  # Leave room for "[X/Y] " metadata

    header = f"--- {char_name} ---\n" if char_name else ""

    if len(reply) <= MAX_LEN:
        full_text = (header + reply).strip()
        if hasattr(source, 'followup'):
            await source.followup.send(full_text)
        else:
            await source.reply(full_text)
        return

    paragraphs = reply.split('\n\n')
    chunks, current_chunk = [], ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > MAX_LEN:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Single paragraph too long → force-split by words
            if len(para) > MAX_LEN:
                words = para.split()
                sub = ""
                for w in words:
                    if len(sub) + len(w) + 1 > MAX_LEN:
                        chunks.append(sub.strip())
                        sub = w
                    else:
                        sub += " " + w if sub else w
                current_chunk = sub
            else:
                current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk.strip())

    for i, chunk in enumerate(chunks, 1):
        meta = f"[{i}/{len(chunks)}] "
        display_text = (header + meta + chunk).strip()
        if hasattr(source, 'followup'):
            await source.followup.send(display_text)
        else:
            await source.reply(display_text)


# ─────────────────────────── Slash Commands ──────────────────────────

@bot.tree.command(name="ai", description="Send a prompt to the AI and get a reply.")
@app_commands.choices(character=_CHAR_CHOICES)
async def ai_command(
    interaction: discord.Interaction,
    message: str,
    character: str | None = None,
):
    """AI chat command with optional character override. Falls back to System if none specified."""
    guild_id = interaction.guild_id or 0

    # Determine which character/model to use
    if character is not None:
        if character not in CHARACTERS:
            avail = ", ".join(f"`{k}`" for k in CHARACTERS)
            await interaction.followup.send(
                f"Unknown character `{character}`. Available: {avail}", ephemeral=True
            )
            return
        model_to_use = CHARACTERS[character]["model"]
        char_name = character
    else:
        # No character specified — use active per-channel character or fall back to DEFAULT_CHARACTER (System)
        if guild_id is not None and (guild_id, interaction.channel_id) in _active_characters:
            name = _active_characters[(guild_id, interaction.channel_id)]
        else:
            name = DEFAULT_CHARACTER  # "System"
        char_data = CHARACTERS.get(name, {"model": DEFAULT_MODEL})
        model_to_use = char_data["model"]
        char_name = name

    # Defer — may fail if the slash-command invocation expired.
    try:
        await interaction.response.defer()
    except discord.NotFound:
        pass  # already stale; skip typing indicator
    except Exception:
        pass

    asyncio.create_task(_typing_loop(interaction.channel))

    reply = await ask_ai_with_model(
        message, model_to_use, guild_id, interaction.channel_id,
        username=interaction.user.display_name
    )

    # Respond — defer may have failed; fall back to direct channel send.
    try:
        await _send_long_response(interaction, reply, char_name)
    except discord.NotFound:
        await interaction.channel.send(reply.strip())
    except Exception as exc:
        log.error("Failed to send AI response: %s", exc)
        await interaction.channel.send(reply.strip())


@bot.tree.command(
    name="character",
    description="Manage AI character/persona settings.",
)
async def character_command(
    interaction: discord.Interaction,
    action: str = "list",
    name: str | None = None,
):
    """Switch or list characters.

    Usage:
        /character              — lists available characters (default)
        /character set <name>   — switch to a character
        /character show         — shows current active character
        /character reset        — revert to default character
    """
    await interaction.response.defer(ephemeral=True)
    if not CHARACTERS:
        await interaction.followup.send("No characters configured.", ephemeral=True)
        return

    guild_id = interaction.guild_id or 0
    current = _active_characters.get((guild_id, interaction.channel_id), DEFAULT_CHARACTER)
    is_current = lambda n: n == current

    if action == "list":
        lines = ["**Available characters:**\n"]
        for cname, cdata in CHARACTERS.items():
            marker = " ← current" if is_current(cname) else ""
            lines.append(f"  • `{cname}` — model: `{cdata['model']}`{marker}")
        await interaction.followup.send("\n".join(lines), ephemeral=True)

    elif action == "set":
        if name is None:
            await interaction.followup.send("Please provide a character name: `/character set <name>`", ephemeral=True)
            return
        ok, msg = _switch_character(guild_id, interaction.channel_id, name)
        await interaction.followup.send(msg, ephemeral=True)

    elif action == "show":
        char_data = CHARACTERS.get(current, {})
        await interaction.followup.send(
            f"**Current character:** `{current}`\n"
            f"**Model:** `{char_data.get('model', 'N/A')}`",
            ephemeral=True,
        )

    elif action == "reset":
        if guild_id is not None:
            _active_characters.pop((guild_id, interaction.channel_id), None)
        await interaction.followup.send(f"Reverted to default character: **{DEFAULT_CHARACTER}**", ephemeral=True)

    else:
        await interaction.followup.send(
            f"Unknown action '{action}'. Use: list, set, show, reset.",
            ephemeral=True,
        )



@bot.tree.command(
    name="remind",
    description="Schedule a reminder for yourself. Example: /remind 15 minutes Say hello to Bob",
)
@app_commands.describe(
    time_value="Amount of time (number)",
    time_unit="Unit of time (seconds, minutes, hours)",
    message="What you want to be reminded about",
)
async def remind_command(
    interaction: discord.Interaction,
    time_value: int,
    time_unit: str,
    message: str,
):
    """Schedule a one-time reminder."""
    multipliers = {
        "second": 1, "seconds": 1, "s": 1,
        "minute": 60, "minutes": 60, "min": 60, "m": 60,
        "hour": 3600, "hours": 3600, "hr": 3600, "h": 3600,
    }
    unit_lower = time_unit.lower()
    if unit_lower not in multipliers:
        await interaction.response.send_message(
            f"Unknown unit `{time_unit}`. Use: seconds, minutes, hours.",
            ephemeral=True,
        )
        return

    delay = time_value * multipliers[unit_lower]
    if delay < 10:
        await interaction.response.send_message(
            "Reminder must be at least 10 seconds in the future.",
            ephemeral=True,
        )
        return

    # Defer while we confirm, then schedule in background
    await interaction.response.defer(ephemeral=True)

    channel_id = interaction.channel.id
    asyncio.create_task(
        _reminder_handler(channel_id, delay, message)
    )

    unit_singular = time_unit.rstrip("s") if time_value != 1 else time_unit
    confirmation_msg = (
        "\u2705 Reminder set for **" + str(time_value) + " " + unit_singular + "** from now!"
        + "\n📝 I'll ping you with: \"" + message + "\""
    )
    await interaction.followup.send(confirmation_msg, ephemeral=True)




# ─────────────────────────── Knowledge Base Uploads ──────────────────

KB_KB_NAME    = os.getenv("KB_KNOWLEDGE_BASE", "HumbleWood")
KB_KB_ID      : str | None = None  # resolved at startup
KB_DIR         = pathlib.Path(__file__).parent / ".kb_tmp"
KB_DIR.mkdir(exist_ok=True)

_MIME_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "text/plain":      ".txt",
    "text/csv":        ".csv",
    "text/html":       ".html",
    "text/xml":        ".xml",
    "text/markdown":   ".md",
    "application/rtf": ".rtf",
}


async def _fetch_remote_file(url: str, dest_dir: pathlib.Path) -> tuple[pathlib.Path, int | None]:
    """Download a remote file to *dest_dir*, infer its extension from headers or URL.

    Returns (local_path, size).  Size is ``None`` when it can't be determined upfront.
    """
    url_base = url.split("?")[0]
    fname = pathlib.Path(url_base.split("/")[-1])
    ext = ".bin"

    async with httpx.AsyncClient() as client:
        head_resp = await client.head(url, follow_redirects=True)
        ct = (head_resp.headers.get("content-type") or "").split(";")[0].lower()
        ext = _MIME_EXT.get(ct, ".bin")
        if fname.suffix and len(fname.suffix) > 1:
            ext = fname.suffix

    local_path = dest_dir / f"{uuid.uuid4().hex}{ext}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)

    size = len(resp.content)
    return local_path, size


def _read_text_excerpt(local_path: pathlib.Path, max_chars: int = 2000) -> str:
    """Read the first *max_chars* of a file as text (tries utf-8, falls back to latin-1)."""
    try:
        return local_path.read_text(encoding="utf-8")[:max_chars]
    except UnicodeDecodeError:
        return local_path.read_bytes()[:max_chars].decode("latin-1")


def _generate_filename_from_content(local_path: pathlib.Path) -> str:
    """Analyse the file content and return a human-readable filename (no extension).

    Strategy:
      1. Try to read text from the file.
      2. Look for a title line (# heading, <title>, first non-blank line).
      3. Fall back to a short excerpt of the opening words.
    """
    text = _read_text_excerpt(local_path)

    # --- Markdown / reStructuredText headings ---
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()               # "# My Doc" → "My Doc"
        if stripped.startswith("====") and any(c.isalnum() for c in stripped):
            return stripped.replace("=", "").strip()

    # --- HTML <title> ---
    m = re.search(r"<title[^>]*>(.+?)</title>", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # --- First non-blank line (up to 80 chars) ---
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) > 3:
            candidate = stripped[:80]
            break
    else:
        candidate = "uploaded_document"

    return candidate


def _build_owui_headers() -> dict:
    api_key = os.getenv("OPENWEBUI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENWEBUI_API_KEY not set in .env")
    return {"Authorization": f"Bearer {api_key}"}


def _owui_base() -> str:
    """Return the OpenWebUI server base (strip '/api/v1' suffix)."""
    return API_BASE_URL.rsplit("/api", 1)[0]


async def _upload_file_to_owui(local_path: pathlib.Path) -> dict:
    """Step 1 – upload file via /api/v1/files/. Returns the JSON response."""
    server = _owui_base()
    url = f"{server}/api/v1/files/"
    headers = _build_owui_headers() | {"Accept": "application/json"}
    with open(local_path, "rb") as fh:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, files={"file": (local_path.name, fh)})
    if resp.status_code in (200, 201):
        return resp.json()
    return {"error": f"HTTP {resp.status_code}: {resp.text[:300]}"}


async def _wait_for_file_processing(file_id: str, timeout: int = 300) -> dict:
    """Step 2 – poll /api/v1/files/{id}/process/status until 'completed' or failure."""
    server = _owui_base()
    url = f"{server}/api/v1/files/{file_id}/process/status"
    headers = _build_owui_headers()
    async with httpx.AsyncClient(timeout=timeout) as client:
        import time
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                return {"error": f"File processing timed out after {timeout}s"}
            resp = await client.get(url, headers=headers)
            data = resp.json()
            status = data.get("status")
            if status == "completed":
                return data
            if status == "failed":
                return {"error": f"File processing failed: {data.get('error')}"}
            # pending — poll again
            await asyncio.sleep(2)


async def _resolve_kb_id(kb_name: str) -> tuple[str | None, str]:
    """Resolve a knowledge-base name (or slug) to its id. Returns (kb_id, error_msg)."""
    global KB_KB_ID
    server = _owui_base()
    headers = _build_owui_headers()

    # Only the canonical OWUI endpoint is needed now
    url = f"{server}/api/v1/knowledge/"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers)

    if resp.status_code != 200 or not resp.text.strip():
        return None, f"KB endpoint returned status {resp.status_code}"

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        log.warning("KB list endpoint %s returned non-JSON body", url)
        return None, "KB endpoint returned invalid JSON"

    kbs = data.get("items") if isinstance(data, dict) and "items" in data else (data if isinstance(data, list) else [])
    if not kbs:
        return None, f"Knowledge base is empty — create a KB called '{kb_name}' in OpenWebUI first"

    kb_id = None
    for kb in kbs:
        name = kb.get("name", "")
        ident = kb.get("id", "")
        if name == kb_name or ident == kb_name:
            kb_id = ident
            break
    if not kb_id:
        for kb in kbs:
            n = kb.get("name", "").lower()
            if kb_name.lower() in n:
                kb_id = kb["id"]
                break
    if not kb_id:
        names = [kb.get("name", "?") for kb in kbs]
        return None, f"KB '{kb_name}' not found (available: {names})"

    KB_KB_ID = kb_id
    log.info("Resolved KB '%s' → id %s", kb_name, kb_id)
    return kb_id, ""


async def _add_file_to_kb(kb_name: str, file_id: str) -> dict:
    """Step 3 – resolve kb_name → kb_id, then POST /api/v1/knowledge/{id}/file/add."""
    kb_id, err = await _resolve_kb_id(kb_name)
    if kb_id is None:
        return {"error": err}
    server = _owui_base()
    headers = _build_owui_headers()
    add_url = f"{server}/api/v1/knowledge/{kb_id}/file/add"
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(add_url, headers=headers | {"Content-Type": "application/json"},
                                 json={"file_id": file_id})
    if resp.status_code in (200, 201):
        return resp.json()
    return {"error": f"Add to KB failed: HTTP {resp.status_code}: {resp.text[:300]}"}


async def _list_kb_documents() -> list[dict]:
    """List all documents across knowledge bases."""
    server = _owui_base()
    headers = _build_owui_headers()
    # List all KBs
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{server}/api/v1/knowledge/", headers=headers)
    log.info("KB list status=%d resp=%s", resp.status_code, resp.text[:500])
    if resp.status_code != 200:
        return []
    kbs_response = resp.json()
    # API returns { "items": [...], "total": N }, not a bare list
    if isinstance(kbs_response, dict):
        kbs = kbs_response.get("items", [])
    else:
        kbs = kbs_response
    all_docs: list[dict] = []
    debug_folder_map: dict = {}  # tmp: track directory memberships
    for kb in kbs:
        if isinstance(kb, dict):
            kb_id = kb.get("id", "")
            kb_name = kb.get("name", kb.get("title", "unknown"))
        else:
            kb_id = str(kb)
            kb_name = str(kb)
        log.info("Checking KB: id=%s name=%s", kb_id, kb_name)
        
        # Try the files endpoint
        files_url = f"{server}/api/v1/knowledge/{kb_id}/files"
        async with httpx.AsyncClient(timeout=30) as client:
            resp3 = await client.get(files_url, headers=headers)
        log.info("GET %s status=%d body=%s", files_url, resp3.status_code, resp3.text[:500])
        if resp3.status_code == 200:
            file_data = resp3.json()
            docs_list = None
            if isinstance(file_data, list):
                docs_list = file_data
            elif isinstance(file_data, dict):
                # API returns { "items": [...], "directories": [...] }
                docs_list = file_data.get("items", [])
                # Track directory -> files mapping and add directories as virtual entries
                for d in file_data.get("directories", []):
                    if isinstance(d, dict):
                        dir_name = d.get("name", "?")
                        all_docs.append({"name": dir_name, "size": 0, "kb_name": kb_name, "is_directory": True})
                        # Try to find which files belong to this directory
                        if "id" in d:
                            debug_folder_map[d["id"]] = dir_name
                        log.info("Found directory: %s keys=%s", dir_name, list(d.keys())[:8])
            if docs_list:
                # Build directory_id -> name mapping
                dir_id_map = {}
                for d in file_data.get("directories", []):
                    if isinstance(d, dict) and "id" in d:
                        dir_id_map[d["id"]] = d.get("name", "?")
                
                for item in docs_list:
                    entry = dict(item)
                    entry["kb_name"] = kb_name
                    # Determine which folder this file belongs to
                    meta_data = (entry.get("meta") or {}).get("data", {})
                    if not isinstance(meta_data, dict):
                        meta_data = {}
                    dir_id = meta_data.get("directory_id")
                    if dir_id and dir_id in dir_id_map:
                        entry["_folder"] = dir_id_map[dir_id]
                    else:
                        entry["_folder"] = None  # root
                    # Check what fields are available on items
                    log.info("Item fields: %s", {k: entry.get(k) for k in ["name", "filename", "folder_id", "path", "metadata", "meta"] if k in entry})
                    all_docs.append(entry)
    log.info("Total documents found: %d", len(all_docs))
    # Filter to only the target KB
    all_docs = [d for d in all_docs if d.get("kb_name") == KB_KB_NAME]
    log.info("Documents in %s: %d", KB_KB_NAME, len(all_docs))
    return all_docs


@bot.tree.command(
    name="upload_kb",
    description="Upload a file to the knowledge base for RAG. Attach a file or provide a URL.",
)
async def upload_kb_command(
    interaction: discord.Interaction,
    kb_name: str | None = None,
    url: str | None = None,
    file: discord.Attachment = None,
):
    global KB_KB_NAME
    if kb_name:
        KB_KB_NAME = kb_name

    await interaction.response.defer(ephemeral=True)

    source_type = None  # 'attachment' or 'url'
    local_path = None
    filename = "uploaded"
    size = None

    # Priority: attachment > url > error
    if file:
        source_type = "attachment"
        local_path, size = await _fetch_remote_file(file.url, KB_DIR)
        filename = file.filename
    elif interaction.attachments:
        att = interaction.attachments[0]
        if att.size and att.size < 20 * 1024 * 1024:
            source_type = "attachment"
            local_path, size = await _fetch_remote_file(att.url, KB_DIR)
            filename = att.filename
    elif url:
        source_type = "url"
        url_stripped = url.rstrip("/")
        basename = url_stripped.split("/")[-1].split("?")[0] if "/" in url_stripped else "downloaded_file"
        filename = basename or "downloaded_file"
        local_path, size = await _fetch_remote_file(url, KB_DIR)

    if not source_type:
        await interaction.followup.send(
            "⚠️ Please attach a file (using the 📎 button below) or provide a URL. Max 20 MB.",
            ephemeral=True,
        )
        return

    log.info("Processing KB upload: %s (%s)", filename, source_type)

    # Auto-generate a descriptive filename from the file content and rename temp file
    kb_display_name = _generate_filename_from_content(local_path)
    safe_name = "".join(c for c in kb_display_name if c.isalnum() or c in "._- ").strip()
    safe_name = (safe_name[:60] + "...") if len(kb_display_name) > 60 else safe_name
    ext = local_path.suffix
    renamed_path = local_path.parent / f"{safe_name}{ext}"
    if renamed_path.exists():
        renamed_path.unlink()
    local_path.rename(renamed_path)
    local_path = renamed_path  # update reference so downstream uses the new name

    # Step 1: Upload the file to OpenWebUI (now uses local_path.name automatically)
    await interaction.followup.send(
        "⏳ Step 1/3: Uploading file...",
        ephemeral=True,
    )
    result = await _upload_file_to_owui(local_path)
    if "error" in result:
        log.warning("KB upload failed at step 1: %s", result["error"])
        await interaction.followup.send(
            f"❌ Upload failed: {result['error']}",
            ephemeral=True,
        )
        local_path.unlink(missing_ok=True)
        return

    file_id = result.get("id", "")
    if not file_id:
        await interaction.followup.send(
            "❌ Upload succeeded but no file ID returned. Response: " + json.dumps(result, indent=2, default=str)[:1000],
            ephemeral=True,
        )
        local_path.unlink(missing_ok=True)
        return

    log.info("File uploaded with id: %s", file_id)

    # Step 2: Wait for processing to complete
    await interaction.followup.send(
        "⏳ Step 2/3: Processing file (extracting text, computing embeddings)...",
        ephemeral=True,
    )
    proc_result = await _wait_for_file_processing(file_id)
    if "error" in proc_result:
        log.warning("KB upload failed at step 2: %s", proc_result["error"])
        await interaction.followup.send(
            f"⚠️ File uploaded but processing failed: {proc_result['error']}",
            ephemeral=True,
        )
        local_path.unlink(missing_ok=True)
        return

    log.info("File processing completed.")

    # Step 3: Add file to knowledge base
    await interaction.followup.send(
        "⏳ Step 3/3: Adding file to knowledge base...",
        ephemeral=True,
    )
    add_result = await _add_file_to_kb(KB_KB_NAME, file_id)
    local_path.unlink(missing_ok=True)

    if "error" in add_result:
        log.warning("KB upload failed at step 3: %s", add_result["error"])
        status_emoji = "⚠️"
        detail = json.dumps(add_result, indent=2, default=str)[:1500]
    else:
        status_emoji = "✅"
        detail = json.dumps(add_result, indent=2, default=str)[:1500]

    msg_lines = []
    kb_display = f"**{KB_KB_NAME}**" if status_emoji == "✅" else f"**{KB_KB_NAME}** (partial)"
    msg_lines.append(f"{status_emoji} Uploaded to KB {kb_display}")
    msg_lines.append(f"📄 File: `{kb_display_name}`")
    msg_lines.append(f"📥 Source: `{source_type}`")
    # `size` was already captured at download time (before the file was unlinked in step 2/3)
    if size is not None:
        msg_lines.append(f"📦 Size: `{size:,}` bytes")
    msg_lines.append(f"🆔 File ID: `{file_id}`")
    msg_lines.append("🔖 Details:")
    msg_lines.append("```\n" + detail + "\n```")
    await interaction.followup.send("\n".join(msg_lines), ephemeral=True)
    local_path.unlink(missing_ok=True)


@bot.tree.command(
    name="list_kb_docs",
    description="List all documents currently in the knowledge base.",
)
async def list_kb_docs_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    docs = await _list_kb_documents()
    if not docs:
        await interaction.followup.send(
            f"📚 Knowledge base **'{KB_KB_NAME}'** is empty. Use `/upload_kb` to add files.",
            ephemeral=True,
        )
        return

    lines = [f"📚 Documents in **'{KB_KB_NAME}'** ({len(docs)}):"]

    # Build a tree using the _folder field set during collection (from directory_id),
    # or fallback to parsing "/" from the name for legacy entries.
    root_map: dict[str, list[dict]] = {}  # folder -> list of file entries
    no_folder_docs: list[dict] = []  # files with no folder path

    for d in docs[:100]:
        full_name = d.get("name", d.get("filename", "unknown"))
        size = d.get("size") or (d.get("meta") or {}).get("size", 0)
        if isinstance(size, str):
            try:
                size = int(size)
            except (ValueError, TypeError):
                size = 0
        is_folder_entry = d.get("is_directory", False)

        # Use _folder from directory_id mapping; fallback to parsing / in name
        folder = d.get("_folder")
        if folder is None and "/" in full_name and not is_folder_entry:
            parts = full_name.rsplit("/", 1)
            folder = parts[0]

        basename = os.path.basename(full_name) if "/" in full_name else full_name
        if folder is not None and not is_folder_entry:
            if folder not in root_map:
                root_map[folder] = []
            root_map[folder].append({"name": basename, "size": size})
        else:
            # File in root or a folder entry
            no_folder_docs.append({"name": full_name, "size": size, "is_folder": is_folder_entry})

    # List root-level folders first, then files
    if no_folder_docs:
        lines.append(f"")
        lines.append(f"📂 **Root:**")
        for entry in sorted(no_folder_docs, key=lambda x: (not x.get("is_folder", False), x["name"].lower())):
            icon = "📁 " if entry.get("is_folder", False) else "  📄 "
            lines.append(f"{icon} `{entry['name']}` — {entry['size']:,} bytes")

    # List files in each folder as sub-sections (after root)
    for folder in sorted(root_map.keys()):
        lines.append(f"")
        lines.append(f"📁 **{folder}/**")
        for entry in sorted(root_map[folder], key=lambda x: (not x.get("is_folder", False), x["name"].lower())):
            icon = "📁 " if entry.get("is_folder", False) else "  📄 "
            lines.append(f"{icon} `{entry['name']}` — {entry['size']:,} bytes")

    await interaction.followup.send("\n".join(lines), ephemeral=True)

@bot.tree.command(
    name="ocr",
    description="Extract text from an image (OCR) using vision AI.",
)
async def ocr_command(interaction: discord.Interaction, image: discord.Attachment = None):
    """Send an image attachment to a vision model and return the extracted text."""
    await interaction.response.defer(ephemeral=True)

    if not image:
        await interaction.followup.send(
            "⚠️ Please attach an image to this command.",
            ephemeral=True,
        )
        return

    async def _ask_vision():
        # Download the image and encode it as a base64 data URI
        async with httpx.AsyncClient() as client:
            img_resp = await client.get(image.url)
        img_data = img_resp.content

        # Determine MIME type from content or filename
        mime = "image/png"
        if image.filename and image.filename.lower().endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        elif image.filename and image.filename.lower().endswith(".gif"):
            mime = "image/gif"
        elif image.filename and image.filename.lower().endswith(".webp"):
            mime = "image/webp"
        import base64
        b64 = base64.b64encode(img_data).decode("utf-8")
        data_uri = f"data:{mime};base64,{b64}"

        client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
        resp = await client.chat.completions.create(
            model=os.getenv("VISION_MODEL", DEFAULT_MODEL),
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Extract all text from this image accurately. Do not interpret or summarize — return the exact text as you see it."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]}],
            temperature=0,
            max_tokens=4096,
        )
        return resp.choices[0].message.content or "(no text found)"

    reply = await _ask_vision()
    MAX_LEN = 1900
    if len(reply) <= MAX_LEN:
        await interaction.followup.send(f"🔍 Extracted text:\n\n{reply}", ephemeral=True)
    else:
        for i in range(0, len(reply), MAX_LEN):
            chunk = reply[i:i+MAX_LEN]
            await interaction.followup.send(f"🔍 Extracted text (part {i//MAX_LEN+1}):\n\n{chunk}", ephemeral=True)


@bot.tree.command(
    name="summarize",
    description="Summarize an uploaded file or the current conversation context.",
)
@app_commands.describe(
    file_url="Optional: URL of a text file to summarize. If omitted, summarizes recent chat history.",
)
async def summarize_command(
    interaction: discord.Interaction,
    file_url: str | None = None,
):
    """Summarize text from a file or recent context."""
    await interaction.response.defer(ephemeral=True)

    text_to_summarize = ""

    if file_url:
        async with httpx.AsyncClient() as client:
            resp = await client.get(file_url)
            resp.raise_for_status()
            text_to_summarize = resp.text[:32000]
        source_label = f"file from `{file_url[:80]}...`"
    else:
        guild_id = interaction.guild_id or 0
        cid = interaction.channel_id
        history = chat_histories.get(guild_id, {}).get(cid, [])
        parts = []
        for msg in history[-30:]:
            role_name = {"user": "User", "assistant": "AI"}.get(msg["role"], msg["role"])
            parts.append(f"[{role_name}]: {msg['content']}")
        text_to_summarize = "\n\n".join(parts) if parts else "(no history available)"
        source_label = "recent conversation"

    if not text_to_summarize or text_to_summarize.strip() in ("(no history available)",):
        await interaction.followup.send("⚠️ Nothing to summarize. Provide a file URL or send messages first.", ephemeral=True)
        return

    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
    resp = await client.chat.completions.create(
        model=os.getenv("SUMMARIZE_MODEL", DEFAULT_MODEL),
        messages=[{"role": "system", "content": f"You are a precise summarizer. Summarize the following text from {source_label} in clear bullet points. Be concise but complete."},
                  {"role": "user", "content": text_to_summarize}],
        temperature=0.3,
        max_tokens=2048,
    )
    summary = resp.choices[0].message.content or "(empty)"

    if len(summary) <= 1900:
        await interaction.followup.send(f"📝 Summary of {source_label}:\n\n{summary}", ephemeral=True)
    else:
        for i in range(0, len(summary), 1900):
            chunk = summary[i:i+1900]
            await interaction.followup.send(f"📝 Summary (part {i//1900+1}):\n\n{chunk}", ephemeral=True)


@bot.tree.command(
    name="translate",
    description="Translate text. Usage: /translate target_language:text_or_no_text",
)
@app_commands.describe(
    target_language="Target language and optional source text (e.g. 'Spanish: Hello world' or just 'Spanish')",
    source_language="Optional source language (default: auto-detect)",
)
async def translate_command(
    interaction: discord.Interaction,
    target_language: str,
    source_language: str | None = None,
):
    """Translate text to a target language."""
    parts = target_language.split(":", 1)
    tgt_lang = parts[0].strip()
    text_to_translate = parts[1].strip() if len(parts) > 1 else None

    if not text_to_translate:
        guild_id = interaction.guild_id or 0
        cid = interaction.channel_id
        history = chat_histories.get(guild_id, {}).get(cid, [])
        if not history:
            await interaction.followup.send("⚠️ No text to translate. Provide text as: `/translate Spanish: Hello world`", ephemeral=True)
            return
        last_user = [m["content"] for m in reversed(history) if m["role"] == "user"]
        if last_user:
            text_to_translate = last_user[0]
        else:
            await interaction.followup.send("⚠️ No user message found to translate.", ephemeral=True)
            return

    src_clause = f" from {source_language}" if source_language else ""

    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
    resp = await client.chat.completions.create(
        model=os.getenv("TRANSLATE_MODEL", DEFAULT_MODEL),
        messages=[{
            "role": "system",
            "content": f"You are a professional translator. Translate the following text{src_clause} into {tgt_lang}. Return ONLY the translated text — no explanations, no quotes, no metadata.",
        }, {
            "role": "user",
            "content": text_to_translate,
        }],
        temperature=0.3,
        max_tokens=4096,
    )
    translated = resp.choices[0].message.content or "(translation failed)"

    if len(translated) <= 1900:
        await interaction.followup.send(f"🌐 Translated to **{tgt_lang}**:\n\n{translated}", ephemeral=True)
    else:
        for i in range(0, len(translated), 1900):
            chunk = translated[i:i+1900]
            await interaction.followup.send(f"🌐 Translated to **{tgt_lang}** (part {i//1900+1}):\n\n{chunk}", ephemeral=True)


@bot.tree.command(name="clear_history", description="Clear conversation history for this channel.")
async def clear_history_command(interaction: discord.Interaction):
    if interaction.guild_id and interaction.channel_id:
        _clear_history(interaction.guild_id, interaction.channel_id)
    await interaction.response.send_message("Conversation history cleared.", ephemeral=True)


# ─────────────────────────── Event Handlers ──────────────────────────

@bot.event
async def on_ready():
    log.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    if bot.guilds:
        bot.tree.copy_global_to(guild=discord.Object(id=bot.guilds[0].id))
        await bot.tree.sync(guild=discord.Object(id=bot.guilds[0].id))
    else:
        await bot.tree.sync()
    log.info("Characters loaded: %s", ", ".join(CHARACTERS) or "(none)")
    
    # Resolve KB ID at startup so /ai commands don't block waiting for it
    try:
        kb_id, err = await _resolve_kb_id(KB_KB_NAME)
        if kb_id:
            global KB_KB_ID
            KB_KB_ID = kb_id
            log.info("✅ Knowledge base '%s' resolved at startup (ID: %s)", KB_KB_NAME, kb_id)
        else:
            log.error("❌ Failed to resolve knowledge base at startup: %s", err)
    except Exception as e:
        log.error("❌ Error resolving KB ID at startup: %s", e)


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    content = message.content.strip()
    if not content.startswith(prefix):
        return

    prompt = content[len(prefix):].strip()
    if not prompt:
        await message.channel.send(f"Usage: {prefix} <your question>")
        return

    guild_id = message.guild_id or 0
    log.info("%s (%s) in #%s: %s",
             message.author, message.author.id, message.channel.name, prompt[:80])

    await message.channel.typing()
    # Prefix command always uses default character (System)
    sys_model = CHARACTERS.get(DEFAULT_CHARACTER, {}).get("model", DEFAULT_MODEL)
    reply = await ask_ai_with_model(prompt, sys_model, guild_id, message.channel.id, username=message.author.display_name)
    await _send_long_response(message, reply, DEFAULT_CHARACTER)


# ─────────────────────────── Single-instance lock (prevents duplicate bots) ────────

PIDFILE = pathlib.Path(__file__).parent / ".bot.pid"
SOCK_FILE = pathlib.Path(__file__).parent / ".bot.sock"


def _enforce_single_instance():
    """Exit immediately if another instance of this bot is already running.

    Checks both a PID file and a Unix domain socket to reliably detect stale or
    live duplicate processes, even after crashes where the PID file was not cleaned up.
    """
    import socket as _socket

    # 1) Check for an active listener on our unique port (fallback in case sock is gone)
    try:
        test_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        test_sock.settimeout(0.5)
        test_sock.connect(("127.0.0.1", 18765))  # arbitrary unused port
        test_sock.close()
        log.info("Another bot instance is already running (port 18765). Exiting.")
        sys.exit(0)
    except Exception:
        test_sock.close() if not test_sock._closed else None
        pass  # nobody listening — safe to proceed

    # 2) Check PID file for a stale entry
    if PIDFILE.exists():
        try:
            old_pid = int(PIDFILE.read_text().strip())
            import os as _os, signal as _signal
            _os.kill(old_pid, 0)  # raises ProcessLookupError if dead
            log.info("Another bot instance (PID %d) is already running. Exiting.", old_pid)
            sys.exit(0)
        except (ProcessLookupError, ValueError):
            PIDFILE.unlink(missing_ok=True)  # stale file — remove it

    # 3) Write our own PID and set cleanup on exit
    PIDFILE.write_text(str(os.getpid()))
    import atexit as _atexit

    @_atexit.register
    def _cleanup_lock():
        try:
            PIDFILE.unlink(missing_ok=True)
        except OSError:
            pass


# ─────────────────────────── Startup ─────────────────────────────────

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        log.error("Please set the DISCORD_BOT_TOKEN environment variable.")
        raise SystemExit(1)

    _enforce_single_instance()

    log.info("Connecting to local AI backend at: %s", API_BASE_URL)
    log.info("Using default model:  %s", DEFAULT_MODEL)
    bot.run(DISCORD_TOKEN)

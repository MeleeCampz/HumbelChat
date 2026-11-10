# Bot Permissions

What Discord permissions the bot needs to work, and where to set them.

## Minimum required (server-wide)

| Permission | Why |
|---|---|
| **View Channel** | The bot must be able to see each channel it posts in — chat replies, session reports, and scheduled reminder delivery all fail silently-invisible if this is missing (Discord simply refuses the send). This is also the permission that lets background senders fetch the channel object at all. |
| **Send Messages** | Every outbound feature: `/ai` replies, prefix (`!ai`) replies, command follow-ups, and reminder messages. |
| **Read Message History** | Needed so the bot can read a channel's recent context when it posts there (e.g. delivery into channels it hasn't seen yet). Grant this to be safe. |
| **Attach Files** | Anything that sends a file: session highlight PNGs (`bot_core/session_images.py`), `/generate_image` results, `/session_file` exports, `/speak` audio. Without it every one of those sends is refused. |

That's the entire functional set. Nothing else is required for the core loop.

> Docs note: this row used to say "no feature sends files yet". That stopped being
> true with session highlights (#69), `/generate_image`, `/session_file` and `/speak`.

## Recommended (nice to have)

| Permission | Why |
|---|---|
| **Send Messages in Threads** | If you ever run sessions or reminders inside threads, sends there need this. Not needed for plain channels. |
| **Create Public Threads / Manage Threads** | Only if the target is a **forum** channel: a forum parent is not directly messageable, so posting means creating a post. |

## What the bot does NOT need

No admin, no kick/ban, no manage channels, no webhooks, no emoji management. Keep the bot role minimal — it is a chat participant, not a moderator.

## Where to set them

### Option A — invite link (first-time install)
When adding the bot via its OAuth2 URL, the permission selector should show only:
**View Channels, Send Messages, Read Message History, Attach Files.** If the generated URL requests more, regenerate it without the extras.

### Option B — existing role
`Server Settings → Roles → <bot's role> → Permissions`, then enable **View Channel**, **Send Messages**, **Read Message History**, and **Attach Files**.

## Why chat can work while background posts fail

The two outbound paths have **different requirements**, which is why a broken
channel often looks healthy:

| Path | Used by | Needs |
|---|---|---|
| **Interaction webhook** (`interaction.response.*`, `followup.send`) | every slash command reply, including `/ai`, `/end_session`, `/session_notes` | nothing from the channel — Discord authorizes the reply with the interaction token |
| **Channel send** (`resolve_channel()` → `chan.send()`) | session highlights, next-session reminders, saved roll views, `/remind` delivery | `View Channel` (to fetch the object by id) **and** the send permissions |

So "the bot answers me fine" proves nothing about the second path. A channel that
rejects `GET /channels/<id>` with `403 Missing Access` still serves slash
commands all day, while `bot_core/session_images.py` logs
`highlight channel … undeliverable — images will only be archived to disk`.

### Diagnose it with `/botperms`

```
/botperms                      # the channel you ran it in
/botperms 869602082552954973   # or a <#mention>, or a channel name
```

It runs the same two-step resolution background senders run (`bot.get_channel()`
then the REST fallback), prints the bot's permission bits in that channel, flags
forum/thread targets, and ends with a verdict naming the permission to fix. When
the channel is unreadable, `/botperms <id>` answering *"Discord refused to hand
channel … 403 Missing Access"* **is** the diagnosis.

## Per-channel overrides (important)

Server-wide grants can be overridden per channel, which is a common trap:

1. Right-click the target channel → **Edit Channel → Role Permissions**
2. Select the bot's role and confirm **View Channel** and **Send Messages** are allowed (not ignored/denied by another row)
3. The same applies to any "only members of X can see this" private setup — the bot's role must be explicitly allowed there, it does not inherit from your own membership

Real-world example: the `#private-bot` channel initially rejected all reminder sends even though the bot worked everywhere else — its Role Permissions tab had no explicit allow for the bot's role. Adding **View Channel** + **Send Messages** fixed delivery (verified 2026-08-28).

## Symptom table

| Symptom | Likely permission cause |
|---|---|
| Bot replies fine in some channels, silent in others | Per-channel override is denying View/Send on the affected channel — check its Role Permissions tab |
| Slash commands work, but session highlights are archived to disk only | The channel object cannot be fetched (`View Channel` denied, or the id is a forum parent rather than a post) — run `/botperms` |
| Next-session reminder never arrives | Same as above: the delivery is a background send, not an interaction reply |
| `roll_buttons: channel … unavailable; view not re-attached` in the logs | Channel missing from the gateway cache **and** refused by REST — `/botperms` shows which |
| Reminder never arrives, no log error about delivery target | Target channel denies View Channel (creation of the reminder may be rejected with a permissions note) |
| Bot appears online but ignores you completely | It can't see your channel at all (View Channel denied server-wide or per-channel) |

If the logs show a clean send attempt and Discord still refuses, use `./botctl.sh logs` — the delivery layer logs every failed send with the channel name since 2026-08-28.

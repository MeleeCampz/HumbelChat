# Bot Permissions

What Discord permissions the bot needs to work, and where to set them.

## Minimum required (server-wide)

| Permission | Why |
|---|---|
| **View Channel** | The bot must be able to see each channel it posts in — chat replies, session reports, and scheduled reminder delivery all fail silently-invisible if this is missing (Discord simply refuses the send). |
| **Send Messages** | Every outbound feature: `/ai` replies, prefix (`!ai`) replies, command follow-ups, and reminder messages. |
| **Read Message History** | Needed so the bot can read a channel's recent context when it posts there (e.g. delivery into channels it hasn't seen yet). Grant this to be safe. |

That's the entire functional set. Nothing else is required for the core loop.

## Recommended (nice to have)

| Permission | Why |
|---|---|
| **Send Messages in Threads** | If you ever run sessions or reminders inside threads, sends there need this. Not needed for plain channels. |
| **Attach Files** | Only if you enable features that send files back out (none currently do — KB uploads read your attachment from *you*, they don't upload anything). |

## What the bot does NOT need

No admin, no kick/ban, no manage channels, no webhooks, no emoji management. Keep the bot role minimal — it is a chat participant, not a moderator.

## Where to set them

### Option A — invite link (first-time install)
When adding the bot via its OAuth2 URL, the permission selector should show only:
**View Channels, Send Messages, Read Message History.** If the generated URL requests more, regenerate it without the extras.

### Option B — existing role
`Server Settings → Roles → <bot's role> → Permissions`, then enable **View Channel**, **Send Messages**, and **Read Message History**.

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
| Reminder never arrives, no log error about delivery target | Target channel denies View Channel (creation of the reminder may be rejected with a permissions note) |
| Bot appears online but ignores you completely | It can't see your channel at all (View Channel denied server-wide or per-channel) |

If the logs show a clean send attempt and Discord still refuses, use `./botctl.sh logs` — the delivery layer logs every failed send with the channel name since 2026-08-28.

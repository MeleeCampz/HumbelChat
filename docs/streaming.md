# Streaming delivery

AI replies are delivered **streamed**: the bot posts a message and edits it in
place as tokens arrive, so long answers appear to "grow" live instead of
appearing all at once after a long wait. Delivery is throttled so it never
hammers Discord's rate limits (message edits are limited to ~5/s per channel).

## How it works

- The first visible update fires once at least `STREAM_MIN_FLUSH_CHARS` of
  text has arrived, so one-word replies don't get posted as an empty stub.
- Subsequent updates fire when the buffer grows by another
  `STREAM_MIN_FLUSH_CHARS` **or** `STREAM_MAX_FLUSH_INTERVAL` seconds have
  passed since the last edit (whichever comes first).
- The final flush guarantees the complete text lands, even if the stream ends
  abruptly (errors are still reported to the user).

## Long replies: multi-message streaming with freeze-on-split

Discord messages are hard-capped at **2000 characters**. When a reply outgrows
that limit it is split into sections, and each section gets its own message:

1. Message 1 streams in via edits like any other message.
2. The moment the next section begins, message 1 receives its **final** edit
   and is **frozen** — it is never edited again.
3. Section 2 starts as a new message and streams in live the same way. A
   third section freezes the second, and so on.

Split points are chosen deterministically from the text already received
(prefering paragraph breaks, then line breaks), which is what makes freezing
safe: once a section's first 2000 characters have arrived, its boundary can
never move, so no re-edit of a frozen message is ever needed.

If an edit fails mid-flight (e.g. a transient rate limit), it is retried on
the next chunk — content is never lost or corrupted by the split.

## Tuning

| Variable | Description | Default |
|---|---|---|
| `STREAM_MIN_FLUSH_CHARS` | Minimum new characters before the next edit | `100` |
| `STREAM_MAX_FLUSH_INTERVAL` | Max seconds between edits (keeps slow streams feeling live) | `1.5` |

Lowering either makes replies appear faster on screen at the cost of more API
edits; raising them is gentler on rate limits but slightly lazier-looking.

## Observability

Every delivery step is logged (logger `bot.stream`, visible in `logs/dev.log`):

- `DELIVER stream part N STARTED (streaming via edits): msg_id=... len=...`
- `DELIVER stream part N FROZEN (no further edits): msg_id=... len=...`
- `DELIVER chunk N/M: msg_id=...` — for non-streamed long replies sent as static chunks

These lines show exactly where each split happened and let you verify that no
message was edited after its successor started.

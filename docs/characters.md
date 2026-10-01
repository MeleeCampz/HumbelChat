# Characters

Characters are defined in `characters.json` and control AI persona, model, system prompt, and optionally response length.

## File location

- `characters.json` is private and should not be committed.
- Use `characters.json.example` as a template.
- If the file is missing, the bot will warn and fall back to defaults.

## Format

```json
{
  "default": "System",
  "characters": {
    "System": {
      "model": "qwen3:latest",
      "system_prompt": ""
    },
    "Assistant": {
      "display": "Chat Assistant",
      "model": "gemma4:latest",
      "system_prompt": "..."
    }
  }
}
```

## Fields

| Field | Description |
|---|---|
| `default` | Character used when none is selected |
| `characters.<name>` | Each key becomes an available persona |
| `display` | Human-readable name shown in `/character show` |
| `model` | Model slug used for the inference API |
| `system_prompt` | Custom system prompt for the character |
| `temperature` | Optional sampling temperature (e.g. `0.7`) sent with AI requests |
| `max_tokens` | Optional per-character max tokens; when set, it is sent with AI requests for that character |

## Per-character max_tokens

If a character sets `max_tokens`, that value is used for AI requests using
that character. Otherwise the parameter is **omitted entirely**, so the
backend (LM Studio) lets the model use its maximum output length — there is
no global default or hard cap anymore (`MAX_TOKENS` / `MAX_TOKENS_HARD_CAP`
in `.env` are no longer applied).

Note: a set budget covers internal reasoning **and** the visible answer for
*thinking* models (e.g. Qwen3). If a response is truncated
(`finish_reason "length"`) with an empty answer, the bot reports a friendly
error — with model-max output there is no bigger budget to retry with.

Example character with a custom limit:

```json
{
  "default": "Assistant",
  "characters": {
    "Assistant": {
      "display": "Chat Assistant",
      "model": "gemma4:latest",
      "system_prompt": "You are a helpful assistant.",
      "max_tokens": 1500
    }
  }
}
```

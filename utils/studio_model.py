"""Shared Studio model-slug helper.

Unsloth Studio's ``/api/inference/load`` accepts a plain HuggingFace repo id
plus an optional ``gguf_variant`` quant slug — it does NOT accept the bot's
``"repo:quant"`` model slug (``MODEL_NAME``). Sending the slug verbatim fails
100%: HF repo-id validation rejects the colon (deferred 500 "Repo id must use
alphanumeric chars…", observed live 2026-10-29 — the #73 bug).

Both model-swap consumers (``bot_core.images._warm_chat_reload`` and
``bot_core.tts._ensure_slug_loaded``) must split the slug the same way, so
the rule lives here. ``bot_core.tts`` uses these helpers directly;
``bot_core.images`` carries an inline equivalent that is pinned by its own
live-verified test suite (#73) and stays untouched.
"""
from __future__ import annotations


def split_repo_variant(slug: str) -> tuple[str, str]:
    """Split a ``"repo:quant"`` model slug into ``(repo, quant)``.

    Splits at the FIRST colon (quant slugs may contain '-'/'_'/'.' but not
    ':'); either side may be empty — an empty quant means "backend default".

    >>> split_repo_variant("unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M")
    ('unsloth/Qwen3.8-27B-GGUF', 'Qwen3.8-27B-NVFP4_50-Q4K_M')
    >>> split_repo_variant("unsloth/Qwen-Image-2.1-Turbo-GGUF")
    ('unsloth/Qwen-Image-2.1-Turbo-GGUF', '')
    """
    repo, _, variant = (slug or "").partition(":")
    return repo.strip(), variant.strip()


def load_payload(slug: str, extra: dict[str, object] | None = None) -> dict[str, object]:
    """Build a Studio ``/api/inference/load`` JSON body for *slug*.

    ``{"model_path": repo[, "gguf_variant": quant]}`` (+ any *extra* keys such
    as ``gpu_ids``). *extra* is merged, never mutated.
    """
    repo, variant = split_repo_variant(slug)
    payload: dict[str, object] = {"model_path": repo}
    if variant:
        payload["gguf_variant"] = variant
    if extra:
        payload.update(extra)
    return payload


def is_resident(model: dict[str, object], slug: str) -> bool:
    """True when a ``loaded-models`` entry *model* matches *slug*.

    Studio reports loaded models with the bare ``owner/repo`` id (no quant in
    the id) and a separate ``quant`` field — so match the repo id, and verify
    the quant only when the slug requests one (a resident entry with no
    reported quant counts as a match).
    """
    repo, variant = split_repo_variant(slug)
    if not repo or model.get("id") != repo:
        return False
    if variant and model.get("quant") not in (variant, None):
        return False
    return bool(model.get("loaded"))

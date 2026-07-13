import asyncio
from config.characters import CHARACTER_CHOICES, get_character, default_character

async def handle_ai_command(
    interaction,
    message: str,
    character_name: str | None = None,
) -> None:
    """Core handler for /ai."""

    # 1. Immediate Deferral to prevent "Application did not respond"
    try:
        if not interaction.response.is_done():
            await interaction.response.defer()
    except Exception as e:
        print(f"Error deferring: {e}")
        return

    # 2. Resolve character
    char_key = None
    if character_name:
        for choice in CHARACTER_CHOICES:
            if choice["value"] == character_name:
                char_key = character_name
                break
    
    if char_key is None:
        # Try to use the helper from main.py if available, otherwise fallback
        try:
            from main import _get_active_character_key
            gid = getattr(interaction, 'guild_id', None)
            cid = getattr(interaction, 'channel_id', 0)
            char_key = _get_active_character_key(gid, cid)
        except (ImportError, AttributeError):
            char_key = default_character().key

    char_obj = get_character(char_key)
    if char_obj is None:
        await interaction.followup.send(f"Character `{character_name}` not found.", ephemeral=True)
        return

    model_slug = char_obj.model or ""

    # 3. Start typing indicator (background task)
    if hasattr(interaction, "channel") and interaction.channel is not None:
        try:
            from utils.typing_loop import typing_loop_task
            asyncio.create_task(typing_loop_task(interaction.channel))
        except Exception as e:
            print(f"Typing loop error: {e}")

    # 4. Ask AI
    from bot_core import ask_ai
    try:
        reply_text, _extra = await ask_ai(
            user_message=message,
            model_slug=model_slug,
            guild_id=interaction.guild_id or 0,
            channel_id=interaction.channel_id,
            username=(getattr(interaction, "user", None)
                          and getattr(getattr(interaction, "user"), "display_name", "")
                      ) or "",
        )
    except Exception as e:
        import logging
        logging.error(f"AI request failed: {e}")
        await interaction.followup.send(f"❌ Error calling AI: {e}", ephemeral=True)
        return

    # 5. Send chunked response
    from utils.response_splitter import send_long_response
    try:
        await send_long_response(interaction, reply_text, str(char_obj.display))
    except Exception as e:
        import logging
        logging.error(f"Failed to send AI response: {e}")

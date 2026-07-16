"""Shared helper classes and functions for discord-ai-bot tests."""
from __future__ import annotations


class Followup:
    """A real followup object with an async send method.

    In production slash commands you call ``interaction.followup.send()``
    after deferring the response (or immediately if you used
    ``interaction.response.send_message()``).  This mock always records
    what was sent so assertions can be made against it.
    """
    def __init__(self, sent_list):
        self._sent = sent_list

    async def send(self, content="", ephemeral=False, **kwargs):
        self._sent.append(str(content))


class Response:
    """A real response object with defer and send_message methods.

    In production slash commands you call ``interaction.response.send_message()``
    (direct response) or ``interaction.response.defer()`` followed by
    ``interaction.followup.send()`` (deferred response).  Both paths append
    to the shared _sent list so tests can verify output regardless of which
    approach the bot code uses.

    Usage::
        ix = Interaction()
        await ix.response.send_message("hello", ephemeral=True)
        # OR
        await ix.response.defer(ephemeral=True)
        await ix.followup.send("world")
    """
    def __init__(self, sent_list):
        self._sent = sent_list

    async def defer(self, **kwargs):
        """Discord.py: show typing indicator before sending follow-up."""
        pass

    async def send_message(self, content="", ephemeral=False, **kwargs):
        """Direct (non-deferred) response — immediately sends to Discord."""
        self._sent.append(str(content))


class Interaction:
    """Real mock interaction where followup.send and response methods are awaitable.

    The bot code does ``await interaction.followup.send(msg, ephemeral=True)`` and/or
    ``await interaction.response.send_message(msg, ephemeral=True)`` or
    ``await interaction.response.defer(ephemeral=True)``.  This class provides
    proper async methods for those operations instead of MagicMock coroutines.

    Crucially, the Response and Followup share the SAME _sent list so that regardless
    of which API path is exercised in production code, all output appears in ix._sent
    for test assertions.

    Usage::
        ix = Interaction(guild_id=123456, channel_id=789012)
        await ix.response.send_message("hello", ephemeral=True)
        assert "hello" in ix._sent
    """

    def __init__(self, **attrs):
        # Single shared list for both response and followup paths
        self._sent: list[str] = []
        self.response = Response(self._sent)
        self.followup = Followup(self._sent)
        for k, v in attrs.items():
            setattr(self, k, v)


def make_interaction(**attrs) -> Interaction:
    """Convenience function to create an Interaction."""
    return Interaction(**attrs)

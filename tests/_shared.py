"""Shared helper classes and functions for discord-ai-bot tests."""
from __future__ import annotations


class Followup:
    """A real followup object with an async send method."""
    def __init__(self, sent_list):
        self._sent = sent_list

    async def send(self, content="", ephemeral=False, **kwargs):
        self._sent.append(str(content))


class Response:
    """A real response object with an async defer method."""
    async def defer(self, **kwargs):
        pass


class Interaction:
    """Real mock interaction where followup.send and response.defer are awaitable.

    The bot code does ``await interaction.followup.send(msg, ephemeral=True)`` and
    ``await interaction.response.defer(ephemeral=True)``.  This class provides
    proper async methods for those operations instead of MagicMock coroutines.

    Usage::
        ix = Interaction(guild_id=123456, channel_id=789012)
        await ix.followup.send("hello", ephemeral=True)
        assert "hello" in ix._sent
    """

    def __init__(self, **attrs):
        self._sent: list[str] = []
        self.followup = Followup(self._sent)
        self.response = Response()
        for k, v in attrs.items():
            setattr(self, k, v)


def make_interaction(**attrs) -> Interaction:
    """Convenience function to create an Interaction."""
    return Interaction(**attrs)

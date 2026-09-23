"""P1 #16: SSRF / scheme guard + true (streamed) size caps for URL fetches.

Both ``/summarize <url>`` and ``/upload_kb <url>`` take a user-supplied URL.
Before the hardening they did ``client.get(url)`` + ``resp.text`` /
``resp.content`` — no scheme/host guard and the *entire* body buffered before
the length check.  This tests the shared guard in ``utils/url_fetch.py`` and
that both commands route through it.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import httpx

from utils import url_fetch
from utils.url_fetch import UnsafeUrlError, UrlTooLargeError


# ─────────────────────────── validate_url ───────────────────────────

class TestValidateUrl:
    @pytest.mark.parametrize("url", [
        "http://example.com/a",
        "https://example.com/path?q=1",
        "https://docs.example.org:8443/x",
    ])
    def test_http_https_allowed(self, url):
        assert url_fetch.validate_url(url) == url

    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "file:///home/user/secret.txt",
        "data:text/plain;base64,SGVsbG8=",
        "gopher://example.com/",
        "ftp://example.com/file",
        "javascript:alert(1)",
        "",
        "not a url",
    ])
    def test_disallowed_schemes_rejected(self, url):
        with pytest.raises(UnsafeUrlError):
            url_fetch.validate_url(url)

    @pytest.mark.parametrize("url", [
        "http://localhost/secret",
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",  # cloud instance metadata
        "http://[::1]/x",
        "http://0.0.0.0/",
    ])
    def test_internal_hosts_rejected(self, url):
        # loopback / link-local / reserved are ALWAYS blocked
        with pytest.raises(UnsafeUrlError):
            url_fetch.validate_url(url)

    def test_private_allowed_by_default(self):
        # A self-hosted bot may legitimately fetch LAN docs by default.
        assert url_fetch.validate_url("http://192.168.1.5/doc") == "http://192.168.1.5/doc"

    def test_private_blocked_when_opted_in(self):
        with pytest.raises(UnsafeUrlError):
            url_fetch.validate_url("http://192.168.1.5/doc", block_private=True)


# ─────────────────────────── fetch_url (streaming) ───────────────────────────

class _FakeStream:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, method, url, **kw):
        return _FakeStream(self._resp)


def _client_factory(resp):
    return lambda *a, **k: _FakeClient(resp)


class TestFetchUrlStreaming:
    @pytest.mark.asyncio
    async def test_returns_small_body(self, monkeypatch):
        resp = MagicMock()
        resp.url = httpx.URL("https://example.com/a")
        resp.raise_for_status = lambda: None
        async def aiter():
            yield b"hel"
            yield b"lo"
        resp.aiter_bytes = aiter
        monkeypatch.setattr(url_fetch.httpx, "AsyncClient", _client_factory(resp))
        out = await url_fetch.fetch_url("https://example.com/a", max_bytes=100)
        assert out == b"hello"

    @pytest.mark.asyncio
    async def test_oversized_aborts_mid_download(self, monkeypatch):
        """The cap must trip *while* streaming — not after buffering the body."""
        resp = MagicMock()
        resp.url = httpx.URL("https://example.com/big")
        resp.raise_for_status = lambda: None

        consumed = []

        async def aiter():
            # Three 6-byte chunks; max_bytes=10 -> must abort on the 2nd.
            for i in range(3):
                consumed.append(i)
                yield b"xy" * 3
        resp.aiter_bytes = aiter
        monkeypatch.setattr(url_fetch.httpx, "AsyncClient", _client_factory(resp))

        with pytest.raises(UrlTooLargeError):
            await url_fetch.fetch_url("https://example.com/big", max_bytes=10)

        # Aborted after the chunk that crossed the cap; never read chunk #3.
        assert consumed == [0, 1]

    @pytest.mark.asyncio
    async def test_unsafe_scheme_rejected_before_network(self, monkeypatch):
        """file:// must be refused with no network attempt at all."""
        called = []
        monkeypatch.setattr(
            url_fetch.httpx, "AsyncClient",
            lambda *a, **k: called.append(1) or _FakeClient(MagicMock()),
        )
        with pytest.raises(UnsafeUrlError):
            await url_fetch.fetch_url("file:///etc/passwd", max_bytes=100)
        assert not called  # client was never constructed

    @pytest.mark.asyncio
    async def test_redirect_to_internal_host_rejected(self, monkeypatch):
        """A redirect to a link-local host must be re-validated and blocked."""
        import httpx
        resp = MagicMock()
        resp.url = httpx.URL("http://169.254.169.254/latest/meta-data")
        resp.raise_for_status = lambda: None
        async def aiter():
            yield b"x"
        resp.aiter_bytes = aiter
        monkeypatch.setattr(url_fetch.httpx, "AsyncClient", _client_factory(resp))
        with pytest.raises(UnsafeUrlError):
            await url_fetch.fetch_url("https://ok.example.com/a", max_bytes=100)


# ─────────────────────────── command wiring ───────────────────────────

class TestCommandWiring:
    @pytest.mark.asyncio
    async def test_summarize_rejects_file_url(self, ix):
        from commands.utility_commands import handle_summarize_command
        with patch("utils.url_fetch.fetch_url", new=AsyncMock(
                side_effect=UnsafeUrlError("scheme file not allowed"))):
            await handle_summarize_command(ix, file_url="file:///etc/passwd")
        joined = " ".join(ix._sent)
        assert "URL not allowed" in joined
        assert "not allowed" in joined.lower()

    @pytest.mark.asyncio
    async def test_upload_kb_rejects_file_url(self, ix):
        from commands.kb_commands import handle_upload_kb
        with patch("utils.url_fetch.fetch_url", new=AsyncMock(
                side_effect=UnsafeUrlError("scheme file not allowed"))):
            await handle_upload_kb(ix, url="file:///etc/passwd")
        joined = " ".join(ix._sent)
        assert "URL not allowed" in joined

    @pytest.mark.asyncio
    async def test_upload_kb_rejects_oversized(self, ix):
        """A 25 MB stream must be rejected without holding 25 MB (code path)."""
        from commands.kb_commands import handle_upload_kb
        with patch("utils.url_fetch.fetch_url", new=AsyncMock(
                side_effect=UrlTooLargeError("Response is larger than 20 MB and was aborted before fully downloading."))):
            await handle_upload_kb(ix, url="https://example.com/big.bin")
        joined = " ".join(ix._sent)
        assert "too large" in joined.lower()

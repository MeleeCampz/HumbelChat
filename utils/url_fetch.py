"""Shared, hardened URL fetching for user-supplied URLs (P1 #16).

Two user-facing commands take a free-form URL and fetch it:

* ``/summarize <url>``   (commands/utility_commands.py)
* ``/upload_kb <url>``   (commands/kb_commands.py)

Before the hardening they did ``client.get(url)`` + ``resp.text``/``resp.content``:

1. **No scheme guard** — ``file:///etc/passwd``, ``data:``, ``gopher:``, etc. were
   handed straight to httpx (httpx rejects most non-http schemes, but a bare
   ``file://`` on some builds / a ``data:`` body is an information leak, and the
   bot never said *why* it refused).
2. **No size cap before full read** — the *entire* body was materialised into
   memory before a length check, so a 2 GB response was fully buffered before
   "too large" fired.

This module centralises both protections so the two call sites share one
implementation:

* :func:`validate_url` — rejects non-``http(s)`` schemes and (optionally)
  private / link-local / loopback / reserved hosts (SSRF hardening).
* :func:`fetch_url`    — streams the body, aborting mid-download once
  ``max_bytes`` is exceeded (so an oversized response is rejected *without*
  buffering the whole thing).
"""
from __future__ import annotations

import ipaddress
import logging
from urllib.parse import urlparse

import httpx

log = logging.getLogger("bot.utils.url_fetch")


class UrlFetchError(Exception):
    """Base class for all URL-fetch failures."""


class UnsafeUrlError(UrlFetchError):
    """The URL was rejected before any request (bad scheme / private host)."""


class UrlTooLargeError(UrlFetchError):
    """The response exceeded ``max_bytes`` and was aborted mid-download."""


# Only http/https are acceptable as user-supplied fetch targets.  file://,
# data:, gopher:, ftp:// etc. are all refused up front.
ALLOWED_SCHEMES = ("http", "https")

# Hostnames that are almost never a legitimate fetch target and are the classic
# SSRF / credential-theft sinks (cloud instance metadata, localhost).
_DANGEROUS_HOSTNAMES = {"localhost", "metadata", "ip6-localhost", "ip6-loopback"}


def _ip_disallowed(ip: ipaddress._BaseAddress) -> bool:
    """True for the always-blocked IP classes (loopback, link-local metadata,
    reserved, unspecified).  *Private* ranges (10/8, 172.16/12, 192.168/16) are
    handled separately so self-hosted users can still fetch their own LAN docs."""
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
    )


def _host_disallowed(host: str, *, block_private: bool) -> bool:
    """Return True if *host* is a disallowed fetch target.

    * Loopback / link-local / reserved / metadata hostnames and IPs are
      **always** disallowed (these are the genuine SSRF / credential sinks).
    * Broader *private* ranges are disallowed only when ``block_private`` is
      set (opt-in, since a self-hosted bot may legitimately fetch LAN docs).
    """
    if host.lower() in _DANGEROUS_HOSTNAMES:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # a normal DNS hostname
    if _ip_disallowed(ip):
        return True
    if block_private and (ip.is_private or ip.is_multicast):
        return True
    return False


def validate_url(url: str, *, block_private: bool = False) -> str:
    """Validate a user-supplied URL; raise :class:`UnsafeUrlError` if unsafe.

    Rejects non-http(s) schemes (``file://``, ``data:``, ``gopher:``, …) and
    disallowed hosts.  Returns the original URL when it passes.
    """
    if not url or not isinstance(url, str):
        raise UnsafeUrlError("No URL provided.")
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise UnsafeUrlError(f"Could not parse URL: {exc}") from exc

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(
            f"URL scheme {scheme or '(none)'} is not allowed (use http/https)."
        )

    host = parsed.hostname
    if not host:
        raise UnsafeUrlError("URL has no host.")
    if _host_disallowed(host, block_private=block_private):
        raise UnsafeUrlError(
            f"URL host {host!r} is not allowed (private/internal range)."
        )
    return url


def _revalidate_after_redirect(final_url: httpx.URL, *, block_private: bool) -> None:
    """Re-check the *final* URL after redirects so a redirect can't hop to a
    disallowed scheme/host (e.g. http://ok.example → http://169.254.169.254/)."""
    scheme = (final_url.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"Redirect to disallowed scheme {scheme!r}.")
    host = getattr(final_url, "host", None)  # httpx.URL exposes .host
    if host and _host_disallowed(host, block_private=block_private):
        raise UnsafeUrlError(f"Redirect to disallowed host {host!r}.")


async def fetch_url(
    url: str,
    *,
    max_bytes: int,
    timeout: float = 30.0,
    block_private: bool = False,
) -> bytes:
    """Stream-GET *url* into at most ``max_bytes``; abort if it would exceed.

    Raises :class:`UnsafeUrlError` for a bad scheme/host (before or after
    redirects), :class:`UrlTooLargeError` when the body is too big, and the
    underlying :class:`httpx.HTTPError` / ``TimeoutError`` on transport failure.
    """
    validate_url(url, block_private=block_private)
    if max_bytes <= 0:
        max_bytes = 10 * 1024 * 1024  # sane default if caller passes <= 0

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url, follow_redirects=True) as resp:
            # A redirect may land on a disallowed target — re-validate before
            # we start reading the body.
            _revalidate_after_redirect(resp.url, block_private=block_private)
            resp.raise_for_status()
            buf = bytearray()
            async for chunk in resp.aiter_bytes():
                if len(buf) + len(chunk) > max_bytes:
                    raise UrlTooLargeError(
                        f"Response is larger than {max_bytes // (1024 * 1024)} MB "
                        "and was aborted before fully downloading."
                    )
                buf.extend(chunk)
    return bytes(buf)


__all__ = [
    "UrlFetchError",
    "UnsafeUrlError",
    "UrlTooLargeError",
    "ALLOWED_SCHEMES",
    "validate_url",
    "fetch_url",
]

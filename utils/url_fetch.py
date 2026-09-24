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
import socket
from urllib.parse import urljoin, urlparse

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


def _address_disallowed(ip: ipaddress._BaseAddress, *, block_private: bool) -> bool:
    if _ip_disallowed(ip):
        return True
    if block_private and (ip.is_private or ip.is_multicast):
        return True
    return False


def _checked_connect_ip(host: str, port: int, *, block_private: bool) -> str:
    """Resolve *host* and return one IP that is safe to connect to.

    The hostname string is not enough: a public name can resolve to a
    loopback, link-local, or private address. Every address is checked, and
    the caller connects to the returned IP so a later DNS change cannot
    redirect the request.
    """
    if _host_disallowed(host, block_private=block_private):
        raise UnsafeUrlError(f"URL host {host!r} is not allowed (private/internal range).")
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"Could not resolve host {host!r}.") from exc
    allowed: list[str] = []
    for info in infos:
        raw = info[4][0].split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            raise UnsafeUrlError(f"Could not resolve host {host!r}.")
        if _address_disallowed(ip, block_private=block_private):
            raise UnsafeUrlError(
                f"URL host {host!r} resolves to a disallowed address."
            )
        allowed.append(str(ip))
    if not allowed:
        raise UnsafeUrlError(f"Could not resolve host {host!r}.")
    return allowed[0]


def _port_for(parsed) -> int:
    if parsed.port:
        return parsed.port
    return 443 if (parsed.scheme or "").lower() == "https" else 80


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


def _pinned_request(url: str, *, block_private: bool) -> tuple[str, dict, dict]:
    """Return ``(ip_url, headers, extensions)`` for a checked connection."""
    validate_url(url, block_private=block_private)
    parsed = urlparse(url)
    host = parsed.hostname or ""
    ip = _checked_connect_ip(host, _port_for(parsed), block_private=block_private)
    pinned = str(httpx.URL(url).copy_with(host=ip))
    headers = {"Host": host}
    extensions = {"sni_hostname": host.encode("ascii", "ignore")}
    return pinned, headers, extensions


async def fetch_url(
    url: str,
    *,
    max_bytes: int,
    timeout: float = 30.0,
    block_private: bool = True,
) -> bytes:
    """Stream-GET *url* into at most ``max_bytes``; abort if it would exceed.

    Private and internal addresses are refused by default, including names
    that only look public until they are resolved. Each redirect hop is
    checked the same way before a connection is opened.

    Raises :class:`UnsafeUrlError` for a bad scheme/host (before or after
    redirects), :class:`UrlTooLargeError` when the body is too big, and the
    underlying :class:`httpx.HTTPError` / ``TimeoutError`` on transport failure.
    """
    if max_bytes <= 0:
        max_bytes = 10 * 1024 * 1024  # sane default if caller passes <= 0

    current = validate_url(url, block_private=block_private)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for _hop in range(5):
            pinned, headers, extensions = _pinned_request(current, block_private=block_private)
            async with client.stream(
                "GET", pinned, headers=headers, extensions=extensions, follow_redirects=False,
            ) as resp:
                if resp.status_code in {301, 302, 303, 307, 308}:
                    location = resp.headers.get("location")
                    if not location:
                        raise UnsafeUrlError("Redirect had no Location header.")
                    current = urljoin(current, location)
                    validate_url(current, block_private=block_private)
                    continue
                # A redirect the client already followed (or a test double that
                # reports a different final URL) is checked before the body.
                _revalidate_after_redirect(resp.url, block_private=block_private)
                final_host = getattr(resp.url, "host", None)
                if final_host and final_host != urlparse(current).hostname:
                    _checked_connect_ip(
                        final_host, _port_for(urlparse(str(resp.url))),
                        block_private=block_private,
                    )
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
    raise UnsafeUrlError("Too many redirects.")


__all__ = [
    "UrlFetchError",
    "UnsafeUrlError",
    "UrlTooLargeError",
    "ALLOWED_SCHEMES",
    "validate_url",
    "fetch_url",
]

"""Client-side HTTP fetch (web search itself is Anthropic's server-side tool)."""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

from .registry import Risk, Tool, ToolContext, ToolError, schema

MAX_BYTES = 3_000_000


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        return re.sub(r"\n\s*\n+", "\n\n", raw).strip()


def _reject_private_host(host: str) -> None:
    """Block requests to localhost / private networks (SSRF guard)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ToolError(f"Could not resolve host '{host}'.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ToolError(f"Refusing to fetch '{host}': it resolves to a private/internal address.")


def _fetch_url(args: dict, ctx: ToolContext) -> str:
    url = args["url"].strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolError("Only absolute http(s) URLs are supported.")
    _reject_private_host(parsed.hostname)
    req = urllib.request.Request(url, headers={"User-Agent": "sentinel-agent/0.1 (+autonomous research agent)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            ctype = resp.headers.get("Content-Type", "")
            body = resp.read(MAX_BYTES)
            status = resp.status
    except urllib.error.HTTPError as exc:
        raise ToolError(f"HTTP {exc.code} {exc.reason} for {url}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ToolError(f"Request failed: {exc}") from exc

    charset = "utf-8"
    match = re.search(r"charset=([\w-]+)", ctype)
    if match:
        charset = match.group(1)
    text = body.decode(charset, errors="replace")
    if "html" in ctype:
        parser = _TextExtractor()
        parser.feed(text)
        text = parser.text()
    return f"[HTTP {status}, {ctype or 'unknown type'}]\n{text}"


TOOLS = [
    Tool(
        "fetch_url",
        "HTTP GET a public URL. HTML is converted to plain text; JSON and text are returned as-is. "
        "Use for reading documentation pages or calling public JSON APIs.",
        schema({"url": {"type": "string"}}, ["url"]),
        _fetch_url, Risk.NETWORK,
    ),
]

# Anthropic-hosted web search: runs on Anthropic's servers, results come back in the same response.
WEB_SEARCH_SERVER_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 8}

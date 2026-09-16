"""P18-S1: render assistant Markdown into safe, whitelisted HTML.

The single transformation point between a model's text and the browser.
``html.escape`` runs FIRST over the whole input, so no character the model
emits can ever become markup or script. The later passes introduce only a
fixed whitelist of tags -- ``p br strong em ul ol li hr code a`` -- with
no ``style``/``class``/event attributes and links restricted to ``http(s)``
schemes. Anything the renderer does not understand stays visible as escaped
text; it is never interpreted as HTML.

Deliberately small; extends only when a concrete presenter needs it:
- paragraphs (blank-line separated)
- single line breaks inside a paragraph -> ``<br>``
- ``**bold**`` and ``*italic*``
- ``- `` / ``* `` unordered and ``1. `` ordered lists
- ``---`` separator -> ``<hr>``
- ``[label](http(s) url)`` links (other schemes render as literal text)
- `` `inline code` ``

The UI policy mirroring this contract: assistant messages may use
``innerHTML`` ONLY with this module's output; every other channel stays
``textContent``.
"""

from __future__ import annotations

import html
import re
import urllib.parse

_ALLOWED_URL_SCHEMES = ("http", "https")


def render_assistant_markdown(text: str) -> str:
    """Render one assistant reply as whitelisted HTML (never raw input)."""
    if not text:
        return ""
    escaped = html.escape(text, quote=True)
    blocks = re.split(r"\n\s*\n+", escaped.strip()) if escaped.strip() else []
    return "".join(_render_block(block.strip()) for block in blocks if block.strip() != "")


def _render_block(block: str) -> str:
    if re.fullmatch(r"-{3,}", block) or re.fullmatch(r"\*{3,}", block):
        return "<hr>"
    if block.startswith("- ") or block.startswith("* "):
        items = re.split(r"\n(?=[*] )|\n(?=- )", block)
        lines = [line[2:].strip() if line.startswith("- ") or line.startswith("* ") else line.strip() for line in items]
        lines = [line for line in lines if line != ""]
        if lines:
            return "<ul>" + "".join(f"<li>{_render_inline(line)}</li>" for line in lines) + "</ul>"
    ordered = re.match(r"^\d+\.\s", block)
    if ordered:
        items = re.split(r"\n(?=\d+\.\s)", block)
        lines = [
            re.sub(r"^\d+\.\s", "", line).strip()
            for line in items
        ]
        lines = [line for line in lines if line != ""]
        if lines:
            return "<ol>" + "".join(f"<li>{_render_inline(line)}</li>" for line in lines) + "</ol>"
    return "<p>" + "<br>".join(_render_inline(line) for line in block.split("\n")) + "</p>"


def _render_inline(text: str) -> str:
    # Inline code first and protect it from the label/emphasis passes.
    code_spans: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        code_spans.append(match.group(1))
        return f"{len(code_spans) - 1}"

    staged = re.sub(r"`([^`]+)`", _stash, text)
    staged = _render_links(staged)
    staged = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", staged)
    staged = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", staged)

    def _restore(match: re.Match[str]) -> str:
        return "<code>" + code_spans[int(match.group(1))] + "</code>"

    return re.sub(r"(\d+)", _restore, staged)


def _render_links(text: str) -> str:
    """Markdown links to http(s) targets only; any other URL renders literal."""

    def _replace(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        parsed = urllib.parse.urlparse(html.unescape(url))
        if parsed.scheme not in _ALLOWED_URL_SCHEMES or not parsed.netloc:
            return match.group(0)
        href = html.escape(url, quote=True)
        return (
            f'<a href="{href}" target="_blank" rel="noopener noreferrer">'
            f"{label}</a>"
        )

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _replace, text)
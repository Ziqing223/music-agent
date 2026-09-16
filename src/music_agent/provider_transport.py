"""P10.8: Shared stdlib HTTP transport for the HTTP-based providers.

Both the DeepSeek (OpenAI-compatible) and the Anthropic-format adapter use the same
injected transport boundary and the same typed error mapping, so failures behave
identically regardless of provider. Tests inject a deterministic fake; production uses
stdlib urllib with no new dependency.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Protocol, runtime_checkable

from music_agent.provider_contract import ProviderTimeoutError, ProviderUnavailableError


@runtime_checkable
class HttpTransport(Protocol):
    """One synchronous JSON POST; returns (status, body_text)."""

    def post_json(
        self, url: str, headers: Mapping[str, str], body: Mapping[str, Any], timeout: float
    ) -> tuple[int, str]: ...


class UrllibHttpTransport:
    """Production transport over stdlib urllib (no new dependency)."""

    def post_json(
        self, url: str, headers: Mapping[str, str], body: Mapping[str, Any], timeout: float
    ) -> tuple[int, str]:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            return error.code, raw
        except TimeoutError as error:
            raise ProviderTimeoutError(f"provider request timed out after {timeout}s") from error
        except OSError as error:
            raise ProviderUnavailableError(str(error)) from error


def guarded_post(
    transport: HttpTransport,
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout: float,
) -> tuple[int, str]:
    """Run one transport POST and map transport-level failures to typed provider errors.

    The adapter-level mapping is duplicated here deliberately: any transport
    implementation (including test fakes that raise) gets the same typed behavior.
    """
    try:
        return transport.post_json(url, headers, body, timeout)
    except TimeoutError as error:
        raise ProviderTimeoutError(f"provider request timed out after {timeout}s") from error
    except OSError as error:
        raise ProviderUnavailableError(str(error)) from error

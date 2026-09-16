"""P10.8: Codex CLI provider -- ChatGPT-authenticated Codex as the language layer only.

First-class provider per the P10.8 amendment: the user's existing Codex CLI environment
(``codex exec --json``, authenticated by ``~/.codex/auth.json``) supplies model
intelligence; NO ``OPENAI_API_KEY`` is required. Codex is strictly the language layer:

- The subprocess receives ONLY our rendered prompt on stdin -- the system prompt, the
  P09 tool descriptions, and the conversation. No shell, filesystem, or tool access is
  ever granted to Codex, and only its stdout text is consumed: a model cannot reach
  outside this boundary no matter what it emits.
- Tool requests ride a deterministic text envelope inside the response: a JSON object
  ``{"tool_calls": [{"id", "name", "arguments"}]}``. The adapter parses the REAL
  ``codex exec --json`` event stream (JSONL: ``thread.started`` / ``item.completed``
  with ``item.type == "agent_message"`` / ``turn.completed`` with usage) and takes the
  LAST ``agent_message`` text; ANY other output (plain text, garbled JSON, legacy
  single-object wrappers) is treated as the final answer text -- fail soft, never
  guessed tool calls.
- Tool RESULTS are fed back as rendered text in the next prompt; the P09 execution
  boundary stays untouched (the loop owns every execution).
- Failures map to the shared typed vocabulary: nonzero exit with auth symptoms ->
  ProviderAuthError; timeout -> ProviderTimeoutError; anything else -> unavailable.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from music_agent.provider_contract import (
    ProviderAuthError,
    ProviderError,
    ProviderMessage,
    ProviderMessageRole,
    ProviderResponse,
    ProviderStopReason,
    ProviderTimeoutError,
    ProviderToolCall,
    ProviderToolSchema,
    ProviderUnavailableError,
)

DEFAULT_CODEX_COMMAND = ("codex", "exec", "--json")


class CodexCliError(ProviderError):
    code = "codex_cli_error"


@runtime_checkable
class CodexCommandRunner(Protocol):
    """Injected boundary: one Codex CLI invocation with a prompt on stdin."""

    def run(
        self, command: list[str], prompt: str, timeout: float
    ) -> tuple[int, str, str]:  # (returncode, stdout, stderr)
        ...


class SubprocessCodexRunner:
    """Production runner: subprocess with the prompt on stdin (never a shell)."""

    def run(self, command: list[str], prompt: str, timeout: float) -> tuple[int, str, str]:
        try:
            completed = subprocess.run(
                list(command),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ProviderTimeoutError(f"codex command timed out after {timeout}s") from error
        except OSError as error:
            raise ProviderUnavailableError(str(error)) from error
        return completed.returncode, completed.stdout or "", completed.stderr or ""


@dataclass(frozen=True, slots=True)
class CodexCliConfig:
    """Codex CLI provider configuration (no secrets: auth lives in the CLI environment)."""

    command: tuple[str, ...] = DEFAULT_CODEX_COMMAND
    model: str | None = None
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not isinstance(self.command, tuple) or not self.command or not all(
            isinstance(part, str) and part for part in self.command
        ):
            raise CodexCliError("command must be a non-empty tuple of non-empty strings")
        if self.model is not None and (not isinstance(self.model, str) or not self.model):
            raise CodexCliError("model must be a non-empty string or None")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise CodexCliError("timeout_seconds must be positive")


class CodexCliProvider:
    supports_turn_interpreter = True

    """ChatProvider over the Codex CLI (language layer only, see module docstring)."""

    def __init__(self, config: CodexCliConfig | None = None, runner: CodexCommandRunner | None = None) -> None:
        if config is not None and not isinstance(config, CodexCliConfig):
            raise CodexCliError("config must be a CodexCliConfig")
        if runner is not None and not isinstance(runner, CodexCommandRunner):
            raise CodexCliError("runner must implement CodexCommandRunner")
        self.config = config or CodexCliConfig()
        self._runner = runner or SubprocessCodexRunner()

    def chat(
        self,
        system: str,
        messages: Sequence[ProviderMessage],
        tools: Sequence[ProviderToolSchema],
    ) -> ProviderResponse:
        prompt = self._render_prompt(system, messages, tools)
        command = list(self.config.command)
        if self.config.model:
            command.extend(["--model", self.config.model])
        try:
            returncode, stdout, stderr = self._runner.run(
                command, prompt, self.config.timeout_seconds
            )
        except TimeoutError as error:
            raise ProviderTimeoutError(
                f"codex command timed out after {self.config.timeout_seconds}s"
            ) from error
        except OSError as error:
            raise ProviderUnavailableError(str(error)) from error
        if returncode != 0:
            if _looks_like_auth_failure(stderr):
                raise ProviderAuthError(
                    f"codex CLI authentication failed: {stderr.strip()[:200]}"
                )
            raise ProviderUnavailableError(
                f"codex CLI exited {returncode}: {stderr.strip()[:200]}"
            )
        text, usage = _parse_codex_json_output(stdout)
        if text is None:
            raise ProviderInvalidResponseError("codex output carried no message text")
        return _parse_text_response(text, usage)

    def _render_prompt(
        self,
        system: str,
        messages: Sequence[ProviderMessage],
        tools: Sequence[ProviderToolSchema],
    ) -> str:
        parts = [system, ""]
        if tools:
            parts.append(
                "你可以使用以下工具（通过 Music Agent 的共享服务执行，工具结果会以文字回传）："
            )
            for tool in tools:
                parts.append(
                    f"- {tool.name}: {tool.description} "
                    f"参数 JSON Schema: {json.dumps(dict(tool.input_schema), ensure_ascii=False)}"
                )
            parts.append(
                "当需要调用工具时，只输出一个 JSON 对象，格式为 "
                '{"tool_calls": [{"id": "<唯一ID>", "name": "<工具名>", "arguments": {<参数>}}]}，'
                "不要输出任何其他文字。否则直接输出给用户的最终回答。"
            )
        else:
            # S1: a zero-tool turn (plain chat) carries no tool block -- the
            # codex prompt asks for the final answer directly instead of
            # dangling a tool instruction with nothing to call.
            parts.append("本轮不提供任何工具：请直接输出给用户的最终回答。")
        parts.append("")
        parts.append("--- 对话 ---")
        for message in messages:
            parts.append(_render_message(message))
        parts.append("")
        parts.append("请按上述规则回答：")
        return "\n".join(parts)


def _render_message(message: ProviderMessage) -> str:
    if message.tool_results is not None:
        return "\n".join(
            f"[工具结果 {result.call_id}] {result.content}"
            for result in message.tool_results
        )
    if message.tool_calls is not None:
        envelope = {
            "tool_calls": [
                {"id": call.call_id, "name": call.name, "arguments": json.loads(call.arguments or "{}")}
                for call in message.tool_calls
            ]
        }
        rendered = f"[助手] {json.dumps(envelope, ensure_ascii=False)}"
        if message.text:
            # A preamble alongside tool calls is part of the turn: keep it visible.
            rendered = f"[助手] {message.text}\n{rendered}"
        return rendered
    assert message.text is not None
    label = "用户" if message.role is ProviderMessageRole.USER else "助手"
    return f"[{label}] {message.text}"


def _parse_codex_json_output(stdout: str) -> tuple[str | None, dict[str, int]]:
    """Parse a ``codex exec --json`` stdout stream into (answer text, usage).

    The real CLI emits newline-delimited JSON events (``thread.started``,
    ``turn.started``, ``item.completed`` with ``item.type == "agent_message"``,
    ``turn.completed`` with usage). The assistant's message -- possibly containing the
    tool-call envelope -- is the LAST ``agent_message`` item text. Native Codex tool
    items (``function_call`` / ``function_call_output``) are deliberately ignored:
    this adapter never executes anything itself, so whatever Codex emits, only the
    message text reaches the Music Agent loop and the P09 boundary.

    Non-event output (plain text, or a single JSON object such as older CLI versions
    emitted) falls back to the whole stdout as the text, preserving the previous
    behavior for wrapped/legacy invocations.
    """
    texts: list[str] = []
    usage: dict[str, int] = {}
    saw_event = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type is None:
            continue
        saw_event = True
        if event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    texts.append(text)
        elif event_type == "turn.completed":
            raw_usage = event.get("usage")
            if isinstance(raw_usage, dict):
                for key, value in raw_usage.items():
                    if isinstance(value, int):
                        usage[str(key)] = value
    if not saw_event:
        # Legacy/plain output: keep the previous single-document handling.
        return _extract_legacy_text(stdout), {}
    if texts:
        return texts[-1], usage
    return None, usage


def _extract_legacy_text(stdout: str) -> str | None:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout.strip() or None
    if isinstance(payload, dict):
        for key in ("result", "output", "text", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return stdout.strip() or None


def _parse_text_response(text: str, usage: Mapping[str, int] | None = None) -> ProviderResponse:
    """Parse the envelope from the model's text; anything else is the final answer."""
    usage = dict(usage or {})
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return _text_response(stripped, usage)
    if not isinstance(payload, dict):
        return _text_response(stripped, usage)
    raw_calls = payload.get("tool_calls")
    if raw_calls is None:
        return _text_response(stripped, usage)
    if not isinstance(raw_calls, list):
        return _text_response(stripped, usage)
    calls: list[ProviderToolCall] = []
    for raw in raw_calls:
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("name"), str)
            or not isinstance(raw.get("id"), str)
            or not isinstance(raw.get("arguments"), dict)
        ):
            return _text_response(stripped, usage)  # garbled envelope: fail soft to text
        calls.append(
            ProviderToolCall(
                call_id=raw["id"],
                name=raw["name"],
                arguments=json.dumps(raw["arguments"], ensure_ascii=False),
            )
        )
    if not calls:
        return _text_response(stripped, usage)
    return ProviderResponse(
        message=ProviderMessage(ProviderMessageRole.ASSISTANT, tool_calls=tuple(calls)),
        stop_reason=ProviderStopReason.TOOL_USE,
        usage=usage,
    )


def _text_response(text: str, usage: Mapping[str, int] | None = None) -> ProviderResponse:
    return ProviderResponse(
        message=ProviderMessage(ProviderMessageRole.ASSISTANT, text=text),
        stop_reason=ProviderStopReason.END_TURN,
        usage=dict(usage or {}),
    )


def _looks_like_auth_failure(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        token in lowered
        for token in ("auth", "login", "api key", "401", "unauthorized", "sign in")
    )

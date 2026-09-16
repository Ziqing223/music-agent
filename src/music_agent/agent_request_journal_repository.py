"""P09.4: durable agent request journal -- the replay-safety backbone of the shared service.

One append-only row per executed or refused agent request, keyed by the opaque ``req_`` request
identity. The row stores the canonical deterministic payload text and the full outcome envelope
(payload/error mirror columns), so the service can answer a replayed request id from durable
state without re-executing, and can refuse a mismatched replay fail-closed. No domain user state
lives in this table; it is provenance and replay protection for the agent boundary only.

Safety properties:

* append-only -- no update/delete path exists in this repository, and SQL triggers make any
  out-of-band update/delete fail at the storage layer;
* duplicate request ids fail closed before write (``DuplicateAgentRequestError``), including
  re-verification under the same ``BEGIN IMMEDIATE`` transaction;
* request/result identity coherence -- a recorded result must belong to the recorded request
  (same ``request_id`` and tool name), else the write fails closed;
* readback verifies mirror columns against the decoded canonical texts
  (``CorruptAgentRequestJournalError``), so tampered or damaged rows never decode silently.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from music_agent.agent_contract import (
    AgentRequest,
    AgentToolResult,
    decode_agent_request,
    decode_agent_tool_result,
    encode_agent_payload,
    encode_agent_request,
    encode_agent_tool_result,
    validate_request_id,
)
from music_agent.repository import _open_store_connection


class AgentRequestJournalRepositoryError(ValueError):
    code = "agent_request_journal_repository_error"


class DuplicateAgentRequestError(AgentRequestJournalRepositoryError):
    code = "duplicate_agent_request"


class CorruptAgentRequestJournalError(AgentRequestJournalRepositoryError):
    code = "corrupt_agent_request_journal"


@dataclass(frozen=True, slots=True)
class AgentRequestRecord:
    """One journaled request together with the exact outcome envelope it produced."""

    request: AgentRequest
    result: AgentToolResult


class AgentRequestJournalRepository:
    """Append-only durable journal for executed/refused agent requests."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> AgentRequestJournalRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def record(self, request: AgentRequest, result: AgentToolResult) -> None:
        """Journal one request together with its outcome envelope.

        Fails closed on a non-request/result, on request/result identity mismatch, and on a
        reused request id (pre-checked and re-checked inside the write transaction).
        """
        if not isinstance(request, AgentRequest):
            raise AgentRequestJournalRepositoryError("request must be an AgentRequest")
        if not isinstance(result, AgentToolResult):
            raise AgentRequestJournalRepositoryError("result must be an AgentToolResult")
        if result.request_id != request.request_id:
            raise AgentRequestJournalRepositoryError(
                "result must belong to the same request_id as its request"
            )
        if result.tool != request.tool:
            raise AgentRequestJournalRepositoryError(
                "result must belong to the same tool as its request"
            )
        if self._existing(request.request_id):
            raise DuplicateAgentRequestError(
                f"request {request.request_id} is already journaled"
            )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            if self._existing(request.request_id):
                raise DuplicateAgentRequestError(
                    f"request {request.request_id} is already journaled"
                )
            self._connection.execute(
                """INSERT INTO agent_requests(
                    request_id, client_id, model_id, tool_name, contract_version,
                    request_text, payload_text, outcome, result_text, error_code,
                    error_message, issued_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request.request_id,
                    request.client.client_id,
                    request.client.model_id,
                    request.tool,
                    request.contract_version,
                    encode_agent_request(request),
                    encode_agent_payload(request.payload),
                    result.outcome.value,
                    encode_agent_tool_result(result),
                    result.error_code,
                    result.error_message,
                    request.issued_at.isoformat(),
                    result.completed_at.isoformat(),
                ),
            )
        except sqlite3.IntegrityError as error:
            self._connection.execute("ROLLBACK")
            raise AgentRequestJournalRepositoryError(
                f"agent request row violates the journal schema: {error}"
            ) from error
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")

    def get(self, request_id: str) -> AgentRequestRecord | None:
        """Load one journaled request by its request id; ``None`` when absent."""
        validate_request_id(request_id)
        row = self._connection.execute(
            "SELECT * FROM agent_requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def list(self) -> tuple[AgentRequestRecord, ...]:
        """Load every journaled request in chronological issuance order.

        Ordered by timezone-aware ``issued_at`` ascending, ``request_id`` ascending tie-break.
        Any row that fails canonical decoding or whose mirror columns disagree fails the whole
        call closed.
        """
        rows = self._connection.execute(
            "SELECT * FROM agent_requests ORDER BY issued_at ASC, request_id ASC"
        ).fetchall()
        return tuple(self._record_from_row(row) for row in rows)

    # --- internals ----------------------------------------------------------

    def _existing(self, request_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM agent_requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        return row is not None

    def _record_from_row(self, row: sqlite3.Row) -> AgentRequestRecord:
        try:
            request = decode_agent_request(row["request_text"])
            result = decode_agent_tool_result(row["result_text"])
        except (TypeError, ValueError) as error:
            raise CorruptAgentRequestJournalError(
                f"agent request row {row['request_id']} is not canonical: {error}"
            ) from error
        if row["client_id"] != request.client.client_id:
            raise CorruptAgentRequestJournalError(
                f"agent request row {row['request_id']} client_id mirror disagrees"
            )
        if row["model_id"] != request.client.model_id:
            raise CorruptAgentRequestJournalError(
                f"agent request row {row['request_id']} model_id mirror disagrees"
            )
        if row["tool_name"] != request.tool:
            raise CorruptAgentRequestJournalError(
                f"agent request row {row['request_id']} tool_name mirror disagrees"
            )
        if row["contract_version"] != request.contract_version:
            raise CorruptAgentRequestJournalError(
                f"agent request row {row['request_id']} contract_version mirror disagrees"
            )
        if row["issued_at"] != request.issued_at.isoformat():
            raise CorruptAgentRequestJournalError(
                f"agent request row {row['request_id']} issued_at mirror disagrees"
            )
        if row["payload_text"] != encode_agent_payload(request.payload):
            raise CorruptAgentRequestJournalError(
                f"agent request row {row['request_id']} payload_text mirror disagrees"
            )
        if row["outcome"] != result.outcome.value:
            raise CorruptAgentRequestJournalError(
                f"agent request row {row['request_id']} outcome mirror disagrees"
            )
        if row["completed_at"] != result.completed_at.isoformat():
            raise CorruptAgentRequestJournalError(
                f"agent request row {row['request_id']} completed_at mirror disagrees"
            )
        if row["error_code"] != result.error_code:
            raise CorruptAgentRequestJournalError(
                f"agent request row {row['request_id']} error_code mirror disagrees"
            )
        return AgentRequestRecord(request=request, result=result)

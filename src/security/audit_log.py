"""
Immutable audit log for insurance compliance (7-year retention).

Every agent action, HITL decision, and final outcome is recorded with:
- SHA-256 hash of the entry (tamper detection)
- Timestamp (UTC)
- Claim ID and agent name
- Input/output snapshots (PII-masked)
- Cost attribution

Log format: newline-delimited JSON (NDJSON) for easy streaming and parsing.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

from src.config import get_security_config

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _hash_entry(entry: dict[str, Any]) -> str:
    """Return a SHA-256 hash of an audit log entry."""

    serialize = json.dumps(
        entry,
        sort_keys=True,
        separators=(",",":"),
        default=str,
    )
    return hashlib.sha256(serialize.encode("utf-8")).hexdigest()


def _get_log_path() -> Path:
    """Return today's audit log file path, creating the directory if needed."""

    security_config = get_security_config()

    log_directory = Path(
        security_config["audit_log"]["path"]
    )

    log_directory.mkdir(
        parents= True,
        exist_ok= True,
    )

    log_file = (
        f"audit_{datetime.now(timezone.utc):%Y-%m-%d}.ndjson"
    )

    return log_directory/ log_file


def _write_entry(claim_id: str, entry: dict)->str:
    """Write an immutable audit entry and return its SHA-256 hash."""
    
    audit_entry = dict(entry) # for shallow copy

    entry_hash = _hash_entry(audit_entry)
    audit_entry["hash"] = entry_hash

    try:
        with _get_log_path().open(mode='a',encoding='utf-8' ) as log_file:
            json.dump(audit_entry, log_file, default= str)
            log_file.write("\n")
    except Exception:
        logger.exception("Failed to write audit log for claim '%s'.",claim_id)   

    return entry_hash


def log_agent_action(
        claim_id:str,
        agent_name: str,
        action: str,
        input_summary: Optional[dict[str, Any]] = None,
    output_summary: Optional[dict[str, Any]] = None,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
    duration_ms: int = 0,
    error: Optional[str] = None,
) -> str:
    """Record a single agent action and return its audit hash."""

    return _write_entry(
        claim_id,
        {
            "timestamp": _now_iso(),
            "claim_id": claim_id,
            "agent": agent_name,
            "action": action,
            "input": input_summary or {},
            "output": output_summary or {},
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "error": error,
        },
    )


def log_hitl_event(
    claim_id: str,
    event: str,
    priority: str,
    triggers: list[str],
    reviewer_id: Optional[str] = None,
    human_decision: Optional[str] = None,
    human_notes: Optional[str] = None,
    override_ai: bool = False,
) -> str:
    """Record a human-in-the-loop review event."""

    return _write_entry(
        claim_id,
        {
            "timestamp": _now_iso(),
            "claim_id": claim_id,
            "event_type": "HITL",
            "hitl_event": event,
            "priority": priority,
            "triggers": triggers,
            "reviewer_id": reviewer_id,
            "human_decision": human_decision,
            "human_notes": human_notes,
            "override_ai": override_ai,
        },
    )


def log_final_decision(
    claim_id: str,
    decision: str,
    amount_usd: float,
    total_tokens: int,
    total_cost_usd: float,
    evaluation_score: Optional[float] = None,
    human_reviewed: bool = False,
) -> str:
    """Record the final claim decision."""

    return _write_entry(
        claim_id,
        {
            "timestamp": _now_iso(),
            "claim_id": claim_id,
            "event_type": "FINAL_DECISION",
            "decision": decision,
            "settlement_amount_usd": amount_usd,
            "total_tokens_used": total_tokens,
            "total_cost_usd": total_cost_usd,
            "evaluation_score": evaluation_score,
            "human_reviewed": human_reviewed,
        },
    )


def get_claim_audit_trail(
        claim_id: str,
        days_back: int = 30,
) -> list[dict[str,any]]:
    """Return audit entries for a claim ordered by timestamp."""

    log_directory = Path(
        get_security_config()["audit_log"]["path"]
    )

    if not log_directory.exists():
        return []
    
    entries: list[dict[str,Any]] = []

    for log_file in sorted(log_directory.glob("audit_*.ndjson"), reverse=True)[:days_back]:
        try:
            with log_file.open(encoding="utf-8") as file:
                for line in file:
                    try:
                        entry = json.loads(line)

                        if entry.get("claim_id") == claim_id:
                            entries.append(entry)
                    
                    except json.JSONDecodeError:
                        logger.warning("skipping malformed audit entry in '%s'.",log_file.name)
        
        except Exception:
            logger.exception("Failed to read audit log '%s'.",log_file.name,)
    
    return sorted(entries, key= lambda entry: entry.get("timestamp",""))
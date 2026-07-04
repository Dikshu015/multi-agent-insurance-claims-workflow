"""
LangGraph pipeline state for the Multi-Agent Insurance Claims Workflow.

Uses TypedDict (not Pydantic BaseModel) because LangGraph merges agent
outputs as plain dicts at runtime -- TypedDict gives static type hints
without wrapping data in class instances that LangGraph cannot merge.

Annotated[list, operator.add] fields are append-only and parallel-safe:
when two agents both write to pipeline_trace, their entries are concatenated
rather than one overwriting the other.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from src.models.schemas import (
    ClaimDecision,
    CommunicationOutput,
    DamageAssessmentOutput,
    EvaluationOutput,
    FraudAssessmentOutput,
    HITLPriority,
    IntakeValidationOutput,
    PolicyCheckOutput,
    SettlementOutput,
)


class ClaimInput(TypedDict):
    """
    Raw claim data as submitted by the claimant via the API.

    Intentionally simple (plain strings and floats) because this represents
    data BEFORE validation. The intake agent validates this and produces
    IntakeValidationOutput. Our rich Claim model (src/models/claim.py)
    represents the validated, structured version of the same data.
    """

    claim_id: str
    policy_number: str
    claimant_name: str
    claimant_email: str
    claimant_phone: str
    # stored as string -- parsed and validated by intake agent
    claimant_dob: str
    incident_date: str
    incident_type: str
    incident_description: str
    incident_location: str
    # None if no police report was filed
    police_report_number: Optional[str]
    estimated_amount: float
    # vehicle fields -- None for non-auto claims
    vehicle_year: Optional[int]
    vehicle_make: Optional[str]
    vehicle_model: Optional[str]
    # list of document identifiers (filenames, S3 keys, etc.)
    documents: list[str]
    # True when this submission is an appeal of a previously denied claim
    is_appeal: bool
    # only set when is_appeal=True -- links back to the original claim
    original_claim_id: Optional[str]


class ClaimsState(TypedDict):
    """
    Full pipeline state. Populated incrementally as each agent runs.

    At graph start almost every field is None. By pipeline end every
    field should be populated. pipeline_trace and error_log use
    operator.add so parallel agents can both append entries safely.
    """

    # ── Input ────────────────────────────────────────────────────────────
    claim: ClaimInput
    # PII-masked version of raw claim sent to all agents -- plain dict so
    # LangGraph's Postgres checkpointer can serialize/deserialize it cleanly
    # across HITL pauses and cold restarts. Agents needing typed access call
    # Claim(**state["masked_claim"]) on demand rather than storing a Pydantic
    # object in state directly.
    masked_claim: Optional[dict]

    # ── Agent Outputs (populated incrementally as each agent completes) ──
    intake_output: Optional[IntakeValidationOutput]
    fraud_output: Optional[FraudAssessmentOutput]
    damage_output: Optional[DamageAssessmentOutput]
    policy_output: Optional[PolicyCheckOutput]
    settlement_output: Optional[SettlementOutput]
    evaluation_output: Optional[EvaluationOutput]
    # True/False extracted from evaluation_output for easy conditional routing
    evaluation_passed: Optional[bool]
    communication_output: Optional[CommunicationOutput]

    # ── HITL ─────────────────────────────────────────────────────────────
    # True triggers interrupt() in graph.py, pausing pipeline for human review
    hitl_required: bool
    # reasons HITL was triggered e.g. ["high_value_claim", "fraud_score_above_threshold"]
    hitl_triggers: list[str]
    hitl_priority: Optional[HITLPriority]
    # weighted score from hitl.priority_weights in base.yaml (maps to SLA hours)
    hitl_priority_score: Optional[float]
    hitl_ticket_id: Optional[str]
    # populated by human reviewer via /api/hitl/{ticket_id}/decision endpoint
    human_decision: Optional[str]
    human_reviewer_id: Optional[str]
    human_notes: Optional[str]
    # True when human reviewer disagrees with and overrides AI recommendation
    human_override: bool

    # ── Final Decision ───────────────────────────────────────────────────
    final_decision: Optional[ClaimDecision]
    # OUR IMPROVEMENT: currency-neutral name (not final_amount_usd)
    # actual currency is claim.currency derived from claimant country
    final_amount: Optional[float]

    # ── Guardrails ───────────────────────────────────────────────────────
    # starts True -- set False when any cap from base.yaml guardrails is exceeded
    guardrails_passed: bool
    # append-only -- each violation appends a message, never overwrites
    guardrails_violations: list[str]
    # incremented by guardrails wrapper on every agent call
    agent_call_count: int
    total_tokens_used: int
    total_cost_usd: float
    execution_start_time: Optional[str]

    # ── Audit and Tracing ────────────────────────────────────────────────
    # operator.add = append-only, parallel-safe -- both agents keep their entries
    # each entry: {"agent": "fraud_crew", "duration_ms": 1240, "confidence": 0.72}
    pipeline_trace: Annotated[list[dict], operator.add]
    # same pattern for errors -- append-only so no error entry is ever lost
    error_log: Annotated[list[str], operator.add]


def initial_state(claim: ClaimInput) -> ClaimsState:
    """
    Build a fresh ClaimsState from a raw ClaimInput.

    Only non-None defaults are listed explicitly. The loop at the end
    sets every remaining TypedDict field to None automatically, so we
    never have to list all Optional fields manually here.
    """
    state: dict = {
        "claim": claim,
        # booleans with explicit non-False defaults
        "hitl_required": False,
        "human_override": False,
        # guardrails start as passed=True, violations=empty
        "guardrails_passed": True,
        # numeric counters start at zero
        "agent_call_count": 0,
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        # append-only lists must start as [] not None
        # (operator.add reducer needs an existing list to concatenate onto)
        "hitl_triggers": [],
        "guardrails_violations": [],
        "pipeline_trace": [],
        "error_log": [],
    }
    # set every remaining Optional field to None without listing them all manually
    for key in ClaimsState.__annotations__:
        if key not in state:
            state[key] = None
    return ClaimsState(**state)
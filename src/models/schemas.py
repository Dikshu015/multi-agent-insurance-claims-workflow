"""
Pydantic v2 schemas for structured LLM outputs.

Every agent returns one of these schemas -- zero free-text parsing,
zero format surprises. Schemas are passed to the LLM as the required
JSON output format via LangChain structured output, forcing the LLM
to return valid typed data rather than freeform text.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# get_country_config used in IntakeValidationOutput to validate claim_type
# at runtime against the active country YAML instead of a hardcoded enum
from src.config import get_country_config


# ── Enums ─────────────────────────────────────────────────────────────────────

class ClaimDecision(str, Enum):
    """
    Final verdict on a claim.

    Seven states rather than just approved/denied -- granularity matters
    because each state maps to a different communication template and
    different downstream pipeline behavior.
    """

    APPROVED = "approved"
    # covered but not fully -- deductible, depreciation, or partial exclusion applied
    APPROVED_PARTIAL = "approved_partial"
    DENIED = "denied"
    # not rejected -- claimant can resubmit with missing documents
    PENDING_DOCUMENTS = "pending_documents"
    # human reviewer must decide -- claim enters HITL queue
    ESCALATED_HITL = "escalated_human_review"
    FRAUD_INVESTIGATION = "fraud_investigation"
    # fraud_score >= auto_reject_threshold (0.90 in base.yaml) -- skips HITL entirely
    AUTO_REJECTED = "auto_rejected"


class FraudRiskLevel(str, Enum):
    """
    Fraud risk classification with four levels.

    CONFIRMED is distinct from HIGH -- HIGH means strong signals present,
    CONFIRMED means fraud_score >= auto_reject_threshold, which triggers
    automatic rejection without human review.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    # triggers AUTO_REJECTED decision, bypasses HITL queue
    CONFIRMED = "confirmed"


class HITLPriority(str, Enum):
    """
    Priority level for claims in the HITL review queue.
    Maps directly to SLA hours in base.yaml hitl.sla_hours.
    """

    CRITICAL = "critical"   # resolve within 4 hours
    HIGH = "high"           # resolve within 24 hours
    NORMAL = "normal"       # resolve within 72 hours


class CoverageStatus(str, Enum):
    """
    Policy coverage determination from the policy checker agent.

    NEEDS_VERIFICATION is important -- it means the agent could not
    definitively determine coverage from available data, which triggers
    HITL rather than auto-deciding either way.
    """

    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    NOT_COVERED = "not_covered"
    # agent uncertain -- routes to human reviewer rather than auto-deciding
    NEEDS_VERIFICATION = "needs_verification"


# ── Agent Output Schemas ──────────────────────────────────────────────────────

class IntakeValidationOutput(BaseModel):
    """
    Output schema for the Claims Intake Agent.

    is_valid is the pipeline gateway -- if False, no further agents run
    and the claim routes to PENDING_DOCUMENTS or rejection immediately.

    claim_type is validated at runtime against the active country's
    registered claim types in configs/countries/{code}.yaml so adding
    a new country never requires changes to this file.
    """

    # pipeline gateway -- False stops all downstream agents immediately
    is_valid: bool = Field(description="Whether the claim passes initial validation")
    # validated against active country YAML at runtime, not a hardcoded enum
    # this means adding a new country requires only a new YAML, not a code change
    claim_type: str = Field(description="Detected claim type")
    policy_active: bool = Field(description="Whether the policy is active and not lapsed")
    claimant_eligible: bool = Field(description="Whether the claimant is eligible to file")
    # strings match keys in country YAML required_documents section
    # default_factory=list -- empty list is valid (claim is complete, nothing missing)
    missing_documents: list[str] = Field(
        default_factory=list,
        description="Required documents not yet submitted"
    )
    intake_notes: str = Field(description="Summary of intake findings")
    # ge/le enforced by Pydantic -- LLM cannot hallucinate a value outside 0-1
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    # catch-all for anything unusual that does not fit other fields
    validation_flags: list[str] = Field(
        default_factory=list,
        description="Warnings or flags raised during intake"
    )

    @field_validator("claim_type")
    @classmethod
    def validate_claim_type(cls, v: str) -> str:
        """
        Validate claim_type against the active country's registered claim types.

        'unknown' is always accepted -- the intake agent uses it when the
        claim type cannot be determined from submission data, which is a
        valid signal rather than an error.
        """
        # unknown always valid -- honest signal that type could not be determined
        if v == "unknown":
            return v
        valid_types = get_country_config().get("claim_types", [])
        if v not in valid_types:
            raise ValueError(
                f"claim_type '{v}' is not registered for the active country. "
                f"Valid types: {valid_types}"
            )
        return v


class FraudPatternOutput(BaseModel):
    """
    Output schema for the CrewAI Pattern Analyst agent.

    One of three specialist outputs produced inside the fraud crew.
    Synthesized alongside AnomalyDetectionOutput and SocialValidationOutput
    into a final FraudAssessmentOutput by the crew.
    """

    # default_factory=list -- empty list is valid (no patterns matched is a real outcome)
    pattern_matches: list[str] = Field(
        default_factory=list,
        description="Fraud patterns matched from database"
    )
    risk_indicators: list[str] = Field(
        default_factory=list,
        description="Specific fraud indicators detected"
    )
    # pattern-based probability only -- not the final composite score
    pattern_score: float = Field(ge=0.0, le=1.0, description="Pattern-based fraud probability 0-1")
    analysis: str = Field(description="Detailed pattern analysis narrative")


class AnomalyDetectionOutput(BaseModel):
    """
    Output schema for the CrewAI Anomaly Detector agent.

    Checks claim against fraud_baselines in the active country YAML
    (avg_amount, std_dev, max_normal per claim type) to flag statistical outliers.
    """

    # empty list valid -- no anomalies found is the expected outcome for clean claims
    statistical_anomalies: list[str] = Field(
        default_factory=list,
        description="Statistical outliers detected"
    )
    # True if claimant has filed more claims than normal within repeat_claims_window_days
    claim_frequency_flag: bool = Field(description="High claim frequency detected")
    # checked against fraud_baselines avg_amount +/- std_dev in country YAML
    amount_anomaly: bool = Field(description="Amount deviates significantly from similar claims")
    # e.g. policy purchased 3 days before the claimed incident date
    timing_anomaly: bool = Field(description="Claim timing is suspicious")
    anomaly_score: float = Field(ge=0.0, le=1.0, description="Anomaly-based fraud probability 0-1")
    analysis: str = Field(description="Detailed anomaly analysis narrative")


class SocialValidationOutput(BaseModel):
    """
    Output schema for the CrewAI Social Validator agent.

    Checks internal consistency of the claimant's account and flags
    identity concerns. validation_score is a consistency score (higher = more
    consistent), not a fraud probability -- inverse of pattern_score/anomaly_score.
    """

    story_consistent: bool = Field(description="Claimant account is internally consistent")
    # empty list valid -- no inconsistencies found is the normal outcome
    inconsistencies: list[str] = Field(
        default_factory=list,
        description="Specific inconsistencies in claim narrative"
    )
    identity_flags: list[str] = Field(
        default_factory=list,
        description="Identity verification concerns"
    )
    # higher = more consistent (not a fraud score -- inverse of the other two)
    validation_score: float = Field(ge=0.0, le=1.0, description="Story consistency score 0-1")
    analysis: str = Field(description="Detailed validation analysis narrative")


class FraudAssessmentOutput(BaseModel):
    """
    Final output from the full CrewAI Fraud Detection Crew.

    Synthesizes FraudPatternOutput + AnomalyDetectionOutput + SocialValidationOutput
    into one verdict the rest of the LangGraph pipeline reads. Individual scores
    are kept as separate fields so HITL reviewers can see which crew member
    flagged what, not just the final composite.
    """

    fraud_risk_level: FraudRiskLevel = Field(description="Overall fraud risk classification")
    # composite score -- weighted combination of pattern, anomaly, consistency scores
    fraud_score: float = Field(ge=0.0, le=1.0, description="Composite fraud probability 0-1")
    # empty list valid -- low risk claims may have no primary concerns to highlight
    primary_concerns: list[str] = Field(
        default_factory=list,
        description="Top fraud concerns for HITL reviewer"
    )
    # "proceed" | "escalate" | "reject" -- maps to ClaimDecision in settlement agent
    recommendation: str = Field(description="Recommended action: proceed | escalate | reject")
    crew_summary: str = Field(description="Synthesized findings from all three crew members")
    # individual specialist scores preserved for HITL reviewer transparency
    pattern_score: float = Field(ge=0.0, le=1.0)
    anomaly_score: float = Field(ge=0.0, le=1.0)
    # higher = more consistent (inverse of pattern/anomaly scores)
    consistency_score: float = Field(ge=0.0, le=1.0)


class DamageAssessmentOutput(BaseModel):
    """
    Output schema for the Damage Assessment Agent.

    OUR IMPROVEMENT: field names are currency-neutral (assessed_damage_amount,
    not assessed_damage_usd) because currency is determined by claim.currency
    derived from the claimant's country, not hardcoded as USD.
    """

    # currency is claim.currency (derived from claimant country) not hardcoded USD
    assessed_damage_amount: float = Field(ge=0.0, description="Total assessed damage amount")
    # [{"part": "front bumper", "cost": 1200, "action": "replace"}, ...]
    line_items: list[dict] = Field(
        default_factory=list,
        description="Itemized damage breakdown"
    )
    # "repair" | "replace" | "total_loss"
    repair_vs_replace: str = Field(description="Repair, replace, or total loss recommendation")
    assessment_confidence: float = Field(ge=0.0, le=1.0)
    assessment_notes: str = Field(description="Assessor methodology and reasoning notes")
    # True if damage is too complex or ambiguous for LLM assessment alone
    requires_physical_inspection: bool = Field(description="Physical inspector visit needed")
    # None if no comparable data available for this country + claim_type combination
    comparable_claims_avg: Optional[float] = Field(
        None,
        description="Average payout for comparable claims in this country"
    )


class PolicyCheckOutput(BaseModel):
    """
    Output schema for the Policy Compliance Agent.

    covered_amount and deductible are the two numbers the settlement
    agent needs to calculate the final payout. Field names are
    currency-neutral (not usd-suffixed) matching our DamageAssessmentOutput convention.
    """

    coverage_status: CoverageStatus
    # currency-neutral -- amount is in claim.currency (INR or USD)
    covered_amount: float = Field(ge=0.0, description="Amount covered by policy")
    deductible: float = Field(ge=0.0, description="Applicable deductible amount")
    # e.g. ["act_of_god", "pre_existing_damage"] -- used in denial explanation
    exclusions_triggered: list[str] = Field(
        default_factory=list,
        description="Policy exclusions that reduce or eliminate coverage"
    )
    coverage_notes: str = Field(description="Explanation of coverage determination")
    # regulatory concerns e.g. state-mandated minimum payout requirements
    compliance_flags: list[str] = Field(
        default_factory=list,
        description="Regulatory compliance concerns"
    )
    # raw dict -- policy limit structures vary too much by type to constrain here
    policy_limits: dict = Field(description="Applicable policy limits")
    confidence: float = Field(ge=0.0, le=1.0)


class SettlementOutput(BaseModel):
    """
    Output schema for the Settlement Calculator Agent.

    The most consequential output -- drives the final claim decision,
    the claimant communication, and the HITL reviewer display.
    calculation_breakdown is the step-by-step math trail shown to both
    the claimant and any human reviewer.
    """

    decision: ClaimDecision
    # currency-neutral -- amount is in claim.currency (INR or USD)
    settlement_amount: float = Field(ge=0.0, description="Final settlement amount")
    gross_damage: float = Field(ge=0.0, description="Damage amount before any deductions")
    deductible_applied: float = Field(ge=0.0, description="Deductible deducted from gross")
    depreciation_applied: float = Field(ge=0.0, description="Depreciation deducted from gross")
    # step-by-step trail: ["Gross: ₹50000", "minus deductible: ₹5000", "net: ₹45000"]
    # max 6 steps keeps communication readable for claimants
    calculation_breakdown: list[str] = Field(
        description="Step-by-step calculation, max 6 steps"
    )
    # default_factory=list -- approved claims have no denial reasons (empty is correct)
    denial_reasons: list[str] = Field(
        default_factory=list,
        description="Populated only when decision is DENIED or AUTO_REJECTED"
    )
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    # set False only if payout violates a regulatory rule -- triggers compliance alert
    regulatory_compliance: bool = Field(
        default=True,
        description="Payout passes all applicable regulatory requirements"
    )


class EvaluationOutput(BaseModel):
    """
    Output schema for the LLM-as-Judge Evaluator.

    Scores the settlement decision across five dimensions defined in
    base.yaml evaluation.dimensions. passed=True if overall_score >=
    evaluation.min_score_to_release (0.70). Failed evaluations trigger
    re-evaluation, not automatic rejection.
    """

    # weighted average of the five dimension scores below
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall decision quality 0-1")
    # is the settlement amount correct given damage assessment + policy limits?
    accuracy_score: float = Field(ge=0.0, le=1.0)
    # were all policy clauses, exclusions, and country rules checked?
    completeness_score: float = Field(ge=0.0, le=1.0)
    # consistent with how similar claims were handled previously?
    fairness_score: float = Field(ge=0.0, le=1.0)
    # guardrails followed? PII masked? cost caps respected?
    safety_score: float = Field(ge=0.0, le=1.0)
    # is reasoning traceable and explainable to the claimant?
    transparency_score: float = Field(ge=0.0, le=1.0)
    # True if overall_score >= evaluation.min_score_to_release from base.yaml
    passed: bool = Field(description="Decision passes quality gate")
    feedback: str = Field(description="Improvement recommendations when passed=False")
    # critical issues flagged regardless of overall score
    # e.g. potential regulatory violation even if overall_score is 0.75
    flags: list[str] = Field(
        default_factory=list,
        description="Critical issues requiring immediate attention"
    )


class CommunicationOutput(BaseModel):
    """
    Output schema for the Communication Agent.

    message and internal_notes are strictly separated -- message goes
    to the claimant, internal_notes goes to the adjuster/reviewer only.
    appeal_instructions is None for approved claims, populated only on
    denials or partial approvals.
    """

    subject: str = Field(description="Email/notification subject line")
    # the message the claimant actually receives
    message: str = Field(description="Full notification message to claimant")
    # never sent to claimant -- for adjuster and HITL reviewer only
    internal_notes: str = Field(description="Internal adjuster notes")
    next_steps: list[str] = Field(
        default_factory=list,
        description="Action items for claimant"
    )
    # None for approved claims -- only populated on DENIED or APPROVED_PARTIAL
    appeal_instructions: Optional[str] = Field(
        None,
        description="How to appeal -- populated for DENIED and APPROVED_PARTIAL only"
    )
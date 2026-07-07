"""
PII Masking Layer - country-aware, config-driven.

Loads PII field lists + regex patterns from the active country profile
(configs/countries/{code}.yaml -> pii block). Falls back to a safe default
if no country config is loaded yet.

Replaces real PII with deterministic placeholders so:
  1. The agent can still reason about the claim (e.g. "CLAIMANT_NAME filed on INCIDENT_DATE")
  2. Real data never leaves our infrastructure in LLM prompts
  3. De-masking is possible from the original claim object (held in secure memory)

US: masks SSN, driver license, US phone, ZIP
India: masks Aadhaar, PAN, Indian mobile, pincode
"""

from __future__ import annotations

import copy
import logging
import re
from functools import lru_cache
from typing import Any


logger = logging.getLogger(__name__)


# ── Defaults (used if country config isn't available yet) ────────────────────

_DEFAULT_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "PHONE": re.compile(r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"),
    "SSN": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "DOB": re.compile(r"\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12]\d|3[01])[\/\-](\d{2}|\d{4})\b"),
}

_NAME_FIELDS: set[str] = {
    "claimant_name",
    "name",
    "insured_name",
    "beneficiary_name",
}

_DEFAULT_REDACT_FIELDS: set[str] = {
    "claimant_dob",
    "ssn",
    "bank_account",
    "credit_card",
    "password",
}


def _get_country_pii_config() -> dict[str, Any]:
    """Load PII config from active country. Return empty dict on failure."""
    try:
        from src.config import get_pii_config
        return get_pii_config()
    except Exception:
        logger.exception("Failed to load country PII configuration.")
        return {}
    

@lru_cache(maxsize=1)
def _build_patterns() -> dict[str, re.Pattern[str]]:
    """Build compiled PII regex patterns for the active country."""

    pii = _get_country_pii_config()
    custom_patterns = pii.get("patterns", {})

    if not custom_patterns:
        return dict(_DEFAULT_PATTERNS)
    
    patterns: dict[str, re.Pattern[str]] = {}

    for label, regex in custom_patterns.items():
        try:
            patterns[label.upper()] = re.compile(regex)
        except re.error:
            logger.warning("Invalid PII regex for '%s'.", label)
    
    return patterns


@lru_cache(maxsize=1)
def _get_redact_fields() -> set[str]:
    """Return fields that should always be redacted."""

    pii = _get_country_pii_config()
    country_redact = set(pii.get("redact_fields", []))

    return _DEFAULT_REDACT_FIELDS | country_redact


def clear_pii_cache() -> None:
    """Clear cached PII configuration."""

    _build_patterns.cache_clear()
    _get_redact_fields.cache_clear()


def _mask_string(text:str, patterns: dict[str, re.Pattern[str]]) -> str:
    """Mask PII within a string using compiled regex patterns."""

    masked = text
    for label, pattern in patterns.items():
        masked = pattern.sub(f"[{label}]",masked)
    
    return masked


def mask_text(text:str) -> str:
    """Apply regex-based PII masking to a string."""

    if not isinstance(text, str) or not text:
        return text
    
    return _mask_string(
        text,
        _build_patterns(),
    )


def _mask_recursive(
    obj: Any,
    redact_fields: set[str],
    patterns: dict[str, re.Pattern[str]],
) -> None:
    """Recursively mask PII within dictionaries and lists."""

    # dictionary handling
    if isinstance(obj, dict):
        for key, value in obj.items():
            field = key.lower()

            if field in redact_fields:
                obj[key] = "[REDACTED]"

            elif field in _NAME_FIELDS:
                if isinstance(value, str) and value:
                    obj[key] = "[CLAIMANT_NAME]"

            elif isinstance(value, str):
                obj[key] = _mask_string(
                    value,
                    patterns,
                )

            else:
                _mask_recursive(
                    value,
                    redact_fields,
                    patterns,
                )
    # list handling
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            if isinstance(item, str):
                obj[index] = _mask_string(
                    item,
                    patterns,
                )
            else:
                _mask_recursive(
                    item,
                    redact_fields,
                    patterns,
                )


def mask_claim(claim: dict) -> dict:
    """Deep-copy a claim dict and mask all PII. Safe for LLM consumption."""

    masked = copy.deepcopy(claim)

    redact_fields = _get_redact_fields()
    patterns = _build_patterns()

    _mask_recursive(masked, redact_fields, patterns)

    return masked


def get_masked_summary(claim: dict[str, Any]) -> str:
    """Return a PII-safe natural language summary of a claim."""

    from src.utils import currency_symbol

    masked_claim = mask_claim(claim)
    symbol = currency_symbol()

    return (
        f"Claim ID: {masked_claim.get('claim_id', 'UNKNOWN')} | "
        f"Policy: {masked_claim.get('policy_number', 'UNKNOWN')} | "
        f"Claimant: {masked_claim.get('claimant_name', '[CLAIMANT_NAME]')} | "
        f"Incident: {masked_claim.get('incident_type', 'UNKNOWN')} "
        f"on {masked_claim.get('incident_date', 'UNKNOWN')} | "
        f"Location: {masked_claim.get('incident_location', 'UNKNOWN')} | "
        f"Amount: {symbol}{masked_claim.get('estimated_amount', 0):,.2f} | "
        f"Description: {masked_claim.get('incident_description', 'N/A')}"
    )

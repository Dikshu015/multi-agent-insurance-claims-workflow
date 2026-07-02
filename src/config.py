"""
Configuration loader.

Hierarchy (highest precedence first):
  1. Runtime override  -- via set_llm_provider_override() / set_country_override()
  2. .env              -- LLM_PROVIDER, COUNTRY, API keys, threshold overrides
  3. Country YAML      -- configs/countries/{country}.yaml
  4. Base YAML         -- configs/base.yaml

Never import os.environ or read YAML directly in other modules.
Always go through the accessor functions here instead.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).parent.parent / "configs"

CONFIG_PATH = _CONFIG_DIR / "base.yaml"

# country-specific overrides (currency, PII patterns, depreciation, fraud baselines)
_COUNTRIES_DIR = _CONFIG_DIR / "countries"

# ── YAML loaders ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_raw() -> dict:
    """Load and cache base.yaml. Called once per process lifetime."""
    # utf-8 explicit -- india.yaml contains ₹ symbol which breaks on Windows cp1252 default
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)
    

# code.lower() normalizes "India"/"INDIA"/"india" to the same cache key
@lru_cache(maxsize=4)
def _load_country_yaml(code: str) -> dict:
    """Load and cache a country profile YAML by country code."""
    path = _COUNTRIES_DIR / f"{code.lower()}.yaml"
    # clear error message instead of Python's default FileNotFoundError with no context
    if not path.exists():
        raise FileNotFoundError(f"Country profile not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
    
## _deep_merge is the most important utility in this file.
## Problem it solves: india.yaml only overrides hitl.triggers.min_amount,
## but a shallow {**base, **country} would wipe the ENTIRE hitl.triggers block.
## _deep_merge recurses into nested dicts, merging key by key so only
## the keys the country YAML actually specifies get overridden.
def _deep_merge(base: dict, overlay: dict) -> dict:
    """
    Recursively merge overlay into base. Overlay wins on leaf conflicts.

    Used so country YAMLs can partially override base.yaml without wiping
    entire nested blocks -- e.g. india.yaml sets hitl.triggers.min_amount
    without touching hitl.triggers.fraud_score (which stays from base).
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key],dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key],value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ── Runtime overrides ─────────────────────────────────────────────────────────
# Written by /api/settings endpoints. Sit above .env in the precedence chain.
# Process-scoped -- reset on server restart.

_runtime_overrides: dict = {}


def set_llm_provider_override(provider: str | None) -> None:
    """Set or clear a runtime LLM provider override."""
    if provider is None:
        _runtime_overrides.pop("llm_provider", None)
    else:
        _runtime_overrides["llm_provider"] = provider.lower()


def set_country_override(code: str | None) -> None:
    """Set or clear a runtime country override."""
    if code is None:
        _runtime_overrides.pop("country", None)
    else:
        _runtime_overrides["country"] = code.lower()


# ── Country resolution ────────────────────────────────────────────────────────
def _active_country_code() -> str:
    """Resolve active country code: runtime override > .env > default 'india'."""
    return (
        _runtime_overrides.get("country") 
        or os.getenv("COUNTRY", "india").strip().lower()
    )


def get_available_countries() -> list[str]:
    """Return stored list of country codes that have a YAML profile on disk"""
    return sorted(p.stem for p in _COUNTRIES_DIR.glob("*.yaml"))


# ── Public config accessors ───────────────────────────────────────────────────

def get_config() -> dict:
    """Raw base YAML. Prefer the focused accessors below over calling this directly."""
    return _load_raw()


def get_app_config() -> dict:
    """App metadata block (name, version)"""
    return _load_raw().get("app", {})


def get_country_config() -> dict:
    """Full country profile YAML for the active country."""
    return _load_country_yaml(_active_country_code())


def get_country_meta() -> dict:
    """Just the country.* block (code, name, currency, symbol, regulator, tz)."""
    return get_country_config().get("country", {})


def get_llm_config() -> dict:
    """
    LLM config merging base YAML tunables + active provider block + env overrides.
    Resolves: provider, model, fallback_model, api_key_env, temperature, etc.
    """
    cfg = dict(_load_raw()["llm"]) # dict(existing_dict) creates a shallow copy. works for only top level
    provider = (
        _runtime_overrides.get("llm_provider")
        or os.getenv("LLM_PROVIDER", "groq")
    )
    cfg["provider"] = provider.lower()
    provider_cfg = cfg.get("providers",{}).get(cfg["provider"], {})
    cfg["model"] = os.getenv("LLM_MODEL") or provider_cfg.get("model")
    cfg["fallback_model"] = provider_cfg.get("fallback_model")
    cfg["api_key_env"] = provider_cfg.get("api_key_env", "GROQ_API_KEY") # it's providing name of API_KEY in our .env not direct api_key
    
    temp = os.getenv("LLM_TEMPERATURE")
    if temp is not None:
        cfg["temperature"] = float(temp)

    return cfg


def get_agent_config(agent_name: str) -> dict:
    """Config block for one named agent (e.g. 'fraud_crew', 'intake')."""
    return _load_raw()["agents"].get(agent_name,{})


def get_hitl_config() -> dict:
    """
    HITL config: base.yaml merged with country YAML overrides, then env overrides.
    Country YAMLs override min_amount and first_claim_high_value with local currency values.
    """
    base = copy.deepcopy(_load_raw()["hitl"])
    country_hitl = get_country_config().get("hitl",{})
    cfg = _deep_merge(base, country_hitl)
    # if os.getenv("HITL_MIN_AMOUNT"):
    #     cfg["triggers"]["min_amount"] = float(os.getenv("HITL_MIN_AMOUNT"))
    # if os.getenv("HITL_FRAUD_THRESHOLD"):
    #     cfg["triggers"]["fraud_score"] = float(os.getenv("HITL_FRAUD_THRESHOLD"))
    # if os.getenv("HITL_LOW_CONFIDENCE"):
    #     cfg["triggers"]["low_confidence"] = float(os.getenv("HITL_LOW_CONFIDENCE"))

    overrides = {
        "HITL_MIN_AMOUNT": "min_amount",
        "HITL_FRAUD_THRESHOLD": "fraud_score",
        "HITL_LOW_CONFIDENCE": "low_confidence",
    }

    for env_name, cfg_key in overrides.items():
        value = os.getenv(env_name)
        if value is not None:
            cfg["triggers"][cfg_key] = float(value)
    return cfg   


def get_pii_config() -> dict:
    """PII field names + regex patterns from the active country profile."""
    return get_country_config().get("pii", {})


def get_depreciation_config() -> dict:
    """Depreciation rules from the active country profile."""
    return get_country_config().get("depreciation", {})


def get_settlement_config() -> dict:
    """Settlement rules from the active country profile."""
    return get_country_config().get("settlement", {})


def get_communication_config() -> dict:
    """Communication templates and contact info from the active country profile."""
    return get_country_config().get("communication", {})


def get_fraud_baselines() -> dict:
    """Fraud statistical baselines (avg, std_dev, max_normal) from active country."""
    return get_country_config().get("fraud_baselines", {})


def get_coverage_mapping() -> dict:
    """Claim type -> coverage category mapping for the active country."""
    return get_country_config().get("coverage_mapping", {})


def get_required_documents(claim_type: str) -> list[str]:
    """Required documents list for a specific claim type in the active country."""
    docs = get_country_config().get("required_documents", {})
    return docs.get(claim_type, [])


def get_guardrails_config() -> dict:
    """Guardrails config with env overrides for per-claim cost/token/call caps."""
    cfg = copy.deepcopy(_load_raw()["guardrails"])

    overrides = {
        "MAX_TOKENS_PER_CLAIM": ("max_tokens_per_claim", int),
        "MAX_COST_PER_CLAIM": ("max_cost_usd", float),
        "MAX_AGENT_CALLS": ("max_agent_calls", int),
    }

    for env_var, (key, cast) in overrides.items():
        value = os.getenv(env_var)
        if value is not None:
            cfg[key] = cast(value)

    return cfg


def get_security_config() -> dict:
    """
    Security + audit log config. deepcopy prevents mutating the lru_cache object.
    (Original bug: direct assignment mutated the cached dict across calls.)
    """
    cfg = copy.deepcopy(_load_raw()["security"])
    audit_log_path = os.getenv("AUDIT_LOG_PATH")
    if audit_log_path:
        cfg["audit_log"]["path"] = audit_log_path
    cfg["pii_masking"] = os.getenv("PII_MASKING_ENABLED", "true").lower() == "true"
    return cfg


def get_evaluation_config() -> dict:
    """Evaluation/judge config (dimensions, min score, sample rate)."""
    return _load_raw()["evaluation"]


def get_confidence_gate_config() -> dict:
    """Per-agent confidence gate thresholds for HITL routing."""
    return _load_raw().get("confidence_gates", {"enabled": False})


def get_pipeline_config() -> dict:
    """Pipeline config (fast mode, appeal workflow, parallelism)."""
    return _load_raw()["pipeline"]


def get_output_config() -> dict:
    """Output format config (reasoning, traces, cost breakdown, format)."""
    return _load_raw()["output"]
"""
Application configuration loader.

Combines two sources into a single typed Settings object:
  1. .env          -> secrets and environment-specific values (API keys, DB URL)
  2. configs/base.yaml -> non-secret structural config (thresholds, model names)

Every other module should import `get_settings()` from here rather than
reading os.environ or YAML directly — this keeps config access centralized
and validated in one place instead of scattered string lookups everywhere.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env into the process environment as early as possible,
# before any Settings object is constructed.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "base.yaml"


class ConfidenceGates(BaseModel):
    """Per-agent confidence thresholds below which a claim pauses for human review."""
    intake_agent: float
    fraud_crew: float
    damage_assessor: float
    policy_checker: float
    settlement_calculator: float


class Guardrails(BaseModel):
    """Per-claim usage caps to bound LLM cost if an agent loops or misbehaves."""
    max_agent_calls_per_claim: int
    max_tokens_per_claim: int
    max_cost_usd_per_claim: float


class HitlThresholds(BaseModel):
    """Risk-based routing thresholds, independent of per-agent confidence."""
    high_value_threshold_usd: float
    high_value_threshold_inr: float
    fraud_score_threshold: float


class Settings(BaseModel):
    """
    Fully-resolved application settings, combining secrets (.env)
    and structural config (base.yaml) into one validated object.
    """
    # From .env (secrets / environment-specific)
    llm_provider: str
    groq_api_key: str
    google_api_key: str
    country: str
    database_url: str
    api_secret_key: str

    # From base.yaml (structural, non-secret)
    app_name: str
    groq_model: str
    gemini_model: str
    temperature: float
    confidence_gates: ConfidenceGates
    guardrails: Guardrails
    hitl: HitlThresholds


@lru_cache
def get_settings() -> Settings:
    """
    Build and cache the Settings object.

    lru_cache ensures the YAML file is parsed and .env is read only once
    per process, not on every call -- config doesn't change at runtime.
    """
    with open(CONFIG_PATH, "r") as f:
        yaml_config = yaml.safe_load(f)

    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", yaml_config["llm"]["default_provider"]),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        country=os.getenv("COUNTRY", yaml_config["app"]["default_country"]),
        database_url=os.getenv("DATABASE_URL", ""),
        api_secret_key=os.getenv("API_SECRET_KEY", ""),
        app_name=yaml_config["app"]["name"],
        groq_model=yaml_config["llm"]["groq_model"],
        gemini_model=yaml_config["llm"]["gemini_model"],
        temperature=yaml_config["llm"]["temperature"],
        confidence_gates=ConfidenceGates(**yaml_config["confidence_gates"]),
        guardrails=Guardrails(**yaml_config["guardrails"]),
        hitl=HitlThresholds(**yaml_config["hitl"]),
    )
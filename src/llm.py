"""
LLM factory with pluggable provider (Gemini or Groq).

Provider selection precedence:
  1. runtime override via api.settings /api/settings/llm
  2. LLM_PROVIDER env var
  3. configs/base.yaml -> llm.provider

Each provider declares its own model + fallback_model + api_key_env in base.yaml.
Both providers are kept on production-stable models (verified 2026-04).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
import os
from typing import Any
from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from pydantic import SecretStr

from src.config import get_llm_config


logger = logging.getLogger(__name__)


# ── Token/cost tracking (process-wide, per-claim reset) ─────────────────────

# Approximate pricing per 1M tokens (input/output) - update as rates change.
_PRICING: dict[str, dict[str,float]] = {
    # Groq (free tier shows 0, paid tier is very cheap)
    "llama-3.3-70b-versatile":{
        "input" : 0.59,
        "output" : 0.79,
    },
    "llama-3.1-8b-instant": {
        "input": 0.05,
        "output": 0.08,
    },
    # Gemini (current models in configs/base.yaml)
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.00,
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 2.50,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.10,
        "output": 0.40,
    },
}


_accumulated_tokens: dict[str, float] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "estimated_cost": 0.0,
}


class _TokenTracker(BaseCallbackHandler):
    """Tracks token usage and estimated cost across LLM calls."""

    def __init__(self, model_name: str) -> None:
        pricing = _PRICING.get(model_name,{})
        self._input_price = pricing.get("input",0.0)
        self._output_price = pricing.get("output", 0.0)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Update accumulated token usage after each successful LLM call."""
        
        for generation_list in response.generations:
                for generation in generation_list:
                    try:
                        usage = self._extract_usage(generation)
                        if not usage:
                            continue

                        input_tokens = (
                            usage.get("input_tokens")
                            or usage.get("prompt_tokens")
                            or usage.get("prompt_token_count")
                            or 0
                        )

                        output_tokens = (
                            usage.get("output_tokens")
                            or usage.get("completion_tokens")
                            or usage.get("candidates_token_count")
                            or 0
                        )

                        total_tokens = input_tokens + output_tokens

                        _accumulated_tokens["input_tokens"] += input_tokens
                        _accumulated_tokens["output_tokens"] += output_tokens
                        _accumulated_tokens["total_tokens"] += total_tokens

                        _accumulated_tokens["estimated_cost"] += (
                            input_tokens * self._input_price
                            + output_tokens * self._output_price
                        ) / 1_000_000

                    except AttributeError:
                        logger.exception("Failed to extract token usage from LLM response.")

    
    @staticmethod
    def _extract_usage(generation: Any) -> dict[str, Any]:
        """Extract token usage from provider-specific response metadata."""

        generation_info = getattr(generation, "generation_info", {}) or {}

        usage = (
            generation_info.get("usage_metadata")
            or generation_info.get("token_usage")
        )

        if usage:
            return usage

        message = getattr(generation, "message", None)
        if message is None:
            return {}

        response_metadata = getattr(message, "response_metadata", {}) or {}

        return (
            getattr(message, "usage_metadata", None)
            or response_metadata.get("usage_metadata")
            or response_metadata.get("token_usage")
            or {}
        )
    

def reset_token_tracking() -> None:
    """Reset accumulated token usage for a new pipeline run."""

    _accumulated_tokens.update(
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
        }
    )


def get_token_usage() -> dict[str, float]:
    """Return accumulated token usage and estimated cost."""

    return dict(_accumulated_tokens)


def _build_groq(
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
) -> BaseChatModel:
    """Create a Groq chat model."""

    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming = streaming,
        api_key=SecretStr(api_key),
        callbacks=[_TokenTracker(model)],
    )


def _build_gemini(
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
) -> BaseChatModel:
    """Create a Gemini chat model."""

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming = streaming,
        google_api_key=api_key,
        callbacks=[_TokenTracker(model)],
    )


_BUILDERS: dict[str, Callable[..., BaseChatModel]] = {  # Callable[[ArgumentTypes], ReturnType]
    "groq": _build_groq,
    "gemini": _build_gemini,
}


def get_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,  
) -> BaseChatModel:
    """Return the configured LLM with automatic fallback support."""

    llm_config = get_llm_config()

    provider = llm_config["provider"]
    model = llm_config["model"]
    fallback_model = llm_config["fallback_model"]

    builder = _BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            f"Supported providers: {list(_BUILDERS)}"
        )
    
    api_key = os.getenv(llm_config["api_key_env"])

    if not api_key:
        raise ValueError(
            f"Environment variable '{llm_config['api_key_env']}' is not set."
        )
    
    temperature = (
        temperature
        if temperature is not None
        else llm_config.get("temperature",0.1)
    )

    max_tokens = (
        max_tokens
        if max_tokens is not None
        else llm_config.get("max_tokens", 8192)
    )

    # Fallback logic
    try:
        llm = builder(
            model=model,
            api_key = api_key,
            temperature = temperature,
            max_tokens = max_tokens,
            streaming = streaming
        )

        logger.debug(
            "Initialized %s model '%s'.",
            provider,
            model,
        )

        return llm

    except Exception:
        if not fallback_model or fallback_model == model:
            logger.exception(
                "Failed to initialize model '%s'. No fallback available.",
                model,
            )
            raise

        logger.exception(
            "Failed to initialize primary model '%s'. Falling back to '%s'.",
            model,
            fallback_model,
        )

        return builder(
            model=fallback_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens = max_tokens,
            streaming = streaming,
        )
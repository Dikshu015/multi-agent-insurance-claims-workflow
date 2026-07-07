"""
Shared utilities used across agents and tools.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def currency_symbol() -> str:
    """Return the currency symbol for the active country."""

    try:
        from src.config import get_country_meta

        meta = get_country_meta()
        return meta.get("currency_symbol", "$")

    except Exception:
        logger.exception("Failed to load country metadata.")
        return "$"
"""
Core data models for an insurance claim.

These models define the canonical shape of a claim as it flows through
the system -- from initial submission, through every agent, to final
settlement. FastAPI request/response bodies and LangGraph state both
build on top of these.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, EmailStr


class Country(str, Enum):
    """Supported country profiles -- drives currency, ID type, and regulatory rules."""
    INDIA = "india"
    Us = "us"


class IdType(str, Enum):
    """
    Identity document types, scoped per country.
    Validated against the claimant's country at the model level (see Claimant below)
    rather than as a flat global enum, so an Aadhaar number can't be submitted for a US claim.
    """
    AADHAAR = "aadhaar"
    PAN = "pan"
    SSN = "ssn"
    DRIVERS_LICENSE = "drivers_license"


class ClaimStatus(str, Enum):
    """Lifecycle states a claim moves through in the pipeline."""
    SUBMITTED = "submitted"
    IN_REVIEW = "in_review"
    PENDING_HITL = "pending_hitl"
    APPROVED = "approved"
    REJECTED = "rejected"
    SETTLED = "settled"


class Claimant(BaseModel):
    """The person filing the claim."""
    full_name: str = Field(min_length=1, max_length= 200)
    email: EmailStr
    phone: str = Field(pattern=r"^\+?[0-9\s\-]{7,15}$")
    country: Country
    id_type: IdType
    id_value: str = Field(min_length=1)
    date_of_birth: date
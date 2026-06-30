"""
Core data models for an insurance claim.

These models define the canonical shape of a claim as it flows through
the system -- from initial submission, through every agent, to final
settlement. FastAPI request/response bodies and LangGraph state both
build on top of these.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, EmailStr, model_validator


class Country(str, Enum):
    """Supported country profiles -- drives currency, ID type, and regulatory rules."""
    INDIA = "india"
    US = "us"


class Currency(str, Enum):
    """Supported currencies. Add new entries here when a new Country is added."""
    INR = "INR"
    USD = "USD"


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


# Single source of truth: country -> currency. A claim's currency is
# derived from this map (see Claim.currency), never stored independently,
# so it can never disagree with the claimant's country.
COUNTRY_CURRENCY_MAP: dict[Country, Currency] = {
    Country.INDIA: Currency.INR,
    Country.US: Currency.USD,
}

# Single source of truth: country -> allowed ID document types.
# Unlike currency this is one-to-many (e.g. India allows Aadhaar OR PAN),
# so it's enforced via validation rather than derivation.
COUNTRY_ID_TYPES_MAP: dict[Country, set[IdType]] = {
    Country.INDIA: {IdType.AADHAAR, IdType.PAN},
    Country.US: {IdType.SSN, IdType.DRIVERS_LICENSE},
}


class Claimant(BaseModel):
    """The person filing the claim."""
    full_name: str = Field(min_length=1, max_length= 200)
    email: EmailStr
    phone: str = Field(pattern=r"^\+?[0-9\s\-]{7,15}$")
    country: Country
    id_type: IdType
    id_value: str = Field(min_length=1)
    date_of_birth: date

    @model_validator(mode="after")
    def validate_id_type_matches_country(self) -> "Claimant":
        """
        Rejects mismatched country/id_type combinations, e.g. an SSN
        submitted for an India-based claimant. Runs after individual
        field validation, since it needs both fields populated already.
        """
        allowed_types = COUNTRY_ID_TYPES_MAP[self.country]
        if self.id_type not in allowed_types:
            allowed_names = ", ".join(t.value for t in allowed_types)
            raise ValueError(
                f"id_type '{self.id_type.value}' is not valid for country "
                f"'{self.country.value}'. Allowed: {allowed_names}"
            )
        return self


class PolicyInfo(BaseModel):
    """The insurance policy this claim is being filed against."""
    policy_number: str = Field(min_length=1)
    policy_type: str  # e.g. "auto", "health", "property"
    coverage_limit: Decimal = Field(gt=0)
    policy_start_date: date
    policy_end_date: date


class Location(BaseModel):
    """
    Structured incident location -- supports both human-readable display
    and programmatic distance/clustering checks (used later by fraud detection
    to flag suspiciously identical incident locations across claims).
    """
    address: str = Field(min_length=1, max_length=300)
    city: str = Field(min_length=1, max_length=100)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class IncidentDetails(BaseModel):
    """What actually happened -- the event the claim is based on."""
    incident_date: date
    location: Location
    description: str = Field(min_length=10, max_length=5000)
    estimated_damage_amount: Decimal = Field(gt=0)


class Claim(BaseModel):
    """
    The full claim record -- the central data object that flows through
    every agent in the pipeline, from intake to settlement.
    """
    claim_id: str = Field(min_length=1)
    status: ClaimStatus = ClaimStatus.SUBMITTED
    claimant: Claimant
    policy: PolicyInfo
    incident: IncidentDetails
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def currency(self) -> Currency:
        """
        Currency is derived from the claimant's country, not stored
        independently -- this guarantees a claim can never have a
        currency that doesn't match its claimant's country.
        """
        return COUNTRY_CURRENCY_MAP[self.claimant.country]
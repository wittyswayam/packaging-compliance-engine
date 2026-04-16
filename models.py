"""
schemas/models.py
=================
Pydantic data contracts for the AI Packaging Compliance Checker.

Defines all request/response schemas with field-level validation,
custom validators, and descriptive examples for OpenAPI documentation.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MaterialType(str, Enum):
    """Supported packaging material categories."""
    PLASTIC = "plastic"
    GLASS = "glass"
    METAL = "metal"
    PAPER = "paper"
    COMPOSITE = "composite"
    BIODEGRADABLE = "biodegradable"


class UsageType(str, Enum):
    """Intended packaging usage context."""
    FOOD = "food"
    PHARMACEUTICAL = "pharmaceutical"
    ELECTRONICS = "electronics"
    COSMETICS = "cosmetics"
    INDUSTRIAL = "industrial"
    RETAIL = "retail"


class Region(str, Enum):
    """Regulatory jurisdiction for compliance evaluation."""
    EU = "EU"
    USA = "USA"
    INDIA = "INDIA"
    CHINA = "CHINA"
    UK = "UK"
    GLOBAL = "GLOBAL"


class ComplianceStatus(str, Enum):
    """Final compliance determination."""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    CONDITIONAL = "CONDITIONAL"
    UNDER_REVIEW = "UNDER_REVIEW"


class RiskLevel(str, Enum):
    """Categorical risk band derived from risk_score."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------

class PackagingCheckRequest(BaseModel):
    """
    Inbound payload for a single packaging compliance check.

    Attributes:
        material_type: Primary material composition of the packaging.
        weight_grams: Total packaging weight in grams (must be positive).
        recyclability_pct: Percentage of material that is recyclable (0–100).
        usage_type: Intended use context of the packaging.
        region: Regulatory region to evaluate compliance against.
        batch_id: Optional client-side batch reference for traceability.
    """

    material_type: MaterialType = Field(
        ...,
        description="Primary material type of the packaging unit.",
        examples=["plastic"],
    )
    weight_grams: float = Field(
        ...,
        gt=0,
        le=50_000,
        description="Packaging weight in grams. Must be between 0 and 50,000 g.",
        examples=[250.0],
    )
    recyclability_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of packaging material that is recyclable.",
        examples=[65.0],
    )
    usage_type: UsageType = Field(
        ...,
        description="Intended use context (food, pharma, electronics, etc.).",
        examples=["food"],
    )
    region: Region = Field(
        ...,
        description="Regulatory jurisdiction to validate compliance against.",
        examples=["EU"],
    )
    batch_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional client-supplied batch reference ID.",
        examples=["BATCH-2024-001"],
    )

    @field_validator("weight_grams")
    @classmethod
    def weight_must_be_finite(cls, v: float) -> float:
        """Reject NaN or infinite weight values."""
        import math
        if not math.isfinite(v):
            raise ValueError("weight_grams must be a finite number.")
        return round(v, 4)

    @field_validator("recyclability_pct")
    @classmethod
    def recyclability_precision(cls, v: float) -> float:
        """Normalise recyclability to two decimal places."""
        return round(v, 2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "material_type": "plastic",
                "weight_grams": 250.0,
                "recyclability_pct": 40.0,
                "usage_type": "food",
                "region": "EU",
                "batch_id": "BATCH-2024-001",
            }
        }
    }


# ---------------------------------------------------------------------------
# Response Sub-schemas
# ---------------------------------------------------------------------------

class ViolationDetail(BaseModel):
    """
    A single regulatory or ML-detected violation.

    Attributes:
        rule_id: Unique identifier for the violated rule.
        description: Human-readable explanation of the violation.
        severity: Impact level — info | warning | error | critical.
        regulation_ref: Optional regulatory document / article reference.
    """

    rule_id: str = Field(..., description="Unique rule identifier.", examples=["EU_PLASTIC_001"])
    description: str = Field(..., description="Plain-language violation description.")
    severity: str = Field(..., pattern=r"^(info|warning|error|critical)$")
    regulation_ref: Optional[str] = Field(
        default=None,
        description="Regulatory article or directive reference.",
        examples=["EU Regulation 10/2011 — Art. 3"],
    )


class MitigationStep(BaseModel):
    """
    A single recommended corrective action.

    Attributes:
        step_number: Ordered position in the mitigation plan.
        action: Concrete action the manufacturer should take.
        estimated_cost_usd: Rough remediation cost estimate (optional).
        priority: Urgency level — immediate | short_term | long_term.
    """

    step_number: int = Field(..., ge=1)
    action: str = Field(..., description="Recommended corrective action.")
    estimated_cost_usd: Optional[float] = Field(default=None, ge=0)
    priority: str = Field(..., pattern=r"^(immediate|short_term|long_term)$")


# ---------------------------------------------------------------------------
# Response Schema
# ---------------------------------------------------------------------------

class PackagingCheckResponse(BaseModel):
    """
    Full compliance evaluation result for a packaging unit.

    Attributes:
        status: Final compliance determination.
        risk_score: Continuous risk score in [0.0, 1.0]; higher = riskier.
        risk_level: Categorical band derived from risk_score.
        violation_details: List of individual rule/ML violations found.
        mitigation_plan: Ordered list of recommended corrective actions.
        batch_id: Echoed from the request for correlation.
        evaluated_at: ISO-8601 UTC timestamp of evaluation.
    """

    status: ComplianceStatus
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    violation_details: List[ViolationDetail] = Field(default_factory=list)
    mitigation_plan: List[MitigationStep] = Field(default_factory=list)
    batch_id: Optional[str] = None
    evaluated_at: str = Field(..., description="ISO-8601 UTC evaluation timestamp.")

    @model_validator(mode="after")
    def validate_risk_band(self) -> "PackagingCheckResponse":
        """Ensure risk_level is consistent with risk_score."""
        score = self.risk_score
        expected: RiskLevel
        if score < 0.25:
            expected = RiskLevel.LOW
        elif score < 0.55:
            expected = RiskLevel.MEDIUM
        elif score < 0.80:
            expected = RiskLevel.HIGH
        else:
            expected = RiskLevel.CRITICAL

        if self.risk_level != expected:
            logger.warning(
                "risk_level %s does not match risk_score %.3f; correcting to %s.",
                self.risk_level,
                score,
                expected,
            )
            self.risk_level = expected
        return self


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    sample_request = PackagingCheckRequest(
        material_type=MaterialType.PLASTIC,
        weight_grams=250.0,
        recyclability_pct=40.0,
        usage_type=UsageType.FOOD,
        region=Region.EU,
        batch_id="BATCH-2024-001",
    )
    logger.info("Request schema OK: %s", sample_request.model_dump_json(indent=2))

    sample_response = PackagingCheckResponse(
        status=ComplianceStatus.NON_COMPLIANT,
        risk_score=0.72,
        risk_level=RiskLevel.HIGH,
        violation_details=[
            ViolationDetail(
                rule_id="EU_PLASTIC_001",
                description="Recyclability below EU 2030 target of 55%.",
                severity="error",
                regulation_ref="EU Regulation 10/2011",
            )
        ],
        mitigation_plan=[
            MitigationStep(
                step_number=1,
                action="Switch to mono-material PP to increase recyclability above 55%.",
                estimated_cost_usd=4200.0,
                priority="immediate",
            )
        ],
        batch_id="BATCH-2024-001",
        evaluated_at="2024-01-15T10:30:00Z",
    )
    logger.info("Response schema OK: %s", sample_response.model_dump_json(indent=2))

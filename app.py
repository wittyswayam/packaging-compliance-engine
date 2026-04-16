# MIT License
#
# Copyright (c) 2024 AI Packaging Compliance Checker
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""
api/app.py
==========
FastAPI service layer for the AI Packaging Compliance Checker.

Endpoints:
    POST /v1/compliance/check   — Single-item compliance evaluation.
    POST /v1/compliance/batch   — Batch evaluation (≤50 items).
    GET  /v1/compliance/regions — List supported regulatory regions.
    GET  /healthz               — Liveness probe for container orchestration.

Architecture notes:
    - Lifespan event handler warms up the ML model singleton at startup.
    - Dependency injection supplies a pre-loaded model to route handlers.
    - Global exception handlers translate domain errors to RFC 7807 responses.
    - Structured JSON logging via the stdlib logging module.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Internal imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas.models import (
    ComplianceStatus,
    MitigationStep,
    PackagingCheckRequest,
    PackagingCheckResponse,
    RiskLevel,
    ViolationDetail,
)
from src.compliance_model import (
    ComplianceModelSingleton,
    ModelNotFoundError,
    ModelTrainingError,
    get_ml_risk_score,
)
from src.rules_engine import (
    InvalidPayloadError,
    RuleEngineFactory,
    UnknownRegionError,
    run_rules_check,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Mitigation plan generator
# ---------------------------------------------------------------------------

def _build_mitigation_plan(
    material_type: str,
    recyclability_pct: float,
    usage_type: str,
    region: str,
    risk_score: float,
) -> List[MitigationStep]:
    """Derive an ordered mitigation plan from payload characteristics.

    Args:
        material_type: Primary packaging material.
        recyclability_pct: Current recyclable fraction 0–100.
        usage_type: Intended packaging use.
        region: Regulatory jurisdiction.
        risk_score: Combined ML+rules risk score.

    Returns:
        List of :class:`MitigationStep` ordered by priority.
    """
    steps: List[MitigationStep] = []
    step_n = 1

    if material_type == "plastic" and region in ("EU", "UK", "INDIA"):
        steps.append(MitigationStep(
            step_number=step_n,
            action=(
                "Replace single-use plastic with certified compostable or "
                "mono-material PP/PE to comply with single-use plastics directives."
            ),
            estimated_cost_usd=8500.0,
            priority="immediate",
        ))
        step_n += 1

    if recyclability_pct < 55.0:
        steps.append(MitigationStep(
            step_number=step_n,
            action=(
                f"Redesign packaging to achieve ≥55% recyclability "
                f"(current: {recyclability_pct:.1f}%). Consider mono-material construction "
                "and removal of non-recyclable adhesives/coatings."
            ),
            estimated_cost_usd=4200.0,
            priority="immediate" if recyclability_pct < 30 else "short_term",
        ))
        step_n += 1

    if usage_type == "pharmaceutical" and recyclability_pct < 100.0:
        steps.append(MitigationStep(
            step_number=step_n,
            action=(
                "Engage a certified pharmaceutical packaging supplier to achieve "
                "100% recyclable blister/carton materials required for EU/UK pharma compliance."
            ),
            estimated_cost_usd=15000.0,
            priority="immediate",
        ))
        step_n += 1

    if risk_score >= 0.55:
        steps.append(MitigationStep(
            step_number=step_n,
            action=(
                "Commission a third-party lifecycle assessment (LCA) to quantify "
                "environmental impact and identify further reduction opportunities."
            ),
            estimated_cost_usd=3000.0,
            priority="short_term",
        ))
        step_n += 1

    steps.append(MitigationStep(
        step_number=step_n,
        action=(
            "Implement a Digital Product Passport (DPP) to document material composition, "
            "recyclability, and compliance status across the supply chain."
        ),
        estimated_cost_usd=1200.0,
        priority="long_term",
    ))

    return steps


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------

def _combine_scores(rule_score: float, ml_score: float) -> float:
    """Weighted average: rules (40%) + ML (60%)."""
    return round(0.40 * rule_score + 0.60 * ml_score, 4)


def _score_to_risk_level(score: float) -> RiskLevel:
    if score < 0.25:
        return RiskLevel.LOW
    elif score < 0.55:
        return RiskLevel.MEDIUM
    elif score < 0.80:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def _derive_status(is_rule_compliant: bool, risk_level: RiskLevel) -> ComplianceStatus:
    if not is_rule_compliant:
        return ComplianceStatus.NON_COMPLIANT
    if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        return ComplianceStatus.CONDITIONAL
    return ComplianceStatus.COMPLIANT


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def _evaluate_single(req: PackagingCheckRequest) -> PackagingCheckResponse:
    """Run the full hybrid evaluation pipeline for one request.

    Args:
        req: Validated :class:`PackagingCheckRequest`.

    Returns:
        :class:`PackagingCheckResponse` with full compliance details.
    """
    t0 = time.perf_counter()

    # 1. Rule-based check
    rule_result, evaluated_at = run_rules_check(
        material_type=req.material_type.value,
        weight_grams=req.weight_grams,
        recyclability_pct=req.recyclability_pct,
        usage_type=req.usage_type.value,
        region=req.region.value,
    )

    # 2. ML risk score
    ml_score = get_ml_risk_score(
        material_type=req.material_type.value,
        weight_grams=req.weight_grams,
        recyclability_pct=req.recyclability_pct,
        usage_type=req.usage_type.value,
        region=req.region.value,
        auto_train=True,
    )

    # 3. Combine scores
    combined_score = _combine_scores(rule_result.rule_score, ml_score)
    risk_level = _score_to_risk_level(combined_score)
    status = _derive_status(rule_result.is_compliant, risk_level)

    # 4. Map rule violations to response schema
    violation_details = [
        ViolationDetail(
            rule_id=v.rule_id,
            description=v.description,
            severity=v.severity,
            regulation_ref=v.regulation_ref,
        )
        for v in rule_result.violations
    ]

    # 5. Build mitigation plan
    mitigation_plan = _build_mitigation_plan(
        material_type=req.material_type.value,
        recyclability_pct=req.recyclability_pct,
        usage_type=req.usage_type.value,
        region=req.region.value,
        risk_score=combined_score,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Evaluation complete | status=%s score=%.4f latency=%.1fms batch_id=%s",
        status.value, combined_score, elapsed_ms, req.batch_id,
    )

    return PackagingCheckResponse(
        status=status,
        risk_score=combined_score,
        risk_level=risk_level,
        violation_details=violation_details,
        mitigation_plan=mitigation_plan,
        batch_id=req.batch_id,
        evaluated_at=evaluated_at,
    )


# ---------------------------------------------------------------------------
# Lifespan: warm-up ML model at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan handler — warms ML model before accepting traffic."""
    logger.info("Application startup: warming ML compliance model…")
    try:
        model = ComplianceModelSingleton()
        try:
            model.load()
        except ModelNotFoundError:
            logger.warning("No saved model found — training fresh model (first-run).")
            model.train_and_save()
        logger.info("ML model ready.")
    except Exception as exc:
        logger.error("Model warm-up failed: %s", exc, exc_info=True)
    yield
    logger.info("Application shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Packaging Compliance Checker",
    description=(
        "Hybrid rule-based + ML API for evaluating packaging compliance across "
        "multiple regulatory jurisdictions (EU, USA, India, UK, China, Global)."
    ),
    version="1.0.0",
    contact={"name": "Compliance Team", "email": "compliance@example.com"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(UnknownRegionError)
async def unknown_region_handler(request: Request, exc: UnknownRegionError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"type": "UnknownRegionError", "detail": str(exc)},
    )


@app.exception_handler(InvalidPayloadError)
async def invalid_payload_handler(request: Request, exc: InvalidPayloadError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"type": "InvalidPayloadError", "detail": str(exc)},
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"type": "ModelNotFoundError", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz", tags=["Ops"], summary="Liveness probe")
async def healthz() -> Dict[str, str]:
    """Return service health status for container orchestration probes."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get(
    "/v1/compliance/regions",
    tags=["Compliance"],
    summary="List supported regulatory regions",
)
async def list_regions() -> Dict[str, list]:
    """Return all regulatory regions that the engine supports."""
    return {
        "regions": list(RuleEngineFactory._registry.keys()),
        "count": len(RuleEngineFactory._registry),
    }


@app.post(
    "/v1/compliance/check",
    response_model=PackagingCheckResponse,
    status_code=status.HTTP_200_OK,
    tags=["Compliance"],
    summary="Single packaging compliance check",
)
async def compliance_check(req: PackagingCheckRequest) -> PackagingCheckResponse:
    """Evaluate a single packaging unit against the specified regulatory region.

    Returns a full compliance report including risk score, violations, and
    an ordered mitigation plan.
    """
    try:
        return _evaluate_single(req)
    except (UnknownRegionError, InvalidPayloadError):
        raise
    except Exception as exc:
        logger.error("Unexpected error during compliance check: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please retry or contact support.",
        )


@app.post(
    "/v1/compliance/batch",
    response_model=List[PackagingCheckResponse],
    status_code=status.HTTP_200_OK,
    tags=["Compliance"],
    summary="Batch packaging compliance check (≤50 items)",
)
async def compliance_batch(
    requests: List[PackagingCheckRequest],
) -> List[PackagingCheckResponse]:
    """Evaluate up to 50 packaging units in a single call.

    Each item is evaluated independently; partial failures are surfaced inline.
    """
    if len(requests) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds maximum of 50 items.",
        )

    results: List[PackagingCheckResponse] = []
    for req in requests:
        try:
            results.append(_evaluate_single(req))
        except Exception as exc:
            logger.error("Batch item %s failed: %s", req.batch_id, exc)
            # Surface partial failure as a UNDER_REVIEW result
            results.append(
                PackagingCheckResponse(
                    status=ComplianceStatus.UNDER_REVIEW,
                    risk_score=0.5,
                    risk_level=RiskLevel.MEDIUM,
                    violation_details=[],
                    mitigation_plan=[],
                    batch_id=req.batch_id,
                    evaluated_at=datetime.now(timezone.utc).isoformat(),
                )
            )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

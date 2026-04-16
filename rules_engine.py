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
src/rules_engine.py
===================
Hybrid rule-based compliance engine for packaging regulation checks.

Architecture:
    - Strategy Pattern  : RegionStrategy abstracts per-jurisdiction rule sets.
    - Factory Pattern   : RuleEngineFactory instantiates the correct strategy.
    - Singleton Pattern : RuleConfig loads regulation thresholds exactly once.

Supported regions: EU, USA, INDIA, CHINA, UK, GLOBAL
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class RulesEngineError(Exception):
    """Base exception for all Rules Engine failures."""


class UnknownRegionError(RulesEngineError):
    """Raised when an unsupported regulatory region is requested."""


class InvalidPayloadError(RulesEngineError):
    """Raised when the input payload fails pre-validation checks."""


class RuleConfigLoadError(RulesEngineError):
    """Raised when the singleton config cannot initialise regulation thresholds."""


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleViolation:
    """Immutable record of a single rule violation.

    Attributes:
        rule_id: Unique identifier for the violated rule.
        description: Human-readable violation explanation.
        severity: One of info | warning | error | critical.
        regulation_ref: Optional regulatory article or directive.
    """

    rule_id: str
    description: str
    severity: str  # info | warning | error | critical
    regulation_ref: Optional[str] = None


@dataclass
class RuleEngineResult:
    """Aggregated output from a single compliance evaluation pass.

    Attributes:
        is_compliant: True only when zero error/critical violations exist.
        rule_score: Normalised penalty score in [0.0, 1.0].
        violations: All detected violations, ordered by severity.
    """

    is_compliant: bool = True
    rule_score: float = 0.0
    violations: List[RuleViolation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Singleton: RuleConfig
# ---------------------------------------------------------------------------

class RuleConfig:
    """Thread-safe singleton that holds all regulatory thresholds.

    Loads once on first access; subsequent calls return the same instance.
    Thresholds are hard-coded here but could be externalised to a YAML/JSON
    file referenced via pathlib in a production deployment.
    """

    _instance: ClassVar[Optional["RuleConfig"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    # Recyclability minimums per region (percentage)
    RECYCLABILITY_FLOOR: ClassVar[Dict[str, float]] = {
        "EU": 55.0,
        "USA": 30.0,
        "INDIA": 20.0,
        "CHINA": 25.0,
        "UK": 50.0,
        "GLOBAL": 20.0,
    }

    # Single-use plastic ban regions
    PLASTIC_BANNED_REGIONS: ClassVar[List[str]] = ["EU", "UK", "INDIA"]

    # Maximum food-contact packaging weight (grams) — lightweight mandates
    FOOD_WEIGHT_CAP_G: ClassVar[Dict[str, float]] = {
        "EU": 500.0,
        "USA": 1000.0,
        "INDIA": 200.0,
        "CHINA": 600.0,
        "UK": 500.0,
        "GLOBAL": 1000.0,
    }

    # Pharmaceutical packaging must be 100 % recyclable in these regions
    PHARMA_FULL_RECYCLABILITY_REGIONS: ClassVar[List[str]] = ["EU", "UK"]

    def __new__(cls) -> "RuleConfig":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("RuleConfig: initialising singleton instance.")
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance


# ---------------------------------------------------------------------------
# Strategy: Abstract Base
# ---------------------------------------------------------------------------

class RegionStrategy(ABC):
    """Abstract base class for per-jurisdiction compliance strategies.

    Each concrete strategy encapsulates the regulatory logic for one region,
    evaluating an incoming payload and returning a :class:`RuleEngineResult`.
    """

    def __init__(self) -> None:
        self._config = RuleConfig()

    @abstractmethod
    def evaluate(
        self,
        material_type: str,
        weight_grams: float,
        recyclability_pct: float,
        usage_type: str,
    ) -> RuleEngineResult:
        """Run all applicable rules and return an aggregated result.

        Args:
            material_type: Primary packaging material (e.g. "plastic").
            weight_grams: Packaging weight in grams.
            recyclability_pct: Fraction recyclable, expressed 0–100.
            usage_type: Intended use (food, pharmaceutical, etc.).

        Returns:
            RuleEngineResult with violations and aggregated score.
        """

    # ------------------------------------------------------------------
    # Shared helper rules (available to all strategies)
    # ------------------------------------------------------------------

    def _check_recyclability(
        self, recyclability_pct: float, region: str
    ) -> Optional[RuleViolation]:
        """Return a violation if recyclability is below the regional floor."""
        floor = self._config.RECYCLABILITY_FLOOR.get(region, 20.0)
        if recyclability_pct < floor:
            return RuleViolation(
                rule_id=f"{region}_RECYCLE_001",
                description=(
                    f"Recyclability {recyclability_pct:.1f}% is below the "
                    f"{region} regulatory minimum of {floor:.1f}%."
                ),
                severity="error",
                regulation_ref=f"{region} Packaging & Packaging Waste Directive",
            )
        return None

    def _check_food_weight(
        self, weight_grams: float, usage_type: str, region: str
    ) -> Optional[RuleViolation]:
        """Return a violation if a food-use package exceeds the weight cap."""
        if usage_type != "food":
            return None
        cap = self._config.FOOD_WEIGHT_CAP_G.get(region, 1000.0)
        if weight_grams > cap:
            return RuleViolation(
                rule_id=f"{region}_FOOD_WEIGHT_001",
                description=(
                    f"Food packaging weight {weight_grams:.1f} g exceeds "
                    f"the {region} cap of {cap:.1f} g."
                ),
                severity="warning",
                regulation_ref=f"{region} Food Contact Materials Regulation",
            )
        return None

    def _check_plastic_ban(
        self, material_type: str, region: str
    ) -> Optional[RuleViolation]:
        """Return a critical violation if single-use plastic is banned in region."""
        if (
            material_type == "plastic"
            and region in self._config.PLASTIC_BANNED_REGIONS
        ):
            return RuleViolation(
                rule_id=f"{region}_PLASTIC_BAN_001",
                description=(
                    f"Single-use plastic packaging is restricted or banned in {region}."
                ),
                severity="critical",
                regulation_ref=f"{region} Single-Use Plastics Directive",
            )
        return None

    def _check_pharma_recyclability(
        self, recyclability_pct: float, usage_type: str, region: str
    ) -> Optional[RuleViolation]:
        """Pharmaceutical packaging must be 100 % recyclable in select regions."""
        if (
            usage_type == "pharmaceutical"
            and region in self._config.PHARMA_FULL_RECYCLABILITY_REGIONS
            and recyclability_pct < 100.0
        ):
            return RuleViolation(
                rule_id=f"{region}_PHARMA_RECYCLE_001",
                description=(
                    f"Pharmaceutical packaging must be 100% recyclable in {region}; "
                    f"current: {recyclability_pct:.1f}%."
                ),
                severity="critical",
                regulation_ref=f"{region} Pharmaceutical Packaging Sustainability Act",
            )
        return None

    @staticmethod
    def _score_from_violations(violations: List[RuleViolation]) -> float:
        """Compute a normalised penalty score [0, 1] from violation severities.

        Severity weights: info=0.05, warning=0.15, error=0.35, critical=0.60
        Score is clamped to 1.0 and represents the *rule-based* risk contribution.
        """
        weights = {"info": 0.05, "warning": 0.15, "error": 0.35, "critical": 0.60}
        raw = sum(weights.get(v.severity, 0.0) for v in violations)
        return min(raw, 1.0)


# ---------------------------------------------------------------------------
# Concrete Strategies
# ---------------------------------------------------------------------------

class EUStrategy(RegionStrategy):
    """EU-specific compliance strategy (PPWD 94/62/EC, SUP Directive 2019/904)."""

    def evaluate(
        self,
        material_type: str,
        weight_grams: float,
        recyclability_pct: float,
        usage_type: str,
    ) -> RuleEngineResult:
        violations: List[RuleViolation] = []

        for check_fn, args in [
            (self._check_plastic_ban, (material_type, "EU")),
            (self._check_recyclability, (recyclability_pct, "EU")),
            (self._check_food_weight, (weight_grams, usage_type, "EU")),
            (self._check_pharma_recyclability, (recyclability_pct, usage_type, "EU")),
        ]:
            violation = check_fn(*args)
            if violation:
                violations.append(violation)

        # EU-specific: composite materials must disclose multi-layer composition
        if material_type == "composite":
            violations.append(
                RuleViolation(
                    rule_id="EU_COMPOSITE_001",
                    description="Composite packaging requires multi-layer material disclosure under EU PPWD.",
                    severity="warning",
                    regulation_ref="EU PPWD 94/62/EC — Art. 11",
                )
            )

        score = self._score_from_violations(violations)
        is_compliant = not any(v.severity in ("error", "critical") for v in violations)
        return RuleEngineResult(is_compliant=is_compliant, rule_score=score, violations=violations)


class USAStrategy(RegionStrategy):
    """USA-specific compliance strategy (EPA guidelines, state-level mandates)."""

    def evaluate(
        self,
        material_type: str,
        weight_grams: float,
        recyclability_pct: float,
        usage_type: str,
    ) -> RuleEngineResult:
        violations: List[RuleViolation] = []

        for check_fn, args in [
            (self._check_recyclability, (recyclability_pct, "USA")),
            (self._check_food_weight, (weight_grams, usage_type, "USA")),
            (self._check_pharma_recyclability, (recyclability_pct, usage_type, "USA")),
        ]:
            violation = check_fn(*args)
            if violation:
                violations.append(violation)

        # California-specific: extended producer responsibility surcharge flag
        if material_type == "plastic" and recyclability_pct < 50.0:
            violations.append(
                RuleViolation(
                    rule_id="USA_CA_EPR_001",
                    description="Plastic packaging <50% recyclable may incur CA SB 54 EPR surcharges.",
                    severity="warning",
                    regulation_ref="California SB 54 — Plastic Pollution Prevention Act",
                )
            )

        score = self._score_from_violations(violations)
        is_compliant = not any(v.severity in ("error", "critical") for v in violations)
        return RuleEngineResult(is_compliant=is_compliant, rule_score=score, violations=violations)


class IndiaStrategy(RegionStrategy):
    """India-specific compliance strategy (Plastic Waste Management Rules 2016, amended 2022)."""

    def evaluate(
        self,
        material_type: str,
        weight_grams: float,
        recyclability_pct: float,
        usage_type: str,
    ) -> RuleEngineResult:
        violations: List[RuleViolation] = []

        for check_fn, args in [
            (self._check_plastic_ban, (material_type, "INDIA")),
            (self._check_recyclability, (recyclability_pct, "INDIA")),
            (self._check_food_weight, (weight_grams, usage_type, "INDIA")),
        ]:
            violation = check_fn(*args)
            if violation:
                violations.append(violation)

        # India: plastic thickness < 120 microns banned since 2022
        if material_type == "plastic" and weight_grams < 5.0:
            violations.append(
                RuleViolation(
                    rule_id="INDIA_THIN_PLASTIC_001",
                    description="Lightweight plastic (<5g) likely violates 120-micron thickness ban.",
                    severity="error",
                    regulation_ref="MoEFCC Plastic Waste Management (Amendment) Rules 2022",
                )
            )

        score = self._score_from_violations(violations)
        is_compliant = not any(v.severity in ("error", "critical") for v in violations)
        return RuleEngineResult(is_compliant=is_compliant, rule_score=score, violations=violations)


class GlobalStrategy(RegionStrategy):
    """Baseline GLOBAL strategy using minimum international standards."""

    def evaluate(
        self,
        material_type: str,
        weight_grams: float,
        recyclability_pct: float,
        usage_type: str,
    ) -> RuleEngineResult:
        violations: List[RuleViolation] = []
        for check_fn, args in [
            (self._check_recyclability, (recyclability_pct, "GLOBAL")),
            (self._check_food_weight, (weight_grams, usage_type, "GLOBAL")),
        ]:
            violation = check_fn(*args)
            if violation:
                violations.append(violation)

        score = self._score_from_violations(violations)
        is_compliant = not any(v.severity in ("error", "critical") for v in violations)
        return RuleEngineResult(is_compliant=is_compliant, rule_score=score, violations=violations)


# ---------------------------------------------------------------------------
# Factory: RuleEngineFactory
# ---------------------------------------------------------------------------

class RuleEngineFactory:
    """Factory that maps a region string to the correct :class:`RegionStrategy`.

    Usage::

        strategy = RuleEngineFactory.get_strategy("EU")
        result = strategy.evaluate(material, weight, recyclability, usage)
    """

    _registry: ClassVar[Dict[str, type]] = {
        "EU": EUStrategy,
        "USA": USAStrategy,
        "INDIA": IndiaStrategy,
        "UK": EUStrategy,       # UK mirrors EU post-Brexit baseline
        "CHINA": GlobalStrategy, # China uses global baseline (extendable)
        "GLOBAL": GlobalStrategy,
    }

    @classmethod
    def get_strategy(cls, region: str) -> RegionStrategy:
        """Return an instantiated strategy for the given region.

        Args:
            region: Regulatory jurisdiction string (case-insensitive).

        Returns:
            An initialised :class:`RegionStrategy` instance.

        Raises:
            UnknownRegionError: If the region has no registered strategy.
        """
        key = region.upper()
        strategy_cls = cls._registry.get(key)
        if strategy_cls is None:
            raise UnknownRegionError(
                f"No compliance strategy registered for region '{region}'. "
                f"Supported: {list(cls._registry.keys())}"
            )
        logger.debug("RuleEngineFactory: resolved strategy %s for region %s.", strategy_cls.__name__, key)
        return strategy_cls()

    @classmethod
    def register_strategy(cls, region: str, strategy_cls: type) -> None:
        """Register a custom strategy at runtime (extension point).

        Args:
            region: Jurisdiction key to register.
            strategy_cls: Class implementing :class:`RegionStrategy`.
        """
        if not issubclass(strategy_cls, RegionStrategy):
            raise TypeError(f"{strategy_cls} must subclass RegionStrategy.")
        cls._registry[region.upper()] = strategy_cls
        logger.info("RuleEngineFactory: registered custom strategy for %s.", region)


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

def run_rules_check(
    material_type: str,
    weight_grams: float,
    recyclability_pct: float,
    usage_type: str,
    region: str,
) -> Tuple[RuleEngineResult, str]:
    """Top-level helper that wires factory + strategy and returns a result.

    Args:
        material_type: Primary packaging material.
        weight_grams: Packaging weight in grams.
        recyclability_pct: Recyclable fraction expressed 0–100.
        usage_type: Intended packaging use.
        region: Regulatory jurisdiction.

    Returns:
        Tuple of (RuleEngineResult, ISO-8601 UTC timestamp).

    Raises:
        InvalidPayloadError: If numeric fields are out of acceptable range.
        UnknownRegionError: If the region has no registered strategy.
    """
    if not (0.0 <= recyclability_pct <= 100.0):
        raise InvalidPayloadError("recyclability_pct must be in [0, 100].")
    if weight_grams <= 0:
        raise InvalidPayloadError("weight_grams must be positive.")

    strategy = RuleEngineFactory.get_strategy(region)
    result = strategy.evaluate(material_type, weight_grams, recyclability_pct, usage_type)
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Rules check complete | region=%s material=%s compliant=%s score=%.3f violations=%d",
        region, material_type, result.is_compliant, result.rule_score, len(result.violations),
    )
    return result, timestamp


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(name)s | %(message)s")

    test_cases = [
        ("plastic", 250.0, 40.0, "food", "EU"),
        ("glass", 300.0, 95.0, "food", "USA"),
        ("plastic", 3.5, 15.0, "retail", "INDIA"),
        ("biodegradable", 150.0, 100.0, "pharmaceutical", "EU"),
        ("composite", 400.0, 55.0, "electronics", "GLOBAL"),
    ]

    for args in test_cases:
        result, ts = run_rules_check(*args)
        print(f"\n{'='*60}")
        print(f"Region={args[4]} | Material={args[0]} | Usage={args[3]}")
        print(f"  Compliant : {result.is_compliant}")
        print(f"  Rule Score: {result.rule_score:.3f}")
        print(f"  Violations: {len(result.violations)}")
        for v in result.violations:
            print(f"    [{v.severity.upper():8s}] {v.rule_id}: {v.description}")

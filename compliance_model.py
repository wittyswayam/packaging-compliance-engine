"""
src/compliance_model.py
=======================
ML-based risk scoring layer for the AI Packaging Compliance Checker.

Architecture:
    - Trains a calibrated GradientBoostingClassifier on synthetic labelled data.
    - Serialises the pipeline to disk via joblib (models/ directory).
    - Exposes a clean predict() interface that returns a continuous risk score.
    - Singleton pattern guards the loaded model to prevent redundant I/O.

Risk score semantics:
    0.0 – 0.24  → LOW
    0.25 – 0.54 → MEDIUM
    0.55 – 0.79 → HIGH
    0.80 – 1.0  → CRITICAL
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _REPO_ROOT / "models"
_MODEL_PATH = _MODEL_DIR / "compliance_model.joblib"
_ENCODER_PATH = _MODEL_DIR / "label_encoders.joblib"


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class ModelNotFoundError(FileNotFoundError):
    """Raised when the serialised model artefact is missing from disk."""


class ModelInferenceError(RuntimeError):
    """Raised when a prediction call fails due to unexpected input."""


class ModelTrainingError(RuntimeError):
    """Raised when model training encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

# Canonical ordinal mappings — must match training data exactly
_MATERIAL_MAP: Dict[str, int] = {
    "plastic": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "composite": 4,
    "biodegradable": 5,
}

_USAGE_MAP: Dict[str, int] = {
    "food": 0,
    "pharmaceutical": 1,
    "electronics": 2,
    "cosmetics": 3,
    "industrial": 4,
    "retail": 5,
}

_REGION_MAP: Dict[str, int] = {
    "EU": 0,
    "USA": 1,
    "INDIA": 2,
    "CHINA": 3,
    "UK": 4,
    "GLOBAL": 5,
}


def _encode_features(
    material_type: str,
    weight_grams: float,
    recyclability_pct: float,
    usage_type: str,
    region: str,
) -> np.ndarray:
    """Encode raw payload fields into a 1-D numeric feature vector.

    Args:
        material_type: Primary packaging material string.
        weight_grams: Packaging weight in grams.
        recyclability_pct: Recyclable fraction 0–100.
        usage_type: Intended use string.
        region: Regulatory jurisdiction string.

    Returns:
        Shape-(1, 6) float32 ndarray ready for model input.

    Raises:
        ModelInferenceError: If any categorical field is unrecognised.
    """
    try:
        mat_enc = _MATERIAL_MAP[material_type.lower()]
    except KeyError:
        raise ModelInferenceError(f"Unknown material_type: '{material_type}'.")

    try:
        use_enc = _USAGE_MAP[usage_type.lower()]
    except KeyError:
        raise ModelInferenceError(f"Unknown usage_type: '{usage_type}'.")

    try:
        reg_enc = _REGION_MAP[region.upper()]
    except KeyError:
        raise ModelInferenceError(f"Unknown region: '{region}'.")

    # Derived features
    weight_log = np.log1p(weight_grams)                   # log-scale weight
    recycle_deficit = max(0.0, 55.0 - recyclability_pct)  # EU baseline gap

    features = np.array(
        [[mat_enc, weight_log, recyclability_pct, use_enc, reg_enc, recycle_deficit]],
        dtype=np.float32,
    )
    return features


# ---------------------------------------------------------------------------
# Synthetic training data generator
# ---------------------------------------------------------------------------

def _generate_synthetic_data(n_samples: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate labelled synthetic packaging data for initial model training.

    Labels encode risk severity:
        0 = LOW, 1 = MEDIUM, 2 = HIGH, 3 = CRITICAL

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X: feature matrix, y: integer label vector).
    """
    rng = np.random.default_rng(seed)

    materials = rng.integers(0, len(_MATERIAL_MAP), n_samples)
    weights_log = rng.uniform(np.log1p(10), np.log1p(5000), n_samples)
    recyclability = rng.uniform(0, 100, n_samples)
    usages = rng.integers(0, len(_USAGE_MAP), n_samples)
    regions = rng.integers(0, len(_REGION_MAP), n_samples)
    recycle_deficit = np.maximum(0.0, 55.0 - recyclability)

    X = np.stack([materials, weights_log, recyclability, usages, regions, recycle_deficit], axis=1).astype(np.float32)

    # Rule-based heuristic labels (simulates ground truth)
    labels = np.zeros(n_samples, dtype=int)

    # Plastic in banned regions → HIGH or CRITICAL
    plastic_mask = materials == 0
    banned_region_mask = np.isin(regions, [0, 2, 4])  # EU, INDIA, UK
    labels[plastic_mask & banned_region_mask] = 3

    # Low recyclability → escalate risk
    labels[recyclability < 20] = np.maximum(labels[recyclability < 20], 3)
    labels[(recyclability >= 20) & (recyclability < 40)] = np.maximum(
        labels[(recyclability >= 20) & (recyclability < 40)], 2
    )
    labels[(recyclability >= 40) & (recyclability < 55)] = np.maximum(
        labels[(recyclability >= 40) & (recyclability < 55)], 1
    )

    # Pharmaceutical in EU/UK with <100% recyclability → HIGH
    pharma_mask = usages == 1
    strict_region = np.isin(regions, [0, 4])
    labels[pharma_mask & strict_region & (recyclability < 100)] = np.maximum(
        labels[pharma_mask & strict_region & (recyclability < 100)], 2
    )

    # Add noise (5 % random label flip to prevent over-fitting to rules)
    noise_idx = rng.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    labels[noise_idx] = rng.integers(0, 4, len(noise_idx))

    return X, labels


# ---------------------------------------------------------------------------
# Singleton: ComplianceModelSingleton
# ---------------------------------------------------------------------------

class ComplianceModelSingleton:
    """Thread-safe singleton wrapping the trained compliance ML pipeline.

    The pipeline (StandardScaler → GradientBoostingClassifier with Platt
    calibration) is loaded from disk on first access.  If no saved model
    exists, :meth:`train_and_save` must be called first.
    """

    _instance: ClassVar[Optional["ComplianceModelSingleton"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _pipeline: Optional[Pipeline]

    def __new__(cls) -> "ComplianceModelSingleton":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._pipeline = None
                    cls._instance = obj
                    logger.info("ComplianceModelSingleton: instance created.")
        return cls._instance

    def load(self) -> None:
        """Load the serialised pipeline from disk into memory.

        Raises:
            ModelNotFoundError: If the .joblib artefact does not exist.
        """
        if not _MODEL_PATH.exists():
            raise ModelNotFoundError(
                f"Model artefact not found at {_MODEL_PATH}. "
                "Run ComplianceModelSingleton().train_and_save() first."
            )
        self._pipeline = joblib.load(_MODEL_PATH)
        logger.info("ComplianceModelSingleton: pipeline loaded from %s.", _MODEL_PATH)

    def train_and_save(self, n_samples: int = 2000) -> None:
        """Train a fresh pipeline on synthetic data and serialise to disk.

        Args:
            n_samples: Training set size.

        Raises:
            ModelTrainingError: On any training-phase exception.
        """
        try:
            logger.info("ComplianceModelSingleton: generating %d synthetic samples…", n_samples)
            X, y = _generate_synthetic_data(n_samples)

            base_clf = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            calibrated_clf = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", calibrated_clf),
            ])

            logger.info("ComplianceModelSingleton: training pipeline…")
            pipeline.fit(X, y)

            _MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, _MODEL_PATH)
            self._pipeline = pipeline
            logger.info("ComplianceModelSingleton: pipeline saved to %s.", _MODEL_PATH)

        except Exception as exc:
            raise ModelTrainingError(f"Training failed: {exc}") from exc

    def predict_risk_score(
        self,
        material_type: str,
        weight_grams: float,
        recyclability_pct: float,
        usage_type: str,
        region: str,
    ) -> float:
        """Return a continuous risk score in [0.0, 1.0].

        The score is derived from the probability mass assigned to the two
        highest-risk classes (HIGH + CRITICAL) by the calibrated classifier.

        Args:
            material_type: Primary packaging material.
            weight_grams: Packaging weight in grams.
            recyclability_pct: Recyclable fraction 0–100.
            usage_type: Intended use context.
            region: Regulatory jurisdiction.

        Returns:
            Float in [0.0, 1.0]; higher = greater risk.

        Raises:
            ModelNotFoundError: If the pipeline has not been loaded.
            ModelInferenceError: If feature encoding fails.
        """
        if self._pipeline is None:
            raise ModelNotFoundError("Pipeline not loaded. Call .load() or .train_and_save() first.")

        features = _encode_features(material_type, weight_grams, recyclability_pct, usage_type, region)
        proba = self._pipeline.predict_proba(features)[0]  # shape: (4,) for LOW/MED/HIGH/CRIT

        # Weight HIGH and CRITICAL probabilities toward the score
        # classes: 0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL
        score = 0.0 * proba[0] + 0.25 * proba[1] + 0.65 * proba[2] + 1.0 * proba[3]
        score = float(np.clip(score, 0.0, 1.0))

        logger.debug(
            "ML inference | material=%s region=%s recyclability=%.1f → score=%.4f",
            material_type, region, recyclability_pct, score,
        )
        return score


# ---------------------------------------------------------------------------
# Convenience public API
# ---------------------------------------------------------------------------

def get_ml_risk_score(
    material_type: str,
    weight_grams: float,
    recyclability_pct: float,
    usage_type: str,
    region: str,
    auto_train: bool = True,
) -> float:
    """Load (or train) the singleton model and return a risk score.

    Args:
        material_type: Primary packaging material.
        weight_grams: Packaging weight in grams.
        recyclability_pct: Recyclable fraction 0–100.
        usage_type: Intended use context.
        region: Regulatory jurisdiction.
        auto_train: If True, train-and-save a model when none exists on disk.

    Returns:
        Float risk score in [0.0, 1.0].
    """
    model = ComplianceModelSingleton()
    if model._pipeline is None:
        try:
            model.load()
        except ModelNotFoundError:
            if auto_train:
                logger.warning("No saved model found — training a fresh model.")
                model.train_and_save()
            else:
                raise
    return model.predict_risk_score(material_type, weight_grams, recyclability_pct, usage_type, region)


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    test_cases: List[Tuple] = [
        ("plastic", 250.0, 40.0, "food", "EU"),
        ("glass", 300.0, 95.0, "food", "USA"),
        ("biodegradable", 100.0, 100.0, "pharmaceutical", "EU"),
        ("composite", 500.0, 25.0, "electronics", "INDIA"),
        ("paper", 200.0, 85.0, "retail", "UK"),
    ]

    print("\n=== ML Risk Score Predictions ===")
    for mat, wt, rec, use, reg in test_cases:
        score = get_ml_risk_score(mat, wt, rec, use, reg, auto_train=True)
        band = (
            "LOW" if score < 0.25 else
            "MEDIUM" if score < 0.55 else
            "HIGH" if score < 0.80 else
            "CRITICAL"
        )
        print(f"  {reg:6s} | {mat:12s} | recyclability={rec:5.1f}% → score={score:.4f} [{band}]")

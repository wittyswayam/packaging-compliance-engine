# 📦 AI-Based Packaging Compliance Checker

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade **hybrid compliance engine** that combines deterministic regulatory rules with a calibrated ML model to evaluate packaging materials against EU, USA, India, UK, China, and Global regulatory frameworks.

---

## 🏗️ Architecture

```
packaging-compliance/
│
├── api/
│   └── app.py                   # FastAPI service — single & batch endpoints
│
├── models/
│   └── compliance_model.joblib  # Auto-generated on first run
│
├── notebooks/
│   └── compliance_analysis.ipynb  # EDA, model validation, visualisations
│
├── schemas/
│   └── models.py                # Pydantic I/O contracts + enums
│
├── src/
│   ├── compliance_model.py      # ML risk scoring (GBT + calibration)
│   └── rules_engine.py          # Regulatory rule engine (Strategy + Factory)
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the API server

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

The ML model trains automatically on first launch (~10 seconds).  
Interactive docs: **http://localhost:8000/docs**

---

## 🔌 API Usage

### Single Compliance Check

```bash
curl -X POST http://localhost:8000/v1/compliance/check \
  -H "Content-Type: application/json" \
  -d '{
    "material_type": "plastic",
    "weight_grams": 250.0,
    "recyclability_pct": 40.0,
    "usage_type": "food",
    "region": "EU",
    "batch_id": "BATCH-2024-001"
  }'
```

**Response:**
```json
{
  "status": "NON_COMPLIANT",
  "risk_score": 0.7412,
  "risk_level": "HIGH",
  "violation_details": [
    {
      "rule_id": "EU_PLASTIC_BAN_001",
      "description": "Single-use plastic packaging is restricted or banned in EU.",
      "severity": "critical",
      "regulation_ref": "EU Single-Use Plastics Directive"
    }
  ],
  "mitigation_plan": [
    {
      "step_number": 1,
      "action": "Replace single-use plastic with certified compostable or mono-material PP/PE.",
      "estimated_cost_usd": 8500.0,
      "priority": "immediate"
    }
  ],
  "batch_id": "BATCH-2024-001",
  "evaluated_at": "2024-01-15T10:30:00+00:00"
}
```

### Batch Check (up to 50 items)

```bash
curl -X POST http://localhost:8000/v1/compliance/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"material_type": "glass", "weight_grams": 300, "recyclability_pct": 95, "usage_type": "food", "region": "EU"},
    {"material_type": "paper", "weight_grams": 50, "recyclability_pct": 85, "usage_type": "retail", "region": "USA"}
  ]'
```

### List Supported Regions

```bash
curl http://localhost:8000/v1/compliance/regions
```

---

## 🧠 How It Works

```
Input Payload
     │
     ├─── Rule Engine (Strategy Pattern) ──► Rule Score [0,1]
     │         │
     │    EU / USA / India / UK / China / Global strategies
     │    each check: recyclability floors, plastic bans,
     │    food weight caps, pharma mandates
     │
     ├─── ML Model (GBT + Isotonic Calibration) ──► ML Score [0,1]
     │         │
     │    Features: material, weight_log, recyclability,
     │    usage, region, recycle_deficit
     │
     └─── Combined Score = 0.40 × rule_score + 0.60 × ml_score
               │
               ▼
         Risk Level + Status + Mitigation Plan
```

### Risk Score Bands

| Score | Band | Status |
|-------|------|--------|
| 0.00 – 0.24 | 🟢 LOW | COMPLIANT |
| 0.25 – 0.54 | 🟡 MEDIUM | COMPLIANT |
| 0.55 – 0.79 | 🟠 HIGH | CONDITIONAL |
| 0.80 – 1.00 | 🔴 CRITICAL | NON_COMPLIANT |

---

## 🧪 Run Module Tests

```bash
# Test rule engine standalone
python src/rules_engine.py

# Test ML model standalone
python src/compliance_model.py

# Test schema validation
python schemas/models.py
```

---

## 📊 Jupyter Notebook

```bash
jupyter notebook notebooks/compliance_analysis.ipynb
```

The notebook covers EDA, feature correlations, ROC curves, calibration plots, and feature importance — all with Seaborn/Matplotlib visualisations.

---

## 🗺️ Supported Regulations

| Region | Key Regulations Covered |
|--------|------------------------|
| **EU** | PPWD 94/62/EC, SUP Directive 2019/904, EU 2030 recyclability targets |
| **USA** | EPA packaging guidelines, California SB 54 (EPR surcharges) |
| **India** | Plastic Waste Management Rules 2016 (amended 2022), 120-micron ban |
| **UK** | Post-Brexit EU baseline + UK Plastics Packaging Tax |
| **China** | Global baseline (extensible) |
| **Global** | Minimum international standards |

---

## 📄 License

MIT © 2024 AI Packaging Compliance Checker

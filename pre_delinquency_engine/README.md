# Pre-Delinquency Intervention Engine

**Executive Summary**

The Pre-Delinquency Intervention Engine is an institutional-grade early warning risk platform that predicts customer loan/EMI default likelihood **2–4 weeks before delinquency** using transaction-level behavioral signals and triggers proactive, ROI-optimized intervention. It demonstrates credit risk discipline (PD–LGD–EAD), portfolio monitoring, model governance, fairness validation, and scalable architecture suitable for banking and lending use cases.

---

## Banking Context

- **Problem**: Late identification of at-risk borrowers leads to higher defaults, lower recovery, and higher cost of collections.
- **Solution**: Behavioral stress signals (salary delay, balance decline, lending app usage, failed debits, etc.) are turned into a 30-day Probability of Default (PD). Exposure at Default (EAD) and Loss Given Default (LGD) are applied to compute Expected Loss (EL). High-risk customers receive targeted interventions (payment holiday, EMI restructure, SMS) with tracked ROI.
- **Outcome**: Lower portfolio loss, improved recovery rates, and audit-ready governance and explainability.

---

## Credit Risk Math

- **PD**: Model output interpreted as 30-day Probability of Default.
- **EAD**: Simulated as a multiple of monthly EMI (e.g. 3×) for exposure at default.
- **LGD**: Loss Given Default (e.g. 45%) applied to EAD.
- **Expected Loss**: **EL = PD × LGD × EAD** (per customer and aggregated for portfolio).
- **Threshold optimization**: Two strategies—AUC/Youden-optimized threshold vs loss-optimized threshold that minimizes expected cost (false positives vs missed defaults). Reports compare both and are stored in `reports/credit_risk_report.json`.

---

## Governance & Compliance

- **Audit trail**: All model versions, training timestamps, feature sets, risk score timestamps, explanation snapshots, and intervention decisions are logged in SQLite. Use `generate_audit_report(customer_id)` for per-customer compliance reports.
- **Model versioning**: Retraining pipeline increments version and logs to governance DB; old vs new AUC comparison supports performance degradation detection.
- **Fairness**: Optional protected attributes (not used in the model) support post-hoc fairness reporting: subgroup AUC, Precision/Recall by segment, Disparate Impact Ratio, Statistical Parity Difference. Output: `reports/fairness_report.json`.

---

## Portfolio Monitoring

- **Risk distribution**: Histogram and tier mix (Low / Medium / High).
- **Portfolio Stress Index**: Mean risk score (0–100) across the book.
- **% High Risk**: Share of customers in the High risk tier.
- **Segment-wise breakdown**: Risk and count by segment (e.g. risk tier or region).
- **Drift detection**: KS test between baseline and current risk score distribution; configurable threshold with Green / Yellow / Red alerting.
- **Alerts**: High-risk population increase beyond X%; significant distribution shift.

---

## ROI Impact

- **Intervention cost** per type (SMS, payment holiday, EMI restructure) is configurable.
- **Expected benefit** per prevented default (exposure × LGD) is computed.
- **Net benefit** and **ROI %** per intervention type and at portfolio level are available in the dashboard and via `intervention_roi` module.

---

## Scalability Strategy

- **Logical architecture**: Dashboard and API sit above a core risk layer (scoring, credit risk, trajectory, portfolio monitoring, interventions) and a model/governance layer (training, SHAP, audit, monitoring, fairness, retraining). See `architecture.md`.
- **Production**: Stateless API and dashboard; model artifacts on shared store (e.g. S3); feature store (offline Parquet + optional online key-value) for batch and real-time scoring.
- **AWS**: ALB + ECS/Lambda, S3/RDS, ElastiCache, CloudWatch for scaling and monitoring.

---

## Responsible AI

- **Explainability**: SHAP global importance and per-customer top-3 risk factors.
- **No protected attributes in model**: Only behavioral/transaction features; protected attributes used only for fairness reporting.
- **Limitations**: Synthetic data and simulated EAD/LGD; production would use actual exposure and recovery data. Threshold and cost parameters should be tuned to institution policy.

---

## Limitations

- Data is synthetic; stress patterns and default labels are simulated.
- EAD and LGD are simplified (e.g. fixed multiplier and mean LGD); production systems use facility-level EAD and segment-level LGD.
- Dashboard login is simulated (hardcoded credentials); production requires proper authentication and authorization.

---

## Project Structure

```
pre_delinquency_engine/
├── data/                    # Synthetic data, SQLite (interventions, audit, risk history)
├── models/                  # Trained models, version.json, metrics
├── reports/                 # credit_risk_report.json, fairness_report.json
├── src/
│   ├── config.py            # Paths, env-based configuration
│   ├── data_generation.py   # Synthetic transactions & stress
│   ├── feature_engineering.py
│   ├── model_training.py    # LR + XGBoost
│   ├── credit_risk.py       # PD-LGD-EAD, EL, threshold optimization
│   ├── risk_trajectory.py   # Risk history, velocity, spike detection
│   ├── portfolio_monitoring.py  # Stress index, drift, alerts
│   ├── explainability.py    # SHAP
│   ├── risk_scoring.py
│   ├── intervention_engine.py
│   ├── intervention_roi.py  # ROI per type, net benefit
│   ├── business_impact.py
│   ├── governance.py        # Audit trail, generate_audit_report
│   ├── fairness.py          # Fairness validation, fairness_report.json
│   ├── monitoring.py        # Feature/score drift, macro stress simulation
│   └── retraining.py        # Periodic retrain, version increment
├── api/main.py              # FastAPI
├── dashboard/app.py         # Executive dashboard
├── run_pipeline.py          # End-to-end pipeline
├── architecture.md          # Logical & production architecture
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
cd pre_delinquency_engine
pip install -r requirements.txt
python run_pipeline.py
```

Then start the API and dashboard (use **two terminals**):

**Terminal 1 — API**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
→ API: http://localhost:8000 | Docs: http://localhost:8000/docs

**Terminal 2 — Dashboard**
```bash
streamlit run dashboard/app.py
```
→ Dashboard: http://localhost:8501

**Windows (PowerShell):** run both from project root:
```powershell
.\run_apps.ps1
```
(Opens API in a new window, then starts the dashboard.)

**Login:** Admin `admin` / `admin123` — Analyst `analyst` / `analyst123`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/customers` | List customers (optional `risk_tier` filter) |
| GET | `/customer/{id}` | Customer detail and transactions |
| GET | `/risk/{id}` | Risk score, tier, SHAP top-3 factors |
| POST | `/trigger-intervention/{id}` | Trigger intervention (High-risk only) |
| GET | `/interventions` | Intervention logs |

---

## Model lifecycle & reproducibility

- **model_status** in `models/metrics.json`: `production` | `candidate` | `shadow` | `deprecated` for lifecycle visibility.
- **data_hash**: SHA-256 of training data in metrics for reproducibility.
- **Model signature**: `input_features` and `feature_count` in metrics to prevent inference mismatch.
- **Drift snapshot**: When monitoring runs, `models/drift_report.json` is written (timestamp, feature/score drift, alerts).
- **Executive log**: Training logs one line: `Model vX.Y.Z trained — AUC: 0.XX — EL Reduction: X.X%`.

---

## Environment Configuration

Optional environment variables (defaults in `src/config.py`):

- `PDE_DATA_DIR`, `PDE_MODELS_DIR`, `PDE_REPORTS_DIR` — Paths.
- `PDE_N_CUSTOMERS`, `PDE_STRESS_FRACTION` — Data generation.
- `PDE_DEFAULT_LGD`, `PDE_EAD_MULTIPLIER_EMI`, `PDE_AVG_EMI` — Credit risk.
- `PDE_HIGH_RISK_ALERT_PCT`, `PDE_DRIFT_KS_THRESHOLD` — Monitoring.
- `PDE_LOG_LEVEL` — Logging level.

---

## License

MIT (hackathon / demonstration project).

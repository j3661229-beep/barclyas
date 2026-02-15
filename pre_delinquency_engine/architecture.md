# Pre-Delinquency Intervention Engine — Architecture

## Logical Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXECUTIVE DASHBOARD (Streamlit)                        │
│  Executive Summary | Risk Heatmap | Acceleration | Drift | Fairness | ROI     │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REST API (FastAPI)                                   │
│  /customers | /customer/{id} | /risk/{id} | /trigger-intervention | /interventions │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CORE RISK LAYER                                      │
│  Risk Scoring (PD) │ Credit Risk (PD-LGD-EAD, EL) │ Risk Trajectory │         │
│  Portfolio Monitoring │ Intervention Engine │ Intervention ROI               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MODEL & GOVERNANCE LAYER                                  │
│  Model Training (XGBoost/LR) │ Explainability (SHAP) │ Governance (Audit)   │
│  Monitoring (Drift) │ Fairness Validation │ Retraining Pipeline             │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                           │
│  Synthetic Data │ Feature Engineering │ SQLite (Interventions, Audit,        │
│  Risk History) │ Reports (credit_risk_report.json, fairness_report.json)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Production Deployment Diagram

```
                    ┌──────────────┐
                    │   Load       │
                    │   Balancer   │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌─────▼─────┐     ┌─────▼─────┐
    │ Streamlit│      │  FastAPI  │     │  FastAPI  │
    │ Dashboard│      │  (app 1)  │     │  (app 2)  │
    └────┬────┘      └─────┬─────┘     └─────┬─────┘
         │                 │                 │
         │            ┌────▼────────────────▼────┐
         │            │    Shared Model Store     │
         │            │  (S3 / Artifact Registry) │
         │            └────┬────────────────┬─────┘
         │                 │                │
         └─────────────────┼────────────────┘
                           │
              ┌────────────▼────────────┐
              │  Feature Store (optional)│
              │  Redis / DynamoDB /     │
              │  Offline (Parquet)      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Data Lake / Warehouse  │
              │  (Transactions,       │
              │   Risk History, Audit)  │
              └────────────────────────┘
```

---

## AWS Scaling Explanation

- **Compute**: FastAPI and Streamlit behind ALB; ECS Fargate or Lambda for async scoring jobs.
- **Model serving**: Model artifacts in S3; loaded at startup or on-demand; SageMaker optional for A/B and shadow deployment.
- **Data**: RDS (PostgreSQL) for audit and interventions at scale; S3 for raw transactions and risk history (partitioned by date).
- **Caching**: ElastiCache (Redis) for risk scores and feature vectors to reduce model load.
- **Monitoring**: CloudWatch metrics (latency, error rate, drift metrics); alerts on High-risk % increase or KS drift threshold.

---

## Feature Store Integration

- **Offline**: Parquet/Delta in S3 per customer and date; feature_engineering logic runs in batch (e.g. Glue/EMR or SageMaker Processing).
- **Online**: Low-latency key-value (e.g. DynamoDB or Redis) keyed by `customer_id` with precomputed features and last score; API reads from here for `/risk/{id}`.
- **Sync**: Batch job appends new transaction-derived features daily; real-time events (e.g. failed debit) can trigger incremental feature update and re-score.

---

## Model Retraining Strategy

1. **Schedule**: Periodic (e.g. monthly) full retrain on last 6–12 months; trigger also on drift alert (KS or performance drop).
2. **Data**: Train on `customer_features.csv` (or feature store snapshot) with `default_next_30_days`; hold-out for validation.
3. **Comparison**: New model AUC vs current production; deploy only if AUC ≥ production - 0.02 and fairness metrics within policy.
4. **Versioning**: `version.json` and governance DB store model_version, training_timestamp, feature_set, metrics; audit trail for every score and intervention.

---

## Monitoring Layer

- **Feature drift**: KS test per feature (training vs current scoring distribution); alert if statistic > threshold.
- **Risk score drift**: KS on score distribution week-over-week; alert if significant shift.
- **Portfolio**: High-risk % increase alert; Portfolio Stress Index; segment breakdown.
- **Model**: AUC/precision/recall from validation; degradation flag in retraining output.

---

## Governance Layer

- **Audit DB**: Model versions, risk score timestamps, explanation snapshots, intervention decisions; `generate_audit_report(customer_id)` for compliance.
- **Fairness**: Optional protected attributes (not in model); subgroup AUC, Disparate Impact, Statistical Parity; `fairness_report.json` for review.
- **Credit risk**: PD–LGD–EAD, Expected Loss, threshold optimization report; stored in `reports/credit_risk_report.json`.

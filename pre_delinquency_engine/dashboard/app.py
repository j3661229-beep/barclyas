"""
Executive dashboard for Pre-Delinquency Intervention Engine.
Institutional risk platform: Executive Summary, Risk Heatmap, Acceleration, Drift, Fairness, ROI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.feature_engineering import FEATURE_NAMES
from src.risk_scoring import score_batch, score_single
from src.intervention_engine import get_interventions, trigger_intervention
from src.explainability import load_model_and_features, run_global_explanation, local_explanation
from src.business_impact import run_business_impact
from src.credit_risk import run_credit_risk_report
from src.portfolio_monitoring import run_portfolio_monitoring
from src.intervention_roi import roi_per_intervention_type, portfolio_intervention_roi_summary
from src.risk_trajectory import get_risk_history, compute_risk_velocity

# Hardcoded login simulation
CREDENTIALS = {"admin": "admin123", "analyst": "analyst123"}


def check_login(username: str, password: str) -> str | None:
    if username in CREDENTIALS and CREDENTIALS[username] == password:
        return "admin" if username == "admin" else "analyst"
    return None


@st.cache_data(ttl=300)
def load_data():
    """Load customer features and risk scores."""
    try:
        path = DATA_DIR / "customer_features.csv"
        if not path.exists():
            return None, None
        df = pd.read_csv(path)
        X = df[FEATURE_NAMES].fillna(0)
        scores = score_batch(X)
        full = pd.concat([df[["customer_id"]], scores], axis=1)
        return full, df
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None, None


@st.cache_data(ttl=300)
def load_metrics():
    """Load model metrics from JSON."""
    try:
        p = MODELS_DIR / "metrics.json"
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _drift_signal(alert: bool, drift_stat: float | None) -> tuple[str, str]:
    """Green / Yellow / Red for drift indicator."""
    if drift_stat is None:
        return "🟢", "No baseline"
    if alert:
        return "🔴", "Drift detected"
    if drift_stat and drift_stat > 0.05:
        return "🟡", "Monitor"
    return "🟢", "Stable"


def main():
    # ========== PAGE CONFIG ==========
    st.set_page_config(
        page_title="Pre-Delinquency Intervention Engine",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ========== PROFESSIONAL CSS STYLING ==========
    st.markdown("""
    <style>
    .big-font {
        font-size: 18px !important;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1e3a5f;
    }
    .section-header {
        color: #1e3a5f;
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    .drift-green { color: #059669; font-weight: 600; }
    .drift-yellow { color: #d97706; font-weight: 600; }
    .drift-red { color: #dc2626; font-weight: 600; }
    .info-box {
        background-color: #eff6ff;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

    # ========== HEADER ==========
    st.markdown("### Pre-Delinquency Intervention Engine")
    st.caption("🏦 Institutional Early Warning Risk Platform")
    st.divider()

    # ========== LOGIN ==========
    if "logged_in_role" not in st.session_state:
        st.session_state.logged_in_role = None
        
    if st.session_state.logged_in_role is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("#### System Login")
            with st.form("login"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True):
                    role = check_login(u, p)
                    if role:
                        st.session_state.logged_in_role = role
                        st.session_state.username = u
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Try admin/admin123 or analyst/analyst123.")
            st.info("Demo credentials: **admin** / **admin123** or **analyst** / **analyst123**")
        st.stop ()

    # ========== SIDEBAR ==========
    role = st.session_state.logged_in_role
    st.sidebar.markdown(f"### 👤 {st.session_state.username}")
    st.sidebar.caption(f"Role: **{role.upper()}**")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in_role = None
        st.rerun()
    st.sidebar.divider()
    st.sidebar.markdown("### Navigation")
    section = st.sidebar.radio(
        "Go to",
        ["Executive Summary", "Portfolio Risk", "Model Insights", "Interventions", "Admin Panel" if role == "admin" else None],
        label_visibility="collapsed"
    )

    # ========== LOAD DATA ==========
    full_df, feature_df = load_data()
    if full_df is None:
        st.warning("⚠️ Run the pipeline first. See README for instructions.")
        st.code("python run_pipeline.py", language="bash")
        st.stop()

    # ========== EXECUTIVE SUMMARY SECTION ==========
    if section == "Executive Summary":
        st.markdown('<div class="section-header">📊 Executive Summary</div>', unsafe_allow_html=True)
        
        # Calculate metrics
        try:
            impact = run_business_impact(
                feature_df=feature_df,
                risk_scores_df=full_df[["risk_score", "risk_tier", "default_probability"]]
            )
            s = impact["summary"]
            loss_without = s["estimated_portfolio_loss_without_system"]
            loss_prevented = s["loss_prevented_via_early_detection"]
        except Exception:loss_without = loss_prevented = 0.0

        try:
            credit_report = run_credit_risk_report(
                feature_df,
                risk_scores_df=full_df[["risk_score", "risk_tier", "default_probability"]]
            )
            portfolio_el = credit_report.get("portfolio_expected_loss", 0.0)
        except Exception:
            portfolio_el = 0.0

        loss_reduction_pct = (loss_prevented / loss_without * 100) if loss_without else 0.0
        
        try:
            metrics = load_metrics()
            auc = metrics.get("xgboost", {}).get("auc_roc", 0.0) if metrics else 0.0
        except Exception:
            auc = 0.0

        try:
            mon = run_portfolio_monitoring(full_df)
            high_risk_pct = mon.get("high_risk_pct", 0)
        except Exception:
            high_risk_pct = 0

        # Display 4 key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio Expected Loss", f"₹{portfolio_el:,.0f}")
        with col2:
            st.metric("Loss Reduction %", f"{loss_reduction_pct:.1f}%")
        with col3:
            st.metric("% High Risk", f"{high_risk_pct:.1f}%")
        with col4:
            st.metric("AUC-ROC", f"{auc:.3f}")

        st.divider()

        # Portfolio Status
        try:
            drift_info = mon.get("drift") or {}
            alert = drift_info.get("alert", False)
            ks_stat = drift_info.get("ks_statistic")
            signal, label = _drift_signal(alert, ks_stat)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Stress Index", f"{mon.get('portfolio_stress_index', 0):.1f}")
            with col2:
                st.metric("Distribution Drift", f"{signal} {label}")
            with col3:
                st.metric("Active Alerts", len(mon.get("alerts", [])))
        except Exception as e:
            st.info("Portfolio monitoring temporarily unavailable.")

    # ========== PORTFOLIO RISK SECTION ==========
    elif section == "Portfolio Risk":
        st.markdown('<div class="section-header">📈 Portfolio Risk Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Left column: Risk distribution
        with col1:
            try:
                st.markdown("#### Risk Distribution")
                tier_counts = full_df["risk_tier"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
                fig_pie = px.pie(
                    values=tier_counts.values,
                    names=tier_counts.index,
                    title="Customer Count by Risk Tier",
                    color_discrete_sequence=["#059669", "#d97706", "#dc2626"],
                    hole=0.4
                )
                fig_pie.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=12)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            except Exception as e:
                st.error("Risk distribution chart unavailable.")

        # Right column: Risk heatmap
        with col2:
            try:
                st.markdown("#### Risk Score Heatmap")
                mon = run_portfolio_monitoring(full_df)
                seg_breakdown = mon.get("segment_breakdown", [])
                if seg_breakdown:
                    seg_df = pd.DataFrame(seg_breakdown)
                    tier_col = "risk_tier" if "risk_tier" in seg_df.columns else "index"
                    fig_heat = px.bar(
                        seg_df,
                        x=tier_col,
                        y="count",
                        color="mean_risk_score",
                        title="Segment Analysis",
                        color_continuous_scale="RdYlGn_r"
                    )
                    fig_heat.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("No segment data available.")
            except Exception as e:
                st.error("Heatmap unavailable.")

        # Risk Acceleration
        st.markdown("#### 🚀 Risk Acceleration — Top 10 Customers")
        try:
            velocities = []
            for cid in full_df["customer_id"].head(200):
                try:
                    hist = get_risk_history(cid, days=14)
                    if len(hist) >= 2:
                        current_pd = full_df[full_df["customer_id"] == cid]["default_probability"].iloc[0]
                        vel = compute_risk_velocity(current_pd, hist["pd"].tolist()[1:], 7)
                        velocities.append({"customer_id": cid, "risk_velocity": vel})
                except:
                    continue
                    
            if velocities:
                acc_df = pd.DataFrame(velocities).nlargest(10, "risk_velocity")
                st.dataframe(acc_df, use_container_width=True)
            else:
                st.info("ℹ️ Risk trajectory simulation not enabled.")
        except Exception as e:
            st.warning("Explainability temporarily unavailable.")

    # ========== MODEL INSIGHTS SECTION ==========
    elif section == "Model Insights":
        st.markdown('<div class="section-header">🧠 Model Explainability</div>', unsafe_allow_html=True)
        
        try:
            X = feature_df[FEATURE_NAMES].fillna(0)
            importance_result = run_global_explanation(X, plot_path=None)
            
            if isinstance(importance_result, dict) and "error" in importance_result:
                st.warning(f"⚠️ {importance_result['error']}")
            else:
                importance_df = importance_result
                fig_bar = px.bar(
                    importance_df.head(15),
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    title="Top 15 Features by SHAP Importance"
                )
                fig_bar.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.warning("⚠️ Explainability temporarily unavailable.")

        # Fairness section
        st.markdown("#### ⚖️ Fairness & Bias Validation")
        try:
            fairness_path = REPORTS_DIR / "fairness_report.json"
            if fairness_path.exists():
                with open(fairness_path) as f:
                    fair = json.load(f)
                st.caption("Protected attributes are not used in the model; used only for fairness reporting.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Subgroup AUC**")
                    if fair.get("subgroup_auc"):
                        st.dataframe(pd.DataFrame([fair["subgroup_auc"]]).T, use_container_width=True)
                with col2:
                    st.markdown("**Disparate Impact Ratio**")
                    if fair.get("disparate_impact_ratio"):
                        st.dataframe(pd.DataFrame([fair["disparate_impact_ratio"]]).T, use_container_width=True)
            else:
                st.info("Run fairness validation to generate fairness_report.json.")
        except Exception as e:
            st.error("Fairness report unavailable.")

    # ========== INTERVENTIONS SECTION ==========
    elif section == "Interventions":
        st.markdown('<div class="section-header">💼 Intervention Management</div>', unsafe_allow_html=True)
        
        # ROI Summary
        try:
            logs = get_interventions(limit=500)
            if logs:
                roi_summary = portfolio_intervention_roi_summary(pd.DataFrame(logs))
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Net Benefit", f"₹{roi_summary.get('total_net_benefit', 0):,.0f}")
                with col2:
                    st.metric("Total Cost", f"₹{roi_summary.get('total_cost', 0):,.0f}")
                with col3:
                    roi_pct = (roi_summary.get("total_net_benefit", 0) / roi_summary.get("total_cost", 1) * 100) if roi_summary.get("total_cost") else 0
                    st.metric("ROI %", f"{roi_pct:.0f}%")
        except Exception:
            pass

        # ROI by Type
        st.markdown("#### ROI by Intervention Type")
        try:
            roi_by_type = roi_per_intervention_type()
            if roi_by_type:
                st.dataframe(pd.DataFrame(roi_by_type), use_container_width=True)
            else:
                st.info("No intervention data available yet.")
        except Exception:
            st.info("ROI calculation temporarily unavailable.")

        # Intervention Logs
        st.markdown("#### Recent Interventions")
        try:
            logs = get_interventions(limit=100)
            if logs:
                st.dataframe(pd.DataFrame(logs), use_container_width=True, height=400)
            else:
                st.info("No interventions recorded yet.")
        except Exception:
            st.error("Unable to load intervention logs.")

    # ========== ADMIN PANEL ==========
    elif section == "Admin Panel" and role == "admin":
        st.markdown('<div class="section-header">⚙️ Admin Panel</div>', unsafe_allow_html=True)
        
        # Model Performance
        st.markdown("#### Model Performance Metrics")
        try:
            metrics = load_metrics()
            if metrics:
                xgb = metrics.get("xgboost", {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("AUC-ROC", f"{xgb.get('auc_roc', 0):.3f}")
                with col2:
                    st.metric("Precision", f"{xgb.get('precision', 0):.3f}")
                with col3:
                    st.metric("Recall", f"{xgb.get('recall', 0):.3f}")
                with col4:
                    st.metric("F1-Score", f"{xgb.get('f1', 0):.3f}")
            else:
                st.info("Metrics not available.")
        except Exception:
            st.error("Unable to load metrics.")

        # Trigger Intervention
        st.markdown("#### Trigger Intervention")
        try:
            high_risk_ids = full_df[full_df["risk_tier"] == "High"]["customer_id"].tolist()
            if high_risk_ids:
                chosen = st.selectbox(
                    "Select High-Risk Customer",
                    high_risk_ids[:200] if len(high_risk_ids) > 200 else high_risk_ids
                )
                if st.button("🚨 Trigger Intervention", type="primary"):
                    try:
                        row = feature_df[feature_df["customer_id"] == chosen][FEATURE_NAMES].fillna(0).iloc[0]
                        res = score_single(row)
                        recorded = trigger_intervention(chosen, res["risk_tier"], res["risk_score"])
                        st.success(f"✅ Successfully triggered {len(recorded)} intervention(s) for customer {chosen}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.info("No high-risk customers currently identified.")
        except Exception as e:
            st.error(f"Intervention system unavailable: {str(e)}")

        # Customer Explanations
        st.markdown("#### Customer Risk Explanation")
        try:
            high_df = full_df[full_df["risk_tier"] == "High"].head(100)
            if not high_df.empty:
                view_id = st.selectbox("Select Customer for Explanation", high_df["customer_id"].tolist())
                if view_id:
                    try:
                        model, fn = load_model_and_features()
                        expl = local_explanation(model, feature_df, view_id, fn)
                        
                        if "error" in expl:
                            st.warning(f"⚠️ {expl['error']}")
                        else:
                            st.markdown("**Top 3 Risk Factors:**")
                            for i, f in enumerate(expl["top_3_risk_factors"], 1):
                                st.write(f"{i}. **{f['feature']}**: SHAP = {f['shap_value']:.4f}")
                    except Exception as e:
                        st.warning(f"Unable to generate explanation: {str(e)}")
            else:
                st.info("No high-risk customers available for explanation.")
        except Exception as e:
            st.error(f"Explanation system unavailable: {str(e)}")


if __name__ == "__main__":
    main()

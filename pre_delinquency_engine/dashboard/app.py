"""
Executive dashboard for Pre-Delinquency Intervention Engine.
Production-grade institutional risk platform with premium UI/UX.
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
        initial_sidebar_state="expanded",
        page_icon="🏦"
    )

    # ========== PREMIUM PRODUCTION CSS ==========
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Settings */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Container Styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem 3rem;
    }
    
    /* Premium Header Styling */
    .premium-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        animation: fadeIn 0.8s ease-in;
    }
    
    .premium-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .premium-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Metric Cards with Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Section Headers */
    .section-header {
        color: #1a202c;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
        width: 100%;
    }
    
    /* Custom Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #4a5568 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.875rem !important;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.25rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 8px 16px rgba(245, 87, 108, 0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.25rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 8px 16px rgba(79, 172, 254, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.25rem;
        border-radius: 12px;
        color: #1a202c;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 8px 16px rgba(250, 112, 154, 0.2);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3561 0%, #1a1f3a 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Login Container */
    .login-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        max-width: 450px;
        margin: 5rem auto;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .slide-in {
        animation: slideIn 0.6s ease-out;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.5px;
    }
    
    .status-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .status-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #1a202c;
    }
    
    .status-low {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        color: white;
    }
    
    /* Card Grid */
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ========== PREMIUM HEADER ==========
    st.markdown("""
    <div class="premium-header">
        <h1>🏦 Pre-Delinquency Intervention Engine</h1>
        <p>Institutional Early Warning Risk Platform · Real-Time Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # ========== LOGIN ==========
    if "logged_in_role" not in st.session_state:
        st.session_state.logged_in_role = None
        
    if st.session_state.logged_in_role is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="login-container fade-in">', unsafe_allow_html=True)
            st.markdown("#### 🔐 Secure Login")
            st.markdown("---")
            with st.form("login"):
                u = st.text_input("👤 Username", placeholder="Enter username")
                p = st.text_input("🔑 Password", type="password", placeholder="Enter password")
                if st.form_submit_button("Login", use_container_width=True):
                    role = check_login(u, p)
                    if role:
                        st.session_state.logged_in_role = role
                        st.session_state.username = u
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
            
            st.markdown('<div class="info-box">💡 Demo: <strong>admin/admin123</strong> or <strong>analyst/analyst123</strong></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # ========== SIDEBAR ==========
    role = st.session_state.logged_in_role
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 1.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
        <div style="font-size: 3rem;">👤</div>
        <div style="font-size: 1.25rem; font-weight: 600; color: white; margin-top: 0.5rem;">{st.session_state.username}</div>
        <div style="font-size: 0.875rem; color: rgba(255,255,255,0.7); margin-top: 0.25rem;">{role.upper()} ACCESS</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in_role = None
        st.rerun()
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("### 📊 Navigation")
    section = st.sidebar.radio(
        "Go to section",
        ["Executive Summary", "Portfolio Risk", "Model Insights", "Interventions", "Admin Panel" if role == "admin" else None],
        label_visibility="collapsed"
    )

    # ========== LOAD DATA ==========
    full_df, feature_df = load_data()
    if full_df is None:
        st.markdown('<div class="warning-box">⚠️ <strong>Pipeline Required:</strong> Run the pipeline first to generate data</div>', unsafe_allow_html=True)
        st.code("python run_pipeline.py", language="bash")
        st.stop()

    # ========== EXECUTIVE SUMMARY SECTION ==========
    if section == "Executive Summary":
        st.markdown('<div class="section-header slide-in">📊 Executive Summary</div>', unsafe_allow_html=True)
        
        # Calculate metrics
        try:
            impact = run_business_impact(
                feature_df=feature_df,
                risk_scores_df=full_df[["risk_score", "risk_tier", "default_probability"]]
            )
            s = impact["summary"]
            loss_without = s["estimated_portfolio_loss_without_system"]
            loss_prevented = s["loss_prevented_via_early_detection"]
        except Exception:
            loss_without = loss_prevented = 0.0

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

        # Display 4 key metrics with premium styling
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card fade-in">', unsafe_allow_html=True)
            st.metric("Portfolio Expected Loss", f"₹{portfolio_el:,.0f}", delta=None)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card fade-in">', unsafe_allow_html=True)
            st.metric("Loss Reduction", f"{loss_reduction_pct:.1f}%", delta=f"{loss_reduction_pct:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card fade-in">', unsafe_allow_html=True)
            st.metric("High Risk Portfolio", f"{high_risk_pct:.1f}%", delta=f"-{high_risk_pct:.1f}%" if high_risk_pct > 0 else "0%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card fade-in">', unsafe_allow_html=True)
            st.metric("Model AUC-ROC", f"{auc:.3f}", delta="Good" if auc >= 0.75 else "Fair")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Portfolio Status
        try:
            drift_info = mon.get("drift") or {}
            alert = drift_info.get("alert", False)
            ks_stat = drift_info.get("ks_statistic")
            signal, label = _drift_signal(alert, ks_stat)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Portfolio Stress", f"{mon.get('portfolio_stress_index', 0):.1f}")
            with col2:
                st.metric("🔄 Distribution Drift", f"{signal} {label}")
            with col3:
                st.metric("🚨 Active Alerts", len(mon.get("alerts", [])))
        except Exception:
            st.markdown('<div class="info-box">ℹ️ Portfolio monitoring temporarily unavailable</div>', unsafe_allow_html=True)

    # ========== PORTFOLIO RISK SECTION ==========
    elif section == "Portfolio Risk":
        st.markdown('<div class="section-header slide-in">📈 Portfolio Risk Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Left column: Risk distribution
        with col1:
            try:
                st.markdown("#### 🎯 Risk Distribution")
                tier_counts = full_df["risk_tier"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
                fig_pie = px.pie(
                    values=tier_counts.values,
                    names=tier_counts.index,
                    title="Customer Risk Tiers",
                    color_discrete_sequence=["#30cfd0", "#fa709a", "#f5576c"],
                    hole=0.5
                )
                fig_pie.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=13, family="Inter"),
                    showlegend=True,
                    height=400
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            except Exception:
                st.markdown('<div class="warning-box">⚠️ Risk distribution chart unavailable</div>', unsafe_allow_html=True)

        # Right column: Risk heatmap
        with col2:
            try:
                st.markdown("#### 🔥 Risk Score Heatmap")
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
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter"),
                        height=400
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.markdown('<div class="info-box">ℹ️ No segment data available</div>', unsafe_allow_html=True)
            except Exception:
                st.markdown('<div class="warning-box">⚠️ Heatmap unavailable</div>', unsafe_allow_html=True)

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
                st.dataframe(acc_df, use_container_width=True, height=350)
            else:
                st.markdown('<div class="info-box">ℹ️ Risk trajectory simulation not enabled</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="warning-box">⚠️ Acceleration data unavailable</div>', unsafe_allow_html=True)

    # ========== MODEL INSIGHTS SECTION ==========
    elif section == "Model Insights":
        st.markdown('<div class="section-header slide-in">🧠 Model Explainability</div>', unsafe_allow_html=True)
        
        try:
            X = feature_df[FEATURE_NAMES].fillna(0)
            importance_result = run_global_explanation(X, plot_path=None)
            
            if isinstance(importance_result, dict) and "error" in importance_result:
                st.markdown(f'<div class="warning-box">⚠️ {importance_result["error"]}</div>', unsafe_allow_html=True)
            else:
                importance_df = importance_result
                fig_bar = px.bar(
                    importance_df.head(15),
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    title="Top 15 SHAP Feature Importance",
                    color="mean_abs_shap",
                    color_continuous_scale="Viridis"
                )
                fig_bar.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter"),
                    height=500
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        except Exception:
            st.markdown('<div class="warning-box">⚠️ Explainability temporarily unavailable</div>', unsafe_allow_html=True)

        # Fairness section
        st.markdown("#### ⚖️ Fairness & Bias Validation")
        try:
            fairness_path = REPORTS_DIR / "fairness_report.json"
            if fairness_path.exists():
                with open(fairness_path) as f:
                    fair = json.load(f)
                st.caption("Protected attributes not used in model — validation only")
                
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
                st.markdown('<div class="info-box">ℹ️ Run fairness validation to generate report</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="warning-box">⚠️ Fairness report unavailable</div>', unsafe_allow_html=True)

    # ========== INTERVENTIONS SECTION ==========
    elif section == "Interventions":
        st.markdown('<div class="section-header slide-in">💼 Intervention Management</div>', unsafe_allow_html=True)
        
        # ROI Summary
        try:
            logs = get_interventions(limit=500)
            if logs:
                roi_summary = portfolio_intervention_roi_summary(pd.DataFrame(logs))
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💰 Total Net Benefit", f"₹{roi_summary.get('total_net_benefit', 0):,.0f}")
                with col2:
                    st.metric("💸 Total Cost", f"₹{roi_summary.get('total_cost', 0):,.0f}")
                with col3:
                    roi_pct = (roi_summary.get("total_net_benefit", 0) / roi_summary.get("total_cost", 1) * 100) if roi_summary.get("total_cost") else 0
                    st.metric("📊 ROI Percentage", f"{roi_pct:.0f}%")
        except Exception:
            pass

        # ROI by Type
        st.markdown("#### 📋 ROI by Intervention Type")
        try:
            roi_by_type = roi_per_intervention_type()
            if roi_by_type:
                st.dataframe(pd.DataFrame(roi_by_type), use_container_width=True)
            else:
                st.markdown('<div class="info-box">ℹ️ No intervention data available yet</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="info-box">ℹ️ ROI calculation temporarily unavailable</div>', unsafe_allow_html=True)

        # Intervention Logs
        st.markdown("#### 📝 Recent Interventions")
        try:
            logs = get_interventions(limit=100)
            if logs:
                st.dataframe(pd.DataFrame(logs), use_container_width=True, height=400)
            else:
                st.markdown('<div class="info-box">ℹ️ No interventions recorded yet</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="warning-box">⚠️ Unable to load intervention logs</div>', unsafe_allow_html=True)

    # ========== ADMIN PANEL ==========
    elif section == "Admin Panel" and role == "admin":
        st.markdown('<div class="section-header slide-in">⚙️ Admin Panel</div>', unsafe_allow_html=True)
        
        # Model Performance
        st.markdown("#### 📊 Model Performance Metrics")
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
                st.markdown('<div class="info-box">ℹ️ Metrics not available</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="warning-box">⚠️ Unable to load metrics</div>', unsafe_allow_html=True)

        # Trigger Intervention
        st.markdown("#### 🚨 Trigger Intervention")
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
                        st.markdown(f'<div class="success-box">✅ Successfully triggered {len(recorded)} intervention(s) for customer {chosen}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="warning-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">ℹ️ No high-risk customers currently identified</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="warning-box">⚠️ Intervention system unavailable: {str(e)}</div>', unsafe_allow_html=True)

        # Customer Explanations
        st.markdown("#### 🔍 Customer Risk Explanation")
        try:
            high_df = full_df[full_df["risk_tier"] == "High"].head(100)
            if not high_df.empty:
                view_id = st.selectbox("Select Customer for Explanation", high_df["customer_id"].tolist())
                if view_id:
                    try:
                        model, fn = load_model_and_features()
                        expl = local_explanation(model, feature_df, view_id, fn)
                        
                        if "error" in expl:
                            st.markdown(f'<div class="warning-box">⚠️ {expl["error"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown("**Top 3 Risk Factors:**")
                            for i, f in enumerate(expl["top_3_risk_factors"], 1):
                                st.write(f"{i}. **{f['feature']}**: SHAP = {f['shap_value']:.4f}")
                    except Exception as e:
                        st.markdown(f'<div class="warning-box">⚠️ Unable to generate explanation: {str(e)}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">ℹ️ No high-risk customers available for explanation</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="warning-box">⚠️ Explanation system unavailable: {str(e)}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

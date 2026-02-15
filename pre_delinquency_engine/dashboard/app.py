"""
Executive dashboard for Pre-Delinquency Intervention Engine.
PREMIUM PROFESSIONAL UI with Top Navigation Bar - World-Class Design
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
        page_title="Pre-Delinquency Engine",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="🏦"
    )

    # ========== WORLD-CLASS PREMIUM CSS ==========
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@600;700&display=swap');
    
    /* === GLOBAL RESET === */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* === MAIN CONTAINER === */
    .main {
        background: #f8f9fc;
        padding: 0 !important;
    }
    
    .block-container {
        padding-top: 0 !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 100% !important;
    }
    
    /* === TOP NAVIGATION BAR === */
    .top-navbar {
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
        margin: 0 -3rem 2rem -3rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .navbar-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.25rem 3rem;
        max-width: 100%;
    }
    
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .navbar-logo {
        font-size: 2rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .navbar-title {
        color: white;
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
    }
    
    .navbar-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.875rem;
        font-weight: 400;
        margin-top: 0.25rem;
    }
    
    .navbar-user {
        display: flex;
        align-items: center;
        gap: 1rem;
        color: white;
    }
    
    .user-avatar {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .user-info {
        text-align: right;
    }
    
    .user-name {
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .user-role {
        font-size: 0.75rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* === NAVIGATION TABS === */
    .nav-tabs {
        display: flex;
        gap: 0.5rem;
        background: white;
        padding: 1rem 0;
        margin: 0 -3rem 2rem -3rem;
        padding-left: 3rem;
        padding-right: 3rem;
        border-bottom: 2px solid #e2e8f0;
        position: sticky;
        top: 70px;
        z-index: 998;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .nav-tab {
        padding: 0.75rem 1.5rem;
        background: transparent;
        border: none;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .nav-tab:hover {
        background: #f1f5f9;
        color: #667eea;
    }
    
    .nav-tab.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* === PREMIUM METRIC CARDS === */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2.5rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.12);
        border-color: #e2e8f0;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    /* === SECTION STYLING === */
    .section {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
    }
    
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-header-icon {
        font-size: 1.75rem;
    }
    
    /* === CUSTOM STREAMLIT OVERRIDES === */
    [data-testid="stMetricValue"] {
        font-size: 2.25rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
    }
    
    /* === ALERT BOXES === */
    .alert {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 1rem;
        border-left: 4px solid;
    }
    
    .alert-success {
        background: #f0fdf4;
        color: #166534;
        border-left-color: #22c55e;
    }
    
    .alert-info {
        background: #eff6ff;
        color: #1e40af;
        border-left-color: #3b82f6;
    }
    
    .alert-warning {
        background: #fffbeb;
        color: #92400e;
        border-left-color: #f59e0b;
    }
    
    .alert-error {
        background: #fef2f2;
        color: #991b1b;
        border-left-color: #ef4444;
    }
    
    /* === BUTTONS === */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        letter-spacing: 0.3px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* === LOGIN CONTAINER === */
    .login-wrapper {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .login-container {
        background: white;
        border-radius: 24px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        max-width: 420px;
        width: 100%;
        animation: slideUp 0.6s ease-out;
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .login-logo {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .login-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .login-subtitle {
        color: #64748b;
        font-size: 0.95rem;
    }
    
    /* === DATAFRAMES === */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #f1f5f9 !important;
    }
    
    /* === ANIMATIONS === */
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

    # ========== SESSION STATE INITIALIZATION ==========
    if "logged_in_role" not in st.session_state:
        st.session_state.logged_in_role = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Dashboard"

    # ========== LOGIN PAGE ==========
    if st.session_state.logged_in_role is None:
        st.markdown("""
        <div class="login-wrapper">
            <div class="login-container">
                <div class="login-header">
                    <div class="login-logo">🏦</div>
                    <h1 class="login-title">Welcome Back</h1>
                    <p class="login-subtitle">Pre-Delinquency Intervention Engine</p>
                </div>
        """, unsafe_allow_html=True)
        
        with st.form("login"):
            u = st.text_input("👤 Username", placeholder="Enter your username")
            p = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Login", use_container_width=True):
                    role = check_login(u, p)
                    if role:
                        st.session_state.logged_in_role = role
                        st.session_state.username = u
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
        
        st.markdown('<div class="alert alert-info">💡 <strong>Demo Access:</strong> admin/admin123 or analyst/analyst123</div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        st.stop()

    # ========== LOGGED IN - SHOW NAVBAR ==========
    role = st.session_state.logged_in_role
    username = st.session_state.username
    
    # Top Navigation Bar
    st.markdown(f"""
    <div class="top-navbar">
        <div class="navbar-content">
            <div class="navbar-brand">
                <div class="navbar-logo">🏦</div>
                <div>
                    <div class="navbar-title">Pre-Delinquency Engine</div>
                    <div class="navbar-subtitle">Institutional Risk Platform</div>
                </div>
            </div>
            <div class="navbar-user">
                <div class="user-info">
                    <div class="user-name">{username}</div>
                    <div class="user-role">{role} Access</div>
                </div>
                <div class="user-avatar">👤</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Tabs
    tabs = ["Dashboard", "Portfolio", "Analytics", "Interventions"]
    if role == "admin":
        tabs.append("Admin")
    
    cols = st.columns(len(tabs) + 1)
    for idx, tab in enumerate(tabs):
        with cols[idx]:
            if st.button(f"{'📊' if tab=='Dashboard' else '📈' if tab=='Portfolio' else '🧠' if tab=='Analytics' else '💼' if tab=='Interventions' else '⚙️'} {tab}", key=f"tab_{tab}", use_container_width=True):
                st.session_state.active_tab = tab
                st.rerun()
    
    with cols[-1]:
        if st.button("🚪 Logout", key="logout", use_container_width=True):
            st.session_state.logged_in_role = None
            st.rerun()

    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # ========== LOAD DATA ==========
    full_df, feature_df = load_data()
    if full_df is None:
        st.markdown('<div class="alert alert-warning">⚠️ <strong>Pipeline Required:</strong> Run the pipeline first to generate data</div>', unsafe_allow_html=True)
        st.code("python run_pipeline.py", language="bash")
        st.stop()

    # ========== DASHBOARD TAB ==========
    if st.session_state.active_tab == "Dashboard":
        # Calculate metrics
        try:
            impact = run_business_impact(feature_df=feature_df, risk_scores_df=full_df[["risk_score", "risk_tier", "default_probability"]])
            s = impact["summary"]
            loss_without = s["estimated_portfolio_loss_without_system"]
            loss_prevented = s["loss_prevented_via_early_detection"]
        except:
            loss_without = loss_prevented = 0.0

        try:
            credit_report = run_credit_risk_report(feature_df, risk_scores_df=full_df[["risk_score", "risk_tier", "default_probability"]])
            portfolio_el = credit_report.get("portfolio_expected_loss", 0.0)
        except:
            portfolio_el = 0.0

        loss_reduction_pct = (loss_prevented / loss_without * 100) if loss_without else 0.0
        
        try:
            metrics = load_metrics()
            auc = metrics.get("xgboost", {}).get("auc_roc", 0.0) if metrics else 0.0
        except:
            auc = 0.0

        try:
            mon = run_portfolio_monitoring(full_df)
            high_risk_pct = mon.get("high_risk_pct", 0)
        except:
            high_risk_pct = 0

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio Expected Loss", f"₹{portfolio_el:,.0f}")
        with col2:
            st.metric("Loss Reduction", f"{loss_reduction_pct:.1f}%", delta=f"{loss_reduction_pct:.1f}%")
        with col3:
            st.metric("High Risk %", f"{high_risk_pct:.1f}%")
        with col4:
            st.metric("Model AUC", f"{auc:.3f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Portfolio Status
        try:
            drift_info = mon.get("drift") or {}
            alert = drift_info.get("alert", False)
            ks_stat = drift_info.get("ks_statistic")
            signal, label = _drift_signal(alert, ks_stat)
            
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span class="section-header-icon">📊</span>Portfolio Status</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stress Index", f"{mon.get('portfolio_stress_index', 0):.1f}")
            with col2:
                st.metric("Drift Status", f"{signal} {label}")
            with col3:
                st.metric("Active Alerts", len(mon.get("alerts", [])))
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            pass

    # ========== PORTFOLIO TAB ==========
    elif st.session_state.active_tab == "Portfolio":
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-header-icon">📈</span>Risk Distribution</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            try:
                tier_counts = full_df["risk_tier"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
                fig_pie = px.pie(values=tier_counts.values, names=tier_counts.index, hole=0.5,
                               color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"])
                fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                     font=dict(family="Inter"), height=350)
                st.plotly_chart(fig_pie, use_container_width=True)
            except:
                st.markdown('<div class="alert alert-error">⚠️ Chart unavailable</div>', unsafe_allow_html=True)
        
        with col2:
            try:
                mon = run_portfolio_monitoring(full_df)
                seg_breakdown = mon.get("segment_breakdown", [])
                if seg_breakdown:
                    seg_df = pd.DataFrame(seg_breakdown)
                    tier_col = "risk_tier" if "risk_tier" in seg_df.columns else "index"
                    fig_bar = px.bar(seg_df, x=tier_col, y="count", color="mean_risk_score", color_continuous_scale="RdYlGn_r")
                    fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=350)
                    st.plotly_chart(fig_bar, use_container_width=True)
            except:
                pass
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== ANALYTICS TAB ==========
    elif st.session_state.active_tab == "Analytics":
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-header-icon">🧠</span>Model Explainability</div>', unsafe_allow_html=True)
        
        try:
            X = feature_df[FEATURE_NAMES].fillna(0)
            importance_result = run_global_explanation(X, plot_path=None)
            
            if isinstance(importance_result, dict) and "error" not in importance_result:
                importance_df = importance_result
                fig = px.bar(importance_df.head(15), x="mean_abs_shap", y="feature", orientation="h",
                           color="mean_abs_shap", color_continuous_scale="Viridis")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)', height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('<div class="alert alert-warning">⚠️ Explainability temporarily unavailable</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="alert alert-error">⚠️ Unable to load analytics</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== INTERVENTIONS TAB ==========
    elif st.session_state.active_tab == "Interventions":
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-header-icon">💼</span>Intervention Management</div>', unsafe_allow_html=True)
        
        try:
            logs = get_interventions(limit=100)
            if logs:
                st.dataframe(pd.DataFrame(logs), use_container_width=True, height=500)
            else:
                st.markdown('<div class="alert alert-info">ℹ️ No interventions recorded yet</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="alert alert-error">⚠️ Unable to load interventions</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== ADMIN TAB ==========
    elif st.session_state.active_tab == "Admin" and role == "admin":
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-header-icon">⚙️</span>Admin Panel</div>', unsafe_allow_html=True)
        
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
        except:
            pass
        
        # Trigger Intervention
        st.markdown("#### 🚨 Trigger Intervention")
        try:
            high_risk_ids = full_df[full_df["risk_tier"] == "High"]["customer_id"].tolist()
            if high_risk_ids:
                chosen = st.selectbox("Select High-Risk Customer", high_risk_ids[:200] if len(high_risk_ids) > 200 else high_risk_ids)
                if st.button("🚨 Trigger Intervention", type="primary"):
                    try:
                        row = feature_df[feature_df["customer_id"] == chosen][FEATURE_NAMES].fillna(0).iloc[0]
                        res = score_single(row)
                        recorded = trigger_intervention(chosen, res["risk_tier"], res["risk_score"])
                        st.markdown(f'<div class="alert alert-success">✅ Successfully triggered {len(recorded)} intervention(s)</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="alert alert-error">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
        except:
            pass
        
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

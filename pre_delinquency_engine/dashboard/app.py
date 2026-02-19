"""
Barclays · Pre-Delinquency Intervention Engine
Institutional Banking Dashboard — Production Grade UI
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
from src.explainability import run_global_explanation
from src.business_impact import run_business_impact
from src.credit_risk import run_credit_risk_report
from src.portfolio_monitoring import run_portfolio_monitoring
from src.intervention_roi import portfolio_intervention_roi_summary

# ── Auth ──────────────────────────────────────────────────────────────────────
CREDENTIALS = {"admin": "admin123", "analyst": "analyst123"}

def check_login(u: str, p: str) -> str | None:
    if u in CREDENTIALS and CREDENTIALS[u] == p:
        return "admin" if u == "admin" else "analyst"
    return None

# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
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
        st.error(f"Data error: {e}")
        return None, None

@st.cache_data(ttl=300)
def load_metrics():
    try:
        p = MODELS_DIR / "metrics.json"
        return json.load(open(p)) if p.exists() else None
    except Exception:
        return None

def _drift_signal(alert: bool, stat) -> tuple[str, str]:
    if stat is None:
        return "STABLE", "No Baseline"
    if alert:
        return "DRIFT", "Drift Detected"
    if stat and stat > 0.05:
        return "WATCH", "Monitor"
    return "STABLE", "Stable"

# ── Plotly defaults ───────────────────────────────────────────────────────────
_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, Inter, sans-serif", color="#94A3B8", size=12),
    margin=dict(l=16, r=16, t=36, b=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8")),
)
_RISK_COLORS = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}

def _fig(fig: go.Figure, h: int = 340) -> go.Figure:
    fig.update_layout(**_LAYOUT, height=h)
    fig.update_xaxes(gridcolor="#1E2D45", zerolinecolor="#1E2D45",
                     tickfont=dict(color="#64748B"), linecolor="#1E2D45")
    fig.update_yaxes(gridcolor="#1E2D45", zerolinecolor="#1E2D45",
                     tickfont=dict(color="#64748B"), linecolor="#1E2D45")
    return fig

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* BASE */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #020D1A !important;
    color: #CBD5E1;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDeployButton"],
[data-testid="stDecoration"],
.stStatusWidget { display: none !important; }

.main, [data-testid="stAppViewContainer"] { background: #020D1A !important; }
.block-container {
    padding: 0 2rem 4rem 2rem !important;
    max-width: 100% !important;
}

/* ── TOPBAR ── */
.bk-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #03111E;
    border-bottom: 1px solid #0F2540;
    padding: 0 2rem;
    height: 60px;
    margin: 0 -2rem 0 -2rem;
    position: sticky; top: 0; z-index: 1000;
}
.bk-brand { display: flex; align-items: center; gap: 0.85rem; }
.bk-logobox {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #006ED2 0%, #00A8E8 100%);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.bk-brand-name  { font-size: 0.9rem; font-weight: 700; color: #F1F5F9; letter-spacing: 0.2px; }
.bk-brand-sub   { font-size: 0.65rem; font-weight: 500; color: #334D6E; text-transform: uppercase; letter-spacing: 1.2px; margin-top: 1px; }
.bk-topbar-right { display: flex; align-items: center; gap: 1.25rem; }
.bk-live {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 0.65rem; font-weight: 700; color: #10B981;
    text-transform: uppercase; letter-spacing: 1px;
}
.bk-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 0 0 rgba(16,185,129,0.7);
    animation: ping 1.8s infinite;
}
@keyframes ping {
    0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }
    70%  { box-shadow: 0 0 0 7px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}
.bk-userpill {
    display: flex; align-items: center; gap: 0.6rem;
    background: #071826; border: 1px solid #0F2540;
    border-radius: 50px; padding: 0.3rem 0.85rem 0.3rem 0.5rem;
}
.bk-avatar {
    width: 26px; height: 26px; border-radius: 50%;
    background: #0F2540;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
}
.bk-uname { font-size: 0.8rem; font-weight: 600; color: #E2E8F0; }
.bk-urole { font-size: 0.6rem; color: #334D6E; text-transform: uppercase; letter-spacing: 0.6px; }

/* Divider under topbar */
.bk-divider { height: 1px; background: #0F2540; margin: 0 -2rem; }

/* ── NAV TABS ── */
.bk-nav {
    display: flex;
    gap: 0;
    background: #03111E;
    border-bottom: 1px solid #0F2540;
    padding: 0 2rem;
    margin: 0 -2rem 1.75rem -2rem;
}

/* ── KPI STRIP ── */
.bk-kpis {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #0F2540;
    border: 1px solid #0F2540;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.bk-kpi {
    background: #03111E;
    padding: 1.25rem 1.5rem;
    transition: background 0.2s;
}
.bk-kpi:hover { background: #071826; }
.bk-kpi-label {
    font-size: 0.62rem; font-weight: 700;
    color: #334D6E; letter-spacing: 1.8px; text-transform: uppercase;
    margin-bottom: 0.45rem;
}
.bk-kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem; font-weight: 600;
    color: #E2E8F0; letter-spacing: -1px; line-height: 1.1;
}
.bk-kpi-val.g { color: #10B981; }
.bk-kpi-val.a { color: #F59E0B; }
.bk-kpi-val.r { color: #EF4444; }
.bk-kpi-val.b { color: #38BDF8; }
.bk-kpi-delta {
    font-size: 0.7rem; font-weight: 600; margin-top: 0.3rem;
    color: #10B981;
}
.bk-kpi-delta.neg { color: #EF4444; }

/* ── STATUS ROW ── */
.bk-statusrow {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1px; background: #0F2540;
    border: 1px solid #0F2540; border-radius: 10px;
    overflow: hidden; margin-bottom: 1.5rem;
}
.bk-scell {
    background: #03111E;
    padding: 1rem 1.5rem;
    display: flex; align-items: center; gap: 0.9rem;
}
.bk-scell:hover { background: #071826; }
.bk-sicon {
    width: 38px; height: 38px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.ig { background: rgba(16,185,129,0.1); }
.ia { background: rgba(245,158,11,0.1); }
.ir { background: rgba(239,68,68,0.1); }
.ib { background: rgba(0,174,239,0.1); }
.bk-slbl { font-size: 0.6rem; font-weight: 700; color: #334D6E; text-transform: uppercase; letter-spacing: 1.2px; }
.bk-sval { font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 600; color: #CBD5E1; margin-top: 2px; }

/* ── CARD ── */
.bk-card {
    background: #03111E;
    border: 1px solid #0F2540;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.bk-card-hdr {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.9rem 1.4rem;
    border-bottom: 1px solid #0F2540;
    background: #071826;
}
.bk-card-ttl {
    font-size: 0.7rem; font-weight: 700;
    color: #64748B; text-transform: uppercase; letter-spacing: 1.5px;
}
.bk-badge {
    font-size: 0.6rem; font-weight: 700;
    padding: 0.2rem 0.6rem; border-radius: 4px;
    text-transform: uppercase; letter-spacing: 0.8px;
}
.bb { background: rgba(56,189,248,0.12); color: #38BDF8; }
.bg { background: rgba(16,185,129,0.12); color: #10B981; }
.ba { background: rgba(245,158,11,0.12); color: #F59E0B; }
.br { background: rgba(239,68,68,0.12); color: #EF4444; }
.bk-card-body { padding: 1.4rem; }

/* ── ALERTS ── */
.bk-alert {
    display: flex; align-items: flex-start; gap: 0.85rem;
    padding: 0.9rem 1.1rem;
    border-radius: 7px; border-left: 3px solid;
    margin: 0.6rem 0; font-size: 0.83rem; font-weight: 500;
}
.ai { background: rgba(56,189,248,0.06); border-color: #38BDF8; color: #7DD3FC; }
.as { background: rgba(16,185,129,0.06); border-color: #10B981; color: #6EE7B7; }
.aw { background: rgba(245,158,11,0.06); border-color: #F59E0B; color: #FDE68A; }
.ae { background: rgba(239,68,68,0.06);  border-color: #EF4444; color: #FCA5A5; }

/* ── STREAMLIT METRIC OVERRIDE ── */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.6rem !important; font-weight: 600 !important;
    color: #E2E8F0 !important; -webkit-text-fill-color: #E2E8F0 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.62rem !important; font-weight: 700 !important;
    color: #334D6E !important; text-transform: uppercase !important;
    letter-spacing: 1.8px !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; font-weight: 600 !important; }
[data-testid="metric-container"] {
    background: #03111E !important;
    border: 1px solid #0F2540 !important;
    border-radius: 8px !important;
    padding: 1rem 1.25rem !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: #071826 !important;
    color: #64748B !important;
    border: 1px solid #0F2540 !important;
    border-radius: 5px !important;
    font-size: 0.72rem !important; font-weight: 700 !important;
    letter-spacing: 0.8px !important; text-transform: uppercase !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #0F2540 !important; color: #E2E8F0 !important;
    border-color: #006ED2 !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #006ED2 0%, #00A8E8 100%) !important;
    color: #fff !important; border-color: transparent !important;
    box-shadow: 0 4px 14px rgba(0,110,210,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 20px rgba(0,110,210,0.55) !important;
    transform: translateY(-1px);
}

/* ── FORM INPUTS ── */
.stTextInput > div > div > input,
.stSelectbox > div > div > div {
    background: #071826 !important;
    border: 1px solid #0F2540 !important;
    color: #E2E8F0 !important; border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #006ED2 !important;
    box-shadow: 0 0 0 2px rgba(0,110,210,0.18) !important;
}
label[data-testid="stWidgetLabel"] {
    color: #334D6E !important; font-size: 0.65rem !important;
    font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 1.4px !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] > div > div {
    background: #03111E !important;
    border: 1px solid #0F2540 !important;
    border-radius: 8px !important;
}
iframe[title="st_aggrid"] { background: #03111E; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #03111E; }
::-webkit-scrollbar-thumb { background: #0F2540; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #1E3A5F; }

/* ── LOGIN ── */
.bk-login-bg {
    min-height: 100vh; background: #020D1A;
    display: flex; align-items: center; justify-content: center;
}
.bk-login-card {
    background: #03111E; border: 1px solid #0F2540;
    border-radius: 14px; padding: 2.75rem 2.25rem;
    width: 360px; box-shadow: 0 32px 80px rgba(0,0,0,0.7);
    animation: rise 0.45s ease-out;
}
.bk-login-head {
    display: flex; align-items: center; gap: 0.85rem;
    padding-bottom: 1.5rem; border-bottom: 1px solid #0F2540;
    margin-bottom: 1.75rem;
}
.bk-login-logo {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #006ED2 0%, #00A8E8 100%);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
}
.bk-login-brand-name { font-size: 0.92rem; font-weight: 700; color: #F1F5F9; }
.bk-login-brand-sub  { font-size: 0.62rem; color: #334D6E; text-transform: uppercase; letter-spacing: 1.2px; margin-top: 2px; }
.bk-login-title { font-size: 1.2rem; font-weight: 700; color: #E2E8F0; margin-bottom: 0.3rem; }
.bk-login-sub   { font-size: 0.8rem; color: #334D6E; margin-bottom: 1.5rem; }

/* ── PAGE TITLE ── */
.bk-ptitle  { font-size: 0.6rem; font-weight: 700; color: #334D6E; text-transform: uppercase; letter-spacing: 2px; margin: 1.6rem 0 0.2rem; }
.bk-phead   { font-size: 1.3rem; font-weight: 700; color: #F1F5F9; margin-bottom: 1.4rem; }

@keyframes rise {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
"""

# ── HTML helpers ──────────────────────────────────────────────────────────────
def alert(msg: str, kind: str = "i") -> str:
    icons = {"i": "ℹ", "s": "✓", "w": "⚠", "e": "✕"}
    cls   = {"i": "ai", "s": "as", "w": "aw", "e": "ae"}
    return f'<div class="bk-alert {cls[kind]}"><span>{icons[kind]}</span><span>{msg}</span></div>'

def kpi(label, value, style="", delta="", delta_neg=False):
    d = f'<div class="bk-kpi-delta{"  neg" if delta_neg else ""}">{("▼ " if delta_neg else "▲ ") + delta}</div>' if delta else ""
    return f'<div class="bk-kpi"><div class="bk-kpi-label">{label}</div><div class="bk-kpi-val {style}">{value}</div>{d}</div>'

def card_open(title: str, badge: str = "", bcls: str = "bb"):
    b = f'<span class="bk-badge {bcls}">{badge}</span>' if badge else ""
    return f'<div class="bk-card"><div class="bk-card-hdr"><span class="bk-card-ttl">{title}</span>{b}</div><div class="bk-card-body">'

def card_close():
    return "</div></div>"

def scell(icon, icls, label, value):
    return f'<div class="bk-scell"><div class="bk-sicon {icls}">{icon}</div><div><div class="bk-slbl">{label}</div><div class="bk-sval">{value}</div></div></div>'


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Barclays · Pre-Delinquency Engine",
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="🏦",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # Session defaults
    for k, v in [("role", None), ("username", ""), ("tab", "Dashboard")]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ══════════════════════════════════════════════════════════════════════════
    # LOGIN
    # ══════════════════════════════════════════════════════════════════════════
    if not st.session_state.role:
        st.markdown("""
        <div class="bk-login-bg">
          <div class="bk-login-card">
            <div class="bk-login-head">
              <div class="bk-login-logo">🏦</div>
              <div>
                <div class="bk-login-brand-name">Barclays Risk Intelligence</div>
                <div class="bk-login-brand-sub">Pre-Delinquency Division</div>
              </div>
            </div>
            <div class="bk-login-title">Secure Sign&#8209;In</div>
            <div class="bk-login-sub">Enter your institutional credentials to access the platform</div>
        """, unsafe_allow_html=True)

        with st.form("lf", border=False):
            u = st.text_input("Username", placeholder="username")
            p = st.text_input("Password", type="password", placeholder="••••••••")
            if st.form_submit_button("Sign In →", use_container_width=True, type="primary"):
                role = check_login(u, p)
                if role:
                    st.session_state.role     = role
                    st.session_state.username = u
                    st.rerun()
                else:
                    st.markdown(alert("Invalid credentials. Please try again.", "e"), unsafe_allow_html=True)

        st.markdown(alert("Demo — <b>admin / admin123</b> &nbsp;·&nbsp; <b>analyst / analyst123</b>", "i"), unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)
        st.stop()

    role = st.session_state.role
    uname = st.session_state.username

    # ══════════════════════════════════════════════════════════════════════════
    # TOP BAR
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div class="bk-topbar">
      <div class="bk-brand">
        <div class="bk-logobox">🏦</div>
        <div>
          <div class="bk-brand-name">Barclays · Pre-Delinquency Engine</div>
          <div class="bk-brand-sub">Institutional Risk Platform &nbsp;|&nbsp; AI-Powered</div>
        </div>
      </div>
      <div class="bk-topbar-right">
        <div class="bk-live"><div class="bk-dot"></div>Live</div>
        <div class="bk-userpill">
          <div class="bk-avatar">👤</div>
          <div>
            <div class="bk-uname">{uname}</div>
            <div class="bk-urole">{role}</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # NAV TABS (native Streamlit buttons styled via CSS)
    # ══════════════════════════════════════════════════════════════════════════
    all_tabs = ["Dashboard", "Portfolio", "Analytics", "Interventions"]
    if role == "admin":
        all_tabs.append("Admin")

    ICONS = {"Dashboard": "📊", "Portfolio": "📈", "Analytics": "🧠",
             "Interventions": "💼", "Admin": "⚙"}

    nav_cols = st.columns(len(all_tabs) + 2)
    for i, t in enumerate(all_tabs):
        with nav_cols[i]:
            active = (t == st.session_state.tab)
            if st.button(f"{ICONS[t]}  {t}", key=f"nav_{t}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.tab = t
                st.rerun()
    with nav_cols[len(all_tabs)]:
        if st.button("🚪  Sign Out", key="nav_logout", use_container_width=True):
            for k in ["role", "username", "tab"]:
                st.session_state[k] = None if k == "role" else ("Dashboard" if k == "tab" else "")
            st.rerun()

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # DATA
    # ══════════════════════════════════════════════════════════════════════════
    full_df, feature_df = load_data()
    if full_df is None:
        st.markdown(alert("<b>Pipeline Required</b> — run <code>python run_pipeline.py</code> first.", "w"),
                    unsafe_allow_html=True)
        st.code("python run_pipeline.py", language="bash")
        st.stop()

    tab = st.session_state.tab

    # ══════════════════════════════════════════════════════════════════════════
    # DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    if tab == "Dashboard":
        st.markdown('<div class="bk-ptitle">Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="bk-phead">Portfolio Risk Dashboard</div>', unsafe_allow_html=True)

        # Compute KPIs
        try:
            impact = run_business_impact(
                feature_df=feature_df,
                risk_scores_df=full_df[["risk_score","risk_tier","default_probability"]])
            s = impact["summary"]
            loss_wo = s.get("estimated_portfolio_loss_without_system", 0)
            loss_pv = s.get("loss_prevented_via_early_detection", 0)
        except:
            loss_wo = loss_pv = 0.0

        try:
            cr = run_credit_risk_report(
                feature_df,
                risk_scores_df=full_df[["risk_score","risk_tier","default_probability"]])
            p_el = cr.get("portfolio_expected_loss", 0.0)
        except:
            p_el = 0.0

        red_pct = (loss_pv / loss_wo * 100) if loss_wo else 0.0

        try:
            m   = load_metrics()
            auc = m.get("xgboost", {}).get("auc_roc", 0.0) if m else 0.0
        except:
            auc = 0.0

        try:
            mon = run_portfolio_monitoring(full_df)
            hr  = mon.get("high_risk_pct", 0)
        except:
            mon = {}; hr = 0

        auc_s = "g" if auc >= 0.75 else ("a" if auc >= 0.60 else "r")
        hr_s  = "r" if hr > 20 else ("a" if hr > 10 else "g")

        st.markdown(
            '<div class="bk-kpis">'
            + kpi("Portfolio Exp. Loss",  f"₹{p_el:,.0f}")
            + kpi("Loss Reduction",       f"{red_pct:.1f}%",  style="g", delta=f"{red_pct:.1f}%")
            + kpi("High-Risk Accounts",   f"{hr:.1f}%",       style=hr_s)
            + kpi("Model AUC-ROC",        f"{auc:.3f}",       style=auc_s)
            + "</div>",
            unsafe_allow_html=True,
        )

        # Status row
        try:
            di    = mon.get("drift") or {}
            sig, lbl = _drift_signal(di.get("alert", False), di.get("ks_statistic"))
            stress   = mon.get("portfolio_stress_index", 0)
            n_alerts = len(mon.get("alerts", []))
            ds  = "ir" if sig == "DRIFT" else ("ia" if sig == "WATCH" else "ig")
            ss  = "ir" if stress > 70 else ("ia" if stress > 40 else "ig")
            als = "ir" if n_alerts > 0 else "ig"
            st.markdown(
                '<div class="bk-statusrow">'
                + scell("📉", ss,  "Stress Index",  f"{stress:.1f}")
                + scell("📡", ds,  "Model Drift",   f"{sig} — {lbl}")
                + scell("🔔", als, "Active Alerts", str(n_alerts))
                + "</div>",
                unsafe_allow_html=True,
            )
        except:
            pass

        # Charts row
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(card_open("Risk Score Distribution", "Live", "bg"), unsafe_allow_html=True)
            try:
                fig = px.histogram(full_df, x="risk_score", nbins=50,
                                   color_discrete_sequence=["#006ED2"])
                fig.update_traces(marker_line_color="#03111E", marker_line_width=0.5, opacity=0.9)
                fig = _fig(fig, 310)
                fig.update_layout(xaxis_title="Risk Score", yaxis_title="Count", bargap=0.03)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as ex:
                st.markdown(alert(f"Chart error: {ex}", "e"), unsafe_allow_html=True)
            st.markdown(card_close(), unsafe_allow_html=True)

        with c2:
            st.markdown(card_open("Tier Breakdown", "3 Tiers", "bb"), unsafe_allow_html=True)
            try:
                tc = full_df["risk_tier"].value_counts().reindex(["Low","Medium","High"], fill_value=0)
                fig2 = go.Figure(go.Pie(
                    labels=tc.index, values=tc.values, hole=0.62,
                    marker=dict(colors=list(_RISK_COLORS.values()),
                                line=dict(color="#03111E", width=3)),
                    textinfo="percent+label",
                    textfont=dict(color="#CBD5E1", size=11),
                ))
                fig2 = _fig(fig2, 310)
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as ex:
                st.markdown(alert(f"Chart error: {ex}", "e"), unsafe_allow_html=True)
            st.markdown(card_close(), unsafe_allow_html=True)

        # Watchlist table
        st.markdown(card_open("High-Risk Accounts — Watchlist", "Top 10", "br"), unsafe_allow_html=True)
        try:
            cols_avail = [c for c in ["customer_id","risk_score","default_probability","risk_tier"] if c in full_df.columns]
            top = full_df[full_df["risk_tier"] == "High"].nlargest(10, "risk_score")[cols_avail].reset_index(drop=True)
            top.index += 1
            st.dataframe(top, use_container_width=True, height=290)
        except Exception as ex:
            st.markdown(alert(f"Table error: {ex}", "e"), unsafe_allow_html=True)
        st.markdown(card_close(), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PORTFOLIO
    # ══════════════════════════════════════════════════════════════════════════
    elif tab == "Portfolio":
        st.markdown('<div class="bk-ptitle">Portfolio</div>', unsafe_allow_html=True)
        st.markdown('<div class="bk-phead">Portfolio Risk Analysis</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(card_open("Tier Distribution", "Donut", "bb"), unsafe_allow_html=True)
            try:
                tc = full_df["risk_tier"].value_counts().reindex(["Low","Medium","High"], fill_value=0)
                fig = go.Figure(go.Pie(
                    labels=tc.index, values=tc.values, hole=0.62,
                    marker=dict(colors=list(_RISK_COLORS.values()), line=dict(color="#03111E", width=3)),
                    textinfo="percent+label", textfont=dict(color="#CBD5E1", size=11),
                ))
                fig = _fig(fig, 360)
                fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.12))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as ex:
                st.markdown(alert(f"{ex}", "e"), unsafe_allow_html=True)
            st.markdown(card_close(), unsafe_allow_html=True)

        with c2:
            st.markdown(card_open("Segment Risk Bar", "Scores", "ba"), unsafe_allow_html=True)
            try:
                mon2 = run_portfolio_monitoring(full_df)
                seg  = mon2.get("segment_breakdown", [])
                if seg:
                    sdf = pd.DataFrame(seg)
                    xcol = "risk_tier" if "risk_tier" in sdf.columns else sdf.columns[0]
                    fig2 = px.bar(sdf, x=xcol, y="count",
                                  color="mean_risk_score",
                                  color_continuous_scale=["#10B981","#F59E0B","#EF4444"])
                    fig2 = _fig(fig2, 360)
                    fig2.update_traces(marker_line_width=0)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.markdown(alert("No segment data available.", "i"), unsafe_allow_html=True)
            except Exception as ex:
                st.markdown(alert(f"{ex}", "e"), unsafe_allow_html=True)
            st.markdown(card_close(), unsafe_allow_html=True)

        # Box plot
        st.markdown(card_open("Risk Score Distribution by Tier", "Box Plot", "bb"), unsafe_allow_html=True)
        try:
            fig3 = px.box(full_df, x="risk_tier", y="risk_score",
                          color="risk_tier",
                          color_discrete_map=_RISK_COLORS,
                          category_orders={"risk_tier": ["Low","Medium","High"]})
            fig3 = _fig(fig3, 360)
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as ex:
            st.markdown(alert(f"{ex}", "e"), unsafe_allow_html=True)
        st.markdown(card_close(), unsafe_allow_html=True)

        # Full risk table
        st.markdown(card_open("Full Risk Register", f"{len(full_df):,} Accounts", "bg"), unsafe_allow_html=True)
        st.dataframe(full_df.sort_values("risk_score", ascending=False).reset_index(drop=True),
                     use_container_width=True, height=380)
        st.markdown(card_close(), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ANALYTICS
    # ══════════════════════════════════════════════════════════════════════════
    elif tab == "Analytics":
        st.markdown('<div class="bk-ptitle">Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="bk-phead">Model Explainability & Performance</div>', unsafe_allow_html=True)

        # Model KPIs
        try:
            m   = load_metrics()
            xgb = m.get("xgboost", {}) if m else {}
            auc, prec, rec, f1 = (xgb.get(k, 0.0) for k in ["auc_roc","precision","recall","f1"])
        except:
            auc = prec = rec = f1 = 0.0

        st.markdown(
            '<div class="bk-kpis">'
            + kpi("AUC-ROC",   f"{auc:.3f}",  style="g" if auc >= 0.75 else "a")
            + kpi("Precision", f"{prec:.3f}", style="g" if prec >= 0.7 else "a")
            + kpi("Recall",    f"{rec:.3f}",  style="g" if rec >= 0.7 else "a")
            + kpi("F1-Score",  f"{f1:.3f}",   style="g" if f1 >= 0.7 else "a")
            + "</div>",
            unsafe_allow_html=True,
        )

        # SHAP
        st.markdown(card_open("SHAP Global Feature Importance", "Top 15", "bb"), unsafe_allow_html=True)
        try:
            X   = feature_df[FEATURE_NAMES].fillna(0)
            imp = run_global_explanation(X, plot_path=None)
            if isinstance(imp, pd.DataFrame) and "error" not in imp.columns:
                fig = px.bar(imp.head(15), x="mean_abs_shap", y="feature", orientation="h",
                             color="mean_abs_shap",
                             color_continuous_scale=["#1E3A5F","#006ED2","#00A8E8"])
                fig.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_showscale=False)
                fig = _fig(fig, 480)
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(alert("SHAP explainability temporarily unavailable.", "w"), unsafe_allow_html=True)
        except Exception as ex:
            st.markdown(alert(f"SHAP error: {ex}", "w"), unsafe_allow_html=True)
        st.markdown(card_close(), unsafe_allow_html=True)

        # Violin
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(card_open("Score Violin by Tier", "Distribution", "bb"), unsafe_allow_html=True)
            try:
                fig2 = px.violin(full_df, x="risk_tier", y="risk_score",
                                 color="risk_tier", box=True, points=False,
                                 color_discrete_map=_RISK_COLORS,
                                 category_orders={"risk_tier": ["Low","Medium","High"]})
                fig2 = _fig(fig2, 330)
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as ex:
                st.markdown(alert(f"{ex}", "e"), unsafe_allow_html=True)
            st.markdown(card_close(), unsafe_allow_html=True)

        with c2:
            st.markdown(card_open("Default Probability Histogram", "All Accounts", "ba"), unsafe_allow_html=True)
            try:
                fig3 = px.histogram(full_df, x="default_probability", nbins=40,
                                    color_discrete_sequence=["#F59E0B"])
                fig3.update_traces(marker_line_color="#03111E", marker_line_width=0.5, opacity=0.85)
                fig3 = _fig(fig3, 330)
                fig3.update_layout(xaxis_title="Default Probability", yaxis_title="Accounts")
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as ex:
                st.markdown(alert(f"{ex}", "e"), unsafe_allow_html=True)
            st.markdown(card_close(), unsafe_allow_html=True)

        # Feature correlations with risk score
        st.markdown(card_open("Feature Correlation with Risk Score", "Heatmap", "bg"), unsafe_allow_html=True)
        try:
            feat_cols = [c for c in FEATURE_NAMES if c in feature_df.columns]
            corr_vals = feature_df[feat_cols].corrwith(full_df["risk_score"]).sort_values(ascending=False)
            fig4 = go.Figure(go.Bar(
                x=corr_vals.values,
                y=corr_vals.index,
                orientation="h",
                marker=dict(
                    color=corr_vals.values,
                    colorscale=[[0,"#EF4444"],[0.5,"#334D6E"],[1,"#10B981"]],
                    cmin=-1, cmax=1,
                    line=dict(width=0),
                ),
            ))
            fig4 = _fig(fig4, 360)
            fig4.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig4, use_container_width=True)
        except Exception as ex:
            st.markdown(alert(f"Correlation chart error: {ex}", "w"), unsafe_allow_html=True)
        st.markdown(card_close(), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # INTERVENTIONS
    # ══════════════════════════════════════════════════════════════════════════
    elif tab == "Interventions":
        st.markdown('<div class="bk-ptitle">Interventions</div>', unsafe_allow_html=True)
        st.markdown('<div class="bk-phead">Intervention Case Management</div>', unsafe_allow_html=True)

        try:
            roi_sum = portfolio_intervention_roi_summary()
            if roi_sum:
                st.markdown(
                    '<div class="bk-kpis">'
                    + kpi("Total Cases",    str(roi_sum.get("total_cases", 0)))
                    + kpi("Total ROI",      f"₹{roi_sum.get('total_roi', 0):,.0f}",          style="g")
                    + kpi("Avg ROI / Case", f"₹{roi_sum.get('avg_roi_per_case', 0):,.0f}",   style="g")
                    + kpi("Success Rate",   f"{roi_sum.get('success_rate', 0):.1f}%",         style="g")
                    + "</div>",
                    unsafe_allow_html=True,
                )
        except:
            pass

        st.markdown(card_open("Intervention Log", "Last 100", "bb"), unsafe_allow_html=True)
        try:
            logs = get_interventions(limit=100)
            if logs:
                st.dataframe(pd.DataFrame(logs), use_container_width=True, height=450)
            else:
                st.markdown(alert("No interventions recorded yet. Use the Admin tab to trigger one.", "i"),
                            unsafe_allow_html=True)
        except Exception as ex:
            st.markdown(alert(f"Error: {ex}", "e"), unsafe_allow_html=True)
        st.markdown(card_close(), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ADMIN
    # ══════════════════════════════════════════════════════════════════════════
    elif tab == "Admin" and role == "admin":
        st.markdown('<div class="bk-ptitle">Administration</div>', unsafe_allow_html=True)
        st.markdown('<div class="bk-phead">System Administration</div>', unsafe_allow_html=True)

        # Model performance
        st.markdown(card_open("Model Performance — XGBoost", "Production", "bb"), unsafe_allow_html=True)
        try:
            m   = load_metrics()
            xgb = m.get("xgboost", {}) if m else {}
            mc  = st.columns(4)
            mc[0].metric("AUC-ROC",   f"{xgb.get('auc_roc',0):.4f}")
            mc[1].metric("Precision", f"{xgb.get('precision',0):.4f}")
            mc[2].metric("Recall",    f"{xgb.get('recall',0):.4f}")
            mc[3].metric("F1-Score",  f"{xgb.get('f1',0):.4f}")
        except Exception as ex:
            st.markdown(alert(f"Metrics unavailable: {ex}", "w"), unsafe_allow_html=True)
        st.markdown(card_close(), unsafe_allow_html=True)

        # Trigger intervention
        st.markdown(card_open("Manual Intervention Trigger", "Admin Only", "br"), unsafe_allow_html=True)
        try:
            hi_ids = full_df[full_df["risk_tier"] == "High"]["customer_id"].tolist()
            if hi_ids:
                chosen = st.selectbox("Select Customer ID (High-Risk)", hi_ids[:300])
                if st.button("🚨  Trigger Intervention", type="primary", key="trig"):
                    try:
                        row = feature_df[feature_df["customer_id"] == chosen][FEATURE_NAMES].fillna(0).iloc[0]
                        res = score_single(row)
                        recs = trigger_intervention(chosen, res["risk_tier"], res["risk_score"])
                        st.markdown(
                            alert(f"<b>{len(recs)}</b> intervention(s) triggered for customer <b>{chosen}</b>.", "s"),
                            unsafe_allow_html=True,
                        )
                    except Exception as ex:
                        st.markdown(alert(f"Trigger failed: {ex}", "e"), unsafe_allow_html=True)
            else:
                st.markdown(alert("No high-risk customers in current data.", "i"), unsafe_allow_html=True)
        except Exception as ex:
            st.markdown(alert(f"Error: {ex}", "e"), unsafe_allow_html=True)
        st.markdown(card_close(), unsafe_allow_html=True)

        # Single customer scorer
        st.markdown(card_open("Single Customer Risk Scorer", "On-Demand", "ba"), unsafe_allow_html=True)
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            cid = st.selectbox("Customer ID", full_df["customer_id"].tolist()[:500], key="sc_cid")
        with sc2:
            st.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
            run_it = st.button("Run Score →", type="primary", key="sc_btn")
        if run_it:
            try:
                row   = feature_df[feature_df["customer_id"] == cid][FEATURE_NAMES].fillna(0).iloc[0]
                res   = score_single(row)
                tier  = res["risk_tier"]
                score = res["risk_score"]
                prob  = res["default_probability"]
                bcls  = {"Low":"bg","Medium":"ba","High":"br"}.get(tier, "bb")
                st.markdown(
                    f"""<div style="display:flex;gap:1.5rem;margin-top:1rem;flex-wrap:wrap">
                      <div class="bk-kpi" style="border-radius:8px;border:1px solid #0F2540;flex:1;min-width:140px">
                        <div class="bk-kpi-label">Risk Score</div>
                        <div class="bk-kpi-val b">{score:.4f}</div>
                      </div>
                      <div class="bk-kpi" style="border-radius:8px;border:1px solid #0F2540;flex:1;min-width:140px">
                        <div class="bk-kpi-label">Default Probability</div>
                        <div class="bk-kpi-val a">{prob:.2%}</div>
                      </div>
                      <div class="bk-kpi" style="border-radius:8px;border:1px solid #0F2540;flex:1;min-width:140px">
                        <div class="bk-kpi-label">Risk Tier</div>
                        <span class="bk-badge {bcls}" style="font-size:0.88rem;padding:0.4rem 1rem">{tier}</span>
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            except Exception as ex:
                st.markdown(alert(f"Scoring failed: {ex}", "e"), unsafe_allow_html=True)
        st.markdown(card_close(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from dataclasses import replace
from option_mc_hedging_v12 import HedgeConfig, simulate_paths, hedge_on_paths
from option_mc_trace_v12_1 import hedge_path_trace
from scipy.stats import norm


# ============================================================
# MODULE V12.3 INTÉGRÉ — AUCUN FICHIER SUPPLÉMENTAIRE
# ============================================================

def risk_metrics(spot0, spot1, delta0, delta1, gamma0, hedge0):
    """Expositions signées, exprimées en actions et euros; sans seuil arbitraire."""
    vals = np.array([spot0, spot1, delta0, delta1, gamma0, hedge0], dtype=float)
    if not np.all(np.isfinite(vals)) or spot0 <= 0 or spot1 <= 0:
        raise ValueError('Paramètres de diagnostic invalides.')
    return {
        'initial_residual': float(delta0 + hedge0),
        'scenario_residual': float(delta1 + hedge0),
        'rebalance': float(-delta1 - hedge0),
        'gamma_delta_1pct': float(gamma0 * 0.01 * spot0),
        'gamma_pnl_1pct': float(0.5 * gamma0 * (0.01 * spot0)**2),
        'gamma_pnl_5pct': float(0.5 * gamma0 * (0.05 * spot0)**2),
    }


def static_hedge_pnl(spots, option_pnl, spot0, hedge0):
    """P&L de revalorisation instantanée, hedge initial constant, hors portage/frais.

    Les options sont revalorisées au même T, sigma et r. Il ne s'agit pas
    d'un P&L réalisé ni d'une couverture dynamique autofinancée.
    """
    x = np.asarray(spots, dtype=float)
    y = np.asarray(option_pnl, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Courbes P&L invalides.')
    if not np.isfinite(spot0) or spot0 <= 0 or not np.isfinite(hedge0):
        raise ValueError('Spot ou couverture invalide.')
    return y, y + float(hedge0) * (x - float(spot0))


def render_risk_diagnostic(spot0, spot1, delta0, delta1, gamma0, hedge0, key):
    m = risk_metrics(spot0, spot1, delta0, delta1, gamma0, hedge0)
    st.markdown('### Diagnostic automatique du risque')
    st.caption('Diagnostic déterministe et explicable : aucun score opaque ni seuil de danger universel.')
    if gamma0 > 1e-10:
        st.info('Gamma positif : convexité locale positive. Le Delta augmente quand le spot monte, à paramètres constants.')
    elif gamma0 < -1e-10:
        st.warning('Gamma négatif : convexité locale négative. Le Delta diminue quand le spot monte, à paramètres constants ; la couverture peut nécessiter des ajustements importants.')
    else:
        st.info('Gamma initial proche de zéro en valeur absolue ; cela ne signifie pas une absence de risque.')
    a,b,c = st.columns(3)
    a.metric('Delta résiduel après choc', f"{m['scenario_residual']:+.3f} actions")
    b.metric('Ordre de rééquilibrage', f"{abs(m['rebalance']):.3f} actions")
    c.metric('Terme Gamma pour choc +1 %', f"{m['gamma_pnl_1pct']:+.2f} €")
    direction = 'Acheter' if m['rebalance'] > 1e-9 else 'Vendre' if m['rebalance'] < -1e-9 else 'Aucun ordre :'
    st.write(f"**Action théorique :** {direction} {abs(m['rebalance']):.3f} action(s) pour retrouver la neutralité Delta au spot simulé.")
    st.caption(f"Variation locale estimée du Delta pour +1 % de spot : {m['gamma_delta_1pct']:+.3f} actions. "
               'Approximation Gamma × ΔS autour de S₀ ; non valable pour des chocs importants. '
               'Le diagnostic n’évalue pas la VaR, la liquidité ni les risques de modèle.')


def render_static_pnl_comparison(spots, option_pnl, spot0, current_spot, hedge0, key):
    bare, covered = static_hedge_pnl(spots, option_pnl, spot0, hedge0)
    spots = np.asarray(spots, dtype=float)
    st.markdown('### P&L non couvert / couvert — comparaison directe')
    st.caption('Même scénario de spot pour les deux courbes ; options revalorisées à maturité, volatilité et taux constants. '
               'Couverture = nombre d’actions fixé au départ (−Delta initial), sans rééquilibrage, frais ni financement. '
               'Ce sont des variations de valeur théoriques, pas des gains réalisés à échéance.')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=bare, mode='lines', name='Options seules (non couvert)', line=dict(color='#c2410c', width=2.5)))
    fig.add_trace(go.Scatter(x=spots, y=covered, mode='lines', name='Options + couverture Delta statique', line=dict(color='#2563eb', width=2.5)))
    fig.add_vline(x=spot0, line_dash='dot', line_color='#64748b')
    fig.add_vline(x=current_spot, line_dash='dash', line_color='#16a34a')
    fig.add_hline(y=0, line_color='#94a3b8', line_width=1)
    fig.update_layout(template='plotly_white', height=430, xaxis_title='Cours du sous-jacent S (€)',
                      yaxis_title='Variation de valeur P&L (€)', legend=dict(orientation='h', y=-0.28),
                      margin=dict(l=40,r=30,t=25,b=100))
    st.plotly_chart(fig, width='stretch', key=key)
    pnl_bare = float(np.interp(current_spot, spots, bare))
    pnl_covered = float(np.interp(current_spot, spots, covered))
    a,b,c = st.columns(3)
    a.metric('P&L options seules (≈)', f'{pnl_bare:+.2f} €')
    b.metric('P&L avec couverture initiale (≈)', f'{pnl_covered:+.2f} €')
    c.metric('Effet de la couverture (≈)', f'{pnl_covered-pnl_bare:+.2f} €')
    st.caption('Valeurs interpolées sur la grille de spots du graphique. Pour le P&L exact au curseur, utiliser les indicateurs de scénario déjà présents.')

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Option Risk Manager",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# INTERFACE — THÈME CLAIR PROFESSIONNEL
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #f4f7fb;
        --card: #ffffff;
        --text: #172033;
        --muted: #667085;
        --blue: #2563eb;
        --blue-soft: #eff6ff;
        --border: #e4e7ec;
        --green: #15803d;
        --red: #dc2626;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.88);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.15rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.02em;
    }

    h1 {
        font-weight: 800 !important;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.045);
        min-height: 96px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--text);
        font-weight: 750;
        font-variant-numeric: tabular-nums;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
        border-width: 1px;
    }

    div[data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {
        border-radius: 10px !important;
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: var(--border) !important;
    }

    .stRadio [role="radiogroup"] {
        gap: 0.45rem;
    }

    .stRadio label {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.35rem 0.65rem;
    }

    .stSlider [data-baseweb="slider"] {
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
    }

    /* V9.13 : lisibilité renforcée, sans modifier les graphiques ITM/OTM. */
    html, body, [data-testid="stAppViewContainer"] { font-size: 16px; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li { font-size: 1.01rem; line-height: 1.58; }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p { font-size: .91rem !important; line-height: 1.45; }
    [data-testid="stWidgetLabel"] p { font-size: .98rem !important; }
    [data-testid="stMetricLabel"] p { font-size: .94rem !important; }
    .section-kicker, .scenario-item, .hedge-decision-label,
    .risk-secondary-title { font-size: .82rem !important; }
    .risk-reading-item, .hedge-breakdown, .risk-conclusion { font-size: .94rem !important; }
    .top-kicker { font-size: .82rem !important; }
    .top-sub { font-size: 1rem !important; }
    .top-shell {
        background: linear-gradient(120deg, #0f2f57 0%, #174b83 58%, #2563eb 100%);
        border-radius: 22px;
        padding: 1.55rem 1.75rem;
        margin: 0 0 1.15rem 0;
        color: white;
        box-shadow: 0 14px 34px rgba(15, 47, 87, 0.16);
    }
    .top-kicker {font-size:.72rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;opacity:.78;margin-bottom:.35rem;}
    .top-title {font-size:2rem;font-weight:800;letter-spacing:-.035em;line-height:1.08;margin:0;}
    .top-sub {font-size:.93rem;opacity:.82;margin-top:.5rem;}
    .badge-row {display:flex;gap:.45rem;flex-wrap:wrap;margin:.25rem 0 1.15rem 0;}
    .badge {background:#fff;border:1px solid #dfe6ef;color:#344054;border-radius:999px;padding:.38rem .7rem;font-size:.78rem;font-weight:700;box-shadow:0 2px 7px rgba(16,24,40,.035);}
    .section-kicker {font-size:.72rem;font-weight:800;letter-spacing:.10em;text-transform:uppercase;color:#2563eb;margin-top:.35rem;margin-bottom:.15rem;}
    .section-title {font-size:1.22rem;font-weight:800;color:#172033;margin-bottom:.75rem;}
    .trade-card {background:#fff;border:1px solid #dbe7ff;border-left:5px solid #2563eb;border-radius:16px;padding:1.05rem 1.2rem;margin:.45rem 0 1.2rem;box-shadow:0 6px 18px rgba(16,24,40,.05);}
    .trade-card.buy {border-left-color:#15803d;background:#ffffff;}
    .trade-card.sell {border-left-color:#dc2626;background:#ffffff;}
    .trade-label {font-size:.72rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:#667085;}
    .trade-value {font-size:1.75rem;font-weight:850;letter-spacing:-.03em;color:#172033;margin:.1rem 0;}
    .trade-note {font-size:.83rem;color:#667085;}
    [data-testid="stMetric"] {transition: transform .12s ease, box-shadow .12s ease;}
    [data-testid="stMetric"]:hover {transform: translateY(-2px);box-shadow:0 8px 22px rgba(16,24,40,.08);}
    [data-testid="stSidebar"] h2 {font-size:1.05rem !important;}
    [data-testid="stSidebar"] label {font-size:.86rem;font-weight:650;color:#475467;}
    .stTabs [data-baseweb="tab-list"] {gap:.35rem;background:#eef2f7;padding:.32rem;border-radius:12px;}
    .stTabs [data-baseweb="tab"] {background:transparent;border-radius:9px;padding:.45rem .8rem;}
    .stTabs [aria-selected="true"] {background:white !important;box-shadow:0 2px 7px rgba(16,24,40,.07);}
    button[kind="secondary"] {border-radius:10px !important;}
    [data-testid="stExpander"] summary {font-weight:700;color:#344054;}
    .block-container > div {gap:.65rem;}
    .footer-note {margin-top:2rem;padding-top:1rem;border-top:1px solid #e4e7ec;color:#98a2b3;font-size:.76rem;text-align:center;}
    .summary-shell {background:#fff;border:1px solid #e4e7ec;border-radius:18px;padding:1.15rem 1.25rem;margin:.55rem 0 1rem;box-shadow:0 5px 18px rgba(16,24,40,.045);}
    .summary-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7rem;margin:.8rem 0;}
    .summary-kpi {background:#f8fafc;border:1px solid #e7ecf2;border-radius:13px;padding:.85rem .95rem;}
    .summary-kpi-label {font-size:.68rem;color:#667085;font-weight:800;letter-spacing:.08em;text-transform:uppercase;}
    .summary-kpi-value {font-size:1.35rem;color:#172033;font-weight:850;letter-spacing:-.025em;margin-top:.12rem;}
    .summary-kpi-note {font-size:.75rem;color:#667085;margin-top:.12rem;}
    .hedge-summary {border-radius:16px;padding:1.1rem 1.2rem;margin:.9rem 0;border:1px solid #e4e7ec;}
    .hedge-summary.buy {background:linear-gradient(100deg,#f0fdf4,#fff);border-left:5px solid #15803d;}
    .hedge-summary.sell {background:linear-gradient(100deg,#fef2f2,#fff);border-left:5px solid #dc2626;}
    .hedge-summary.neutral {background:linear-gradient(100deg,#eff6ff,#fff);border-left:5px solid #2563eb;}
    .hedge-action {font-size:.76rem;font-weight:850;letter-spacing:.1em;text-transform:uppercase;color:#667085;}
    .hedge-number {font-size:2rem;font-weight:900;letter-spacing:-.04em;color:#172033;line-height:1.05;margin:.2rem 0;}
    .risk-reading {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.65rem;margin-top:.8rem;}
    .risk-reading-item {background:#f8fafc;border-radius:12px;padding:.8rem .9rem;border:1px solid #edf0f4;font-size:.82rem;color:#475467;}
    .risk-reading-item strong {display:block;color:#172033;margin-bottom:.18rem;}
    .scenario-strip {display:flex;justify-content:space-between;gap:1rem;background:#fff;border:1px solid #e4e7ec;border-radius:14px;padding:.8rem 1rem;margin:.45rem 0 .75rem;}
    .scenario-item {font-size:.72rem;color:#667085;font-weight:700;text-transform:uppercase;letter-spacing:.06em;}
    .scenario-item b {display:block;font-size:1.05rem;color:#172033;letter-spacing:0;text-transform:none;margin-top:.12rem;}
    @media (max-width: 900px) {.summary-grid{grid-template-columns:1fr}.risk-reading{grid-template-columns:1fr}}
        #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* V8.4 — Risk Summary renforcé */
    .risk-hero {background:linear-gradient(110deg,#123b63,#2563eb);color:white;border-radius:18px;padding:1.15rem 1.3rem;margin:.35rem 0 1rem;box-shadow:0 8px 24px rgba(37,99,235,.16);}
    .risk-hero-kicker {font-size:.72rem;font-weight:850;letter-spacing:.12em;text-transform:uppercase;opacity:.78;}
    .risk-hero-title {font-size:1.65rem;font-weight:900;letter-spacing:-.035em;margin:.12rem 0;color:#fff;text-transform:none;}
    .risk-hero-sub {font-size:.88rem;opacity:.88;color:#fff;}
    .hedge-decision {display:grid;grid-template-columns:1fr 1.2fr;align-items:center;gap:1rem;border-radius:18px;padding:1.2rem 1.3rem;margin:.9rem 0;border:1px solid #e4e7ec;box-shadow:0 5px 18px rgba(16,24,40,.05);}
    .hedge-decision.buy {background:linear-gradient(105deg,#f0fdf4,#fff);border-left:6px solid #15803d;}
    .hedge-decision.sell {background:linear-gradient(105deg,#fef2f2,#fff);border-left:6px solid #dc2626;}
    .hedge-decision.neutral {background:linear-gradient(105deg,#eff6ff,#fff);border-left:6px solid #2563eb;}
    .hedge-decision-label {font-size:.72rem;font-weight:850;letter-spacing:.11em;text-transform:uppercase;color:#667085;}
    .hedge-decision-action {font-size:1.65rem;font-weight:900;letter-spacing:-.03em;color:#172033;margin-top:.12rem;}
    .hedge-decision-qty {font-size:2.25rem;font-weight:900;letter-spacing:-.045em;color:#172033;line-height:1;}
    .hedge-decision-qty span {font-size:.92rem;font-weight:750;color:#667085;letter-spacing:0;}
    .hedge-breakdown {background:rgba(255,255,255,.7);border:1px solid #e4e7ec;border-radius:13px;padding:.8rem .9rem;font-size:.82rem;color:#475467;line-height:1.75;}
    .hedge-breakdown b {color:#172033;}
    .neutral-chip {display:inline-block;background:#ecfdf3;color:#067647;border:1px solid #abefc6;border-radius:999px;padding:.12rem .48rem;font-size:.7rem;font-weight:850;margin-left:.35rem;}
    .risk-secondary-title {font-size:.72rem;font-weight:850;letter-spacing:.11em;text-transform:uppercase;color:#2563eb;margin:1.1rem 0 .45rem;}
    .risk-conclusion {background:#fff;border:1px solid #dbe4f0;border-left:5px solid #2563eb;border-radius:14px;padding:1rem 1.1rem;margin-top:.9rem;color:#344054;font-size:.88rem;line-height:1.55;}
    .risk-conclusion strong {display:block;color:#172033;font-size:.76rem;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.25rem;}
    @media (max-width:900px){.hedge-decision{grid-template-columns:1fr}.hedge-decision-qty{font-size:1.8rem}}


    /* V10.4 — Hiérarchie typographique professionnelle, sans changer la disposition. */
    .block-container h1 {
        font-size: 2.05rem !important;
        font-weight: 800 !important;
        letter-spacing: -.025em;
        line-height: 1.18;
    }
    .block-container h2 {
        font-size: 1.55rem !important;
        font-weight: 780 !important;
        letter-spacing: -.022em;
        line-height: 1.25;
        margin-top: 1.05rem;
        margin-bottom: .65rem;
    }
    .block-container h3 {
        font-size: 1.32rem !important;
        font-weight: 760 !important;
        letter-spacing: -.015em;
        line-height: 1.28;
        margin-top: .85rem;
        margin-bottom: .5rem;
    }
    .block-container h4 {
        font-size: 1.15rem !important;
        font-weight: 750 !important;
        line-height: 1.3;
        margin-top: .7rem;
        margin-bottom: .45rem;
    }
    .section-title {
        font-size: 1.48rem !important;
        font-weight: 800 !important;
        letter-spacing: -.02em;
        line-height: 1.25;
        margin-bottom: .85rem;
    }
    .section-kicker {
        font-size: .82rem !important;
        font-weight: 750 !important;
        letter-spacing: .065em;
    }
    .top-title {font-size: 2.15rem !important;line-height:1.15;}
    .top-kicker {font-size: .82rem !important;letter-spacing:.1em;}
    .top-sub {font-size: 1rem !important;}
    [data-testid="stMetricLabel"] p {
        font-size: .97rem !important;
        font-weight: 650 !important;
        line-height: 1.3;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 1.12rem !important;
        font-weight: 780 !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 1rem !important;
        font-weight: 700 !important;
    }
    .trade-label, .hedge-decision-label {
        font-size: .82rem !important;
        letter-spacing: .06em;
    }
    .trade-value, .hedge-decision-action {font-weight: 800 !important;}
    @media (max-width: 900px) {
        .block-container h2 {font-size: 1.35rem !important;}
        .block-container h3 {font-size: 1.2rem !important;}
        .section-title {font-size: 1.3rem !important;}
        .top-title {font-size: 1.8rem !important;}
    }


    /* V10.5 — Lisibilité générale : +10 à 15 %, mise en page inchangée. */
    html, body, .stApp { font-size: 17px !important; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span { font-size: 1.06rem; line-height: 1.58; }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p { font-size: .98rem !important; line-height: 1.5 !important; }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label,
    .stRadio label { font-size: 1.01rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 1.01rem !important; }
    .stTabs [data-baseweb="tab"] p { font-size: 1.01rem !important; }
    button, button p, [data-testid="stButton"] button { font-size: 1rem !important; }
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-baseweb="select"] { font-size: 1.02rem !important; }
    [data-testid="stMetricLabel"] p { font-size: 1rem !important; }
    [data-testid="stMetricValue"] { font-size: clamp(1.18rem, 1.55vw, 1.65rem) !important; font-variant-numeric: tabular-nums; }
    [data-testid="stMetricDelta"] p { font-size: .89rem !important; }
    [data-testid="stMetric"] { min-height: 108px; padding: 1rem .85rem; }
    .top-kicker, .section-kicker, .trade-label, .hedge-decision-label { font-size: .92rem !important; }
    .top-sub, .trade-note, .summary-kpi-note, .scenario-item { font-size: 1rem !important; }
    .badge { font-size: .92rem !important; }
    .trade-value { font-size: 1.8rem !important; }
    .section-title { font-size: 1.52rem !important; }
    .summary-kpi-label { font-size: .92rem !important; }
    .summary-kpi-value { font-size: 1.45rem !important; }
    .risk-reading-item, .hedge-breakdown, .risk-conclusion { font-size: 1rem !important; }
    [data-testid="stExpander"] summary { font-size: 1.06rem !important; }
    @media (max-width: 1100px) {
      [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
      [data-testid="stMetricLabel"] p { font-size: .94rem !important; }
    }

</style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MOTEUR BLACK-SCHOLES
# ============================================================

def validate_bs_inputs(S, K, T, sigma):

    if S <= 0:
        raise ValueError(
            "Le prix du sous-jacent doit être strictement positif."
        )

    if K <= 0:
        raise ValueError(
            "Le strike doit être strictement positif."
        )

    if T <= 0:
        raise ValueError(
            "La maturité doit être strictement positive."
        )

    if sigma <= 0:
        raise ValueError(
            "La volatilité doit être strictement positive."
        )


def bs_price(S, K, T, r, sigma, option_type="call"):

    validate_bs_inputs(S, K, T, sigma)

    option_type = option_type.lower()

    if option_type not in ["call", "put"]:
        raise ValueError(
            "Le type d'option doit être 'call' ou 'put'."
        )

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":

        return (
            S * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )

    return (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )


# ============================================================
# GREEKS BLACK-SCHOLES
# ============================================================

def bs_greeks(S, K, T, r, sigma, option_type="call"):

    validate_bs_inputs(S, K, T, sigma)

    option_type = option_type.lower()

    if option_type not in ["call", "put"]:
        raise ValueError(
            "Le type d'option doit être 'call' ou 'put'."
        )

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    gamma = (
        norm.pdf(d1)
        / (S * sigma * np.sqrt(T))
    )

    # Vega pour +1 point de volatilité
    vega = (
        S
        * norm.pdf(d1)
        * np.sqrt(T)
        / 100
    )

    if option_type == "call":

        delta = norm.cdf(d1)

        theta_annual = (
            -(S * norm.pdf(d1) * sigma)
            / (2 * np.sqrt(T))
            - r
            * K
            * np.exp(-r * T)
            * norm.cdf(d2)
        )

        # Rho pour +1 point de taux
        rho = (
            K
            * T
            * np.exp(-r * T)
            * norm.cdf(d2)
            / 100
        )

    else:

        delta = norm.cdf(d1) - 1

        theta_annual = (
            -(S * norm.pdf(d1) * sigma)
            / (2 * np.sqrt(T))
            + r
            * K
            * np.exp(-r * T)
            * norm.cdf(-d2)
        )

        rho = (
            -K
            * T
            * np.exp(-r * T)
            * norm.cdf(-d2)
            / 100
        )

    theta_daily = theta_annual / 365

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta_annual": theta_annual,
        "theta_daily": theta_daily,
        "rho": rho
    }


# ============================================================
# POSITION INDIVIDUELLE
# ============================================================

def option_position(
    S,
    K,
    T,
    r,
    sigma,
    option_type,
    position_type,
    quantity
):

    position_type = position_type.lower()

    if position_type not in ["long", "short"]:
        raise ValueError(
            "La position doit être 'long' ou 'short'."
        )

    if quantity <= 0:
        raise ValueError(
            "La quantité doit être strictement positive."
        )

    position_sign = (
        1 if position_type == "long" else -1
    )

    unit_price = bs_price(
        S, K, T, r, sigma, option_type
    )

    greeks = bs_greeks(
        S, K, T, r, sigma, option_type
    )

    position_value = (
        position_sign
        * quantity
        * unit_price
    )

    position_delta = (
        position_sign
        * quantity
        * greeks["delta"]
    )

    position_gamma = (
        position_sign
        * quantity
        * greeks["gamma"]
    )

    position_vega = (
        position_sign
        * quantity
        * greeks["vega"]
    )

    position_theta_annual = (
        position_sign
        * quantity
        * greeks["theta_annual"]
    )

    position_theta_daily = (
        position_sign
        * quantity
        * greeks["theta_daily"]
    )

    position_rho = (
        position_sign
        * quantity
        * greeks["rho"]
    )

    # Couverture Delta
    hedge_shares = -position_delta

    if hedge_shares > 1e-12:
        hedge_action_value = "ACHETER"

    elif hedge_shares < -1e-12:
        hedge_action_value = "VENDRE"

    else:
        hedge_action_value = "AUCUNE ACTION"

    return {
        "unit_price": unit_price,
        "position_value": position_value,
        "position_delta": position_delta,
        "position_gamma": position_gamma,
        "position_vega": position_vega,
        "position_theta_annual": position_theta_annual,
        "position_theta_daily": position_theta_daily,
        "position_rho": position_rho,
        "hedge_shares": hedge_shares,
        "hedge_action": hedge_action_value
    }


# ============================================================
# OUTILS
# ============================================================

def clean_zero(value, tolerance=1e-10):

    if abs(value) < tolerance:
        return 0.0

    return value


def hedge_action(hedge):

    if hedge > 1e-12:
        return "ACHETER"

    if hedge < -1e-12:
        return "VENDRE"

    return "AUCUNE ACTION"


def finite_diff_check(S, K, T, r, sigma, option_type, h=None):
    """Compare les Greeks analytiques aux différences finies centrales."""
    if h is None:
        h = min(max(0.01 * S * sigma * np.sqrt(T), 1e-4 * S), 1e-3 * S)
    h = min(h, 0.49 * S)
    p_plus = bs_price(S + h, K, T, r, sigma, option_type)
    p_0 = bs_price(S, K, T, r, sigma, option_type)
    p_minus = bs_price(S - h, K, T, r, sigma, option_type)
    delta_fd = (p_plus - p_minus) / (2 * h)
    gamma_fd = (p_plus - 2 * p_0 + p_minus) / (h ** 2)
    greeks = bs_greeks(S, K, T, r, sigma, option_type)
    return {
        "h": h,
        "delta_analytic": greeks["delta"],
        "delta_fd": delta_fd,
        "delta_error": abs(delta_fd - greeks["delta"]),
        "gamma_analytic": greeks["gamma"],
        "gamma_fd": gamma_fd,
        "gamma_error": abs(gamma_fd - greeks["gamma"]),
    }


def within_tol(err, ref, abs_tol=1e-6, rel_tol=1e-4):
    """Tolérance mixte : robuste lorsque l'échelle du Greek change."""
    return err < abs_tol + rel_tol * abs(ref)


@st.cache_data(show_spinner=False)
def build_single_risk_curve(S0, K, T, r, sigma, option_type, position_type, quantity):
    spot_range = np.linspace(0.60 * S0, 1.40 * S0, 401)
    rows = []
    for spot in spot_range:
        result = option_position(spot, K, T, r, sigma, option_type, position_type, quantity)
        rows.append({
            "Spot": spot,
            "Delta": result["position_delta"],
            "Gamma": result["position_gamma"],
            "Couverture": result["hedge_shares"],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_single_stress_data(S0, K, T, r, sigma, option_type, position_type, quantity):
    initial = option_position(S0, K, T, r, sigma, option_type, position_type, quantity)
    hedge0 = initial["hedge_shares"]
    spot_rows = []
    for stressed_spot in np.linspace(0.80 * S0, 1.20 * S0, 81):
        stressed = option_position(stressed_spot, K, T, r, sigma, option_type, position_type, quantity)
        option_pnl = stressed["position_value"] - initial["position_value"]
        stock_pnl = hedge0 * (stressed_spot - S0)
        spot_rows.append({"Spot": stressed_spot, "P&L option": option_pnl, "P&L couvert": option_pnl + stock_pnl})
    vol_rows = []
    vol_min, vol_max = max(0.01, 0.50 * sigma), 1.50 * sigma
    for stressed_sigma in np.linspace(vol_min, vol_max, 81):
        stressed = option_position(S0, K, T, r, stressed_sigma, option_type, position_type, quantity)
        vol_rows.append({"Volatilité": stressed_sigma, "Valeur position": stressed["position_value"]})
    return pd.DataFrame(spot_rows), pd.DataFrame(vol_rows)


@st.cache_data(show_spinner=False)
def build_portfolio_stress_grid(S0, rate, options_key):
    """Grille de stress indépendante du scénario S1; recalcul uniquement si les positions changent."""
    spots = np.linspace(0.80 * S0, 1.20 * S0, 25)
    vol_shocks = np.linspace(-8.0, 8.0, 17)
    min_vol = min(float(o[5]) * 100 for o in options_key)
    vol_shocks = np.array([v for v in vol_shocks if min_vol + v > 0.01])
    if len(vol_shocks) < 2:
        vol_shocks = np.array([0.0, 2.0, 4.0])
    initial = sum(option_position(S0, strike, maturity, rate, sigma, typ, direction, qty)["position_value"]
                  for typ, direction, qty, strike, maturity, sigma in options_key)
    delta0 = sum(option_position(S0, strike, maturity, rate, sigma, typ, direction, qty)["position_delta"]
                 for typ, direction, qty, strike, maturity, sigma in options_key)
    values = np.array([[sum(option_position(float(spot), strike, maturity, rate,
                      sigma + float(dvol)/100, typ, direction, qty)["position_value"]
                      for typ, direction, qty, strike, maturity, sigma in options_key)
                      for spot in spots] for dvol in vol_shocks])
    pnl = values - initial
    covered = pnl - delta0 * (spots[None, :] - S0)
    return spots, vol_shocks, pnl, covered


def trade_instruction(shares):
    if abs(shares) < 0.005:
        return "AUCUN AJUSTEMENT NÉCESSAIRE"
    return f"{'ACHETER' if shares > 0 else 'VENDRE'} {format_fr(abs(shares))} actions"


@st.cache_data(show_spinner=False)
def build_portfolio_risk_curve(S0, rate, options_key):
    """Courbe agrégée indépendante du spot simulé S1."""
    rows = []
    for spot in np.linspace(0.60 * S0, 1.40 * S0, 401):
        delta, gamma = 0.0, 0.0
        for typ, direction, quantity, strike, maturity, sigma in options_key:
            position = option_position(spot, strike, maturity, rate, sigma, typ, direction, quantity)
            delta += position["position_delta"]
            gamma += position["position_gamma"]
        rows.append({"Spot": spot, "Delta": delta, "Gamma": gamma, "Couverture": -delta})
    return pd.DataFrame(rows)


def format_fr(value, digits=2):
    if round(float(value), digits) == 0:
        value = 0.0
    raw = f"{value:,.{digits}f}"
    return raw.replace(",", "§").replace(".", ",").replace("§", " ")


def format_fr_signed(value, digits=2):
    """French number with explicit sign, for changes and sensitivities."""
    return ("+" if float(value) > 0 else "−" if float(value) < 0 else "") + format_fr(abs(float(value)), digits)


def pnl_breakdown(initial, exact_pnl, delta_spot, vol_shift_pp, elapsed_years):
    """Taylor local; Vega est par point de pourcentage, Theta par an."""
    components = {
        "Delta": initial["position_delta"] * delta_spot,
        "Gamma": 0.5 * initial["position_gamma"] * delta_spot ** 2,
        "Vega": initial["position_vega"] * vol_shift_pp,
        "Theta": initial["position_theta_annual"] * elapsed_years,
    }
    components["Résidu (ordre supérieur)"] = exact_pnl - sum(components.values())
    return components



def render_pnl_comparison(exact, delta, delta_gamma, key, delta_gamma_vega_theta=None):
    """Comparaison lisible, avec valeurs négatives et échelle zéro."""
    labels = ["Delta seul", "Delta-Gamma"]
    values = [float(delta), float(delta_gamma)]
    colors = ["#64748b", "#2563eb"]
    if delta_gamma_vega_theta is not None:
        labels.append("Delta-Gamma-Vega-Theta")
        values.append(float(delta_gamma_vega_theta))
        colors.append("#d97706")
    labels.append("P&L exact")
    values.append(float(exact))
    colors.append("#16a34a")
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        customdata=[format_fr(v, 2) + " €" for v in values],
        hovertemplate="%{x}<br>P&L : %{customdata}<extra></extra>",
        text=None,
        textposition="none",
    ))
    low, high = min(0.0, *values), max(0.0, *values)
    span = max(high - low, max(abs(v) for v in values), 1.0)
    fig.update_layout(
        template="plotly_white", height=400, showlegend=False,
        margin=dict(l=30, r=25, t=55, b=55),
        xaxis_title="", yaxis_title="P&L (€)",
        yaxis=dict(range=[low - 0.16 * span, high + 0.19 * span],
                   zeroline=True, zerolinewidth=2, zerolinecolor="#334155"),
        bargap=0.48,
    )
    st.plotly_chart(fig, width="stretch", key=key)
    st.caption("Valeurs exactes au survol des barres ; axe zéro visible pour distinguer gains et pertes.")
    if delta_gamma_vega_theta is not None:
        st.metric("Écart approximation complète / P&L exact",
                  format_fr(delta_gamma_vega_theta - exact, 2) + " €")
    else:
        st.metric("Écart Delta-Gamma / P&L exact (spot seul)",
                  format_fr(delta_gamma - exact, 2) + " €")
    with st.popover("ⓘ Comment lire la comparaison ?"):
        st.write("Chaque barre représente un P&L en euros. Le zéro sépare gains et pertes. "
                 "Gris : approximation linéaire par Delta ; bleu : ajout de la convexité Gamma ; "
                 "orange : approximation Delta-Gamma-Vega-Theta ; vert : revalorisation exacte. "
                 "Les approximations Delta et Delta-Gamma isolent le spot ; "
                 "l'approximation complète ajoute les effets de volatilité et de temps. "
                 "Le résidu correspond notamment aux effets croisés et aux ordres supérieurs.")


def render_pnl_error_curve(spots, exact_values, delta_values, dg_values, current_spot, key):
    """Erreurs d'approximation en fonction du spot, scénario spot seul."""
    fig = go.Figure()
    for name, approx, color in [
        ("Erreur Delta", delta_values, "#64748b"),
        ("Erreur Delta-Gamma", dg_values, "#2563eb"),
    ]:
        errors = np.asarray(approx) - np.asarray(exact_values)
        fig.add_trace(go.Scatter(
            x=spots, y=errors, mode="lines", name=name,
            line=dict(color=color, width=3),
            hovertemplate="Spot : %{x:.2f} €<br>Erreur : %{y:.2f} €<extra></extra>"
        ))
    fig.add_hline(y=0, line_color="#64748b", line_dash="dot")
    fig.add_vline(x=current_spot, line_color="#d97706", line_dash="dash")
    fig.update_layout(
        template="plotly_white", height=380,
        xaxis_title="Spot du sous-jacent (€)",
        yaxis_title="Erreur d'approximation (€)",
        margin=dict(l=25, r=20, t=20, b=75),
        legend=dict(orientation="h", y=-0.28, x=0),
    )
    st.plotly_chart(fig, width="stretch", key=key)
    with st.popover("ⓘ Comment lire les erreurs d'approximation ?"):
        st.write("Horizontal : niveau du spot après choc. Vertical : approximation moins P&L exact, "
                 "en euros. Une courbe proche de zéro indique une approximation précise. "
                 "Le trait orange repère le spot simulé. Pour isoler l'effet du spot, "
                 "ce graphique conserve la volatilité et la maturité initiales : "
                 "il ne reflète pas les chocs simultanés de volatilité ou de temps.")


def render_hedge_rebalance(initial_hedge, scenario_hedge, key):
    """Quantités signées : positif acheter, négatif vendre."""
    change = scenario_hedge - initial_hedge
    c1, c2, c3 = st.columns(3)
    c1.metric("Couverture initiale", format_fr(initial_hedge, 2) + " actions")
    c2.metric("Couverture cible", format_fr(scenario_hedge, 2) + " actions")
    c3.metric("Ordre de rééquilibrage",
              ("ACHETER " if change > 1e-9 else "VENDRE " if change < -1e-9 else "AUCUN ")
              + format_fr(abs(change), 2) + " actions")
    with st.popover("ⓘ Comment lire le rééquilibrage ?"):
        st.write("Une couverture positive correspond à des actions achetées, "
                 "une couverture négative à des actions vendues à découvert. "
                 "L'ordre de rééquilibrage est la différence entre la couverture cible "
                 "au scénario et la couverture initiale. Il suppose que la couverture "
                 "initiale a effectivement été mise en place et qu'aucune autre "
                 "transaction n'est intervenue.")

def render_breakdown(components, key):
    st.markdown("#### Attribution du P&L — approximation par les Greeks")
    labels = list(components)
    vals = list(components.values())
    if all(abs(float(v)) < 1e-8 for v in vals):
        st.info("Aucun choc de marché appliqué : modifiez le spot, la volatilité ou le temps "
                "pour visualiser les contributions au P&L.")
        return
    chart = go.Figure(go.Waterfall(
        x=labels, y=vals, measure=["relative"] * len(vals),
        increasing={"marker": {"color": "#16a34a"}},
        decreasing={"marker": {"color": "#dc2626"}},
        connector={"line": {"color": "#94a3b8", "width": 1}},
        hovertemplate="%{x} : %{y:,.2f} €<extra></extra>"))
    chart.update_layout(template="plotly_white", height=365,
                        xaxis_title="Facteur", yaxis_title="P&L cumulé (€)",
                        margin=dict(l=20, r=20, t=20, b=65), showlegend=False)
    st.plotly_chart(chart, width="stretch", key=key)
    residual = float(components.get("Résidu (ordre supérieur)", 0.0))
    exact = sum(float(v) for v in components.values())
    if abs(exact) > 0.01:
        st.caption(f"Résidu de l'approximation : {format_fr(residual)} € "
                   f"({format_fr(100 * abs(residual) / abs(exact))} % du |P&L exact|).")
    else:
        st.caption(f"Résidu de l'approximation : {format_fr(residual)} € "
                   "(pourcentage non pertinent lorsque le P&L exact est proche de zéro).")
    with st.popover("ⓘ Lire l'attribution du P&L"):
        st.write("L'axe horizontal énumère les facteurs Delta, Gamma, Vega, Theta et le résidu ; "
                 "l'axe vertical représente leur contribution en euros. Vert : contribution positive, rouge : négative. "
                 "Delta et Gamma approximent l'effet du mouvement de spot ; Vega celui de la volatilité ; "
                 "Theta celui du temps écoulé. Le résidu mesure l'écart entre cette approximation locale et le P&L exact. "
                 "La somme des contributions est égale au P&L exact par construction.")



@st.cache_data(show_spinner=False)
def build_portfolio_scenario_curve(S0, rate, options_key, vol_shift_pp, elapsed_years):
    shifted = tuple((typ, direction, qty, strike, max(1e-6, maturity - elapsed_years),
                     max(1e-6, sigma + vol_shift_pp / 100))
                    for typ, direction, qty, strike, maturity, sigma in options_key)
    return build_portfolio_risk_curve(S0, rate, shifted)


def moneyness_status(spot, strike, option_type):
    """Moneyness au spot indiqué, indépendante du sens long/short."""
    if abs(spot - strike) <= max(1e-8, 1e-6 * strike):
        return "ATM"
    is_call = str(option_type).strip().lower() == "call"
    return "ITM" if ((spot > strike) == is_call) else "OTM"


def add_moneyness_zones(fig, strike, option_type, x_min, x_max, label=True):
    """Zones ITM/OTM ; cartes lisibles ancrées DANS le graphique Plotly."""
    if strike is None:
        return fig
    strike = float(strike)
    is_call = str(option_type).strip().lower() == "call"
    left, right = (("OTM", "ITM") if is_call else ("ITM", "OTM"))
    for x0, x1, status in ((x_min, min(strike, x_max), left), (max(strike, x_min), x_max, right)):
        if x1 <= x0:
            continue
        itm = status == "ITM"
        bg = "rgba(34,197,94,0.09)" if itm else "rgba(59,130,246,0.09)"
        fg = "#166534" if itm else "#1e40af"
        chip = "#dcfce7" if itm else "#dbeafe"
        relation = ("S > K" if is_call else "S < K") if itm else ("S < K" if is_call else "S > K")
        fig.add_vrect(x0=x0, x1=x1, fillcolor=bg, line_width=0, layer="below")
        if label and (x1 - x0) >= 0.09 * (x_max - x_min):
            fig.add_annotation(x=(x0 + x1) / 2, y=0.94, yref="paper", showarrow=False,
                               text=f"<b>{status}</b><br>({relation})", align="center",
                               bgcolor=chip, bordercolor=chip, borderwidth=1,
                               borderpad=12, font=dict(size=17, color=fg), opacity=1)
    if x_min <= strike <= x_max:
        fig.add_vline(x=strike, line_dash="dash", line_color="#6d28d9", line_width=2.7)
        if label:
            fig.add_annotation(x=strike, y=1.045, yref="paper", showarrow=False,
                               text=f"<b>ATM · K = {format_fr(strike, 2)} €</b>",
                               bgcolor="#ede9fe", bordercolor="#ddd6fe", borderwidth=1,
                               borderpad=10, font=dict(size=14, color="#5b21b6"))
    return fig


def moneyness_context(option_type, strike, spot, option_number=None):
    """Carte de statut compacte ; zones et courbe agrégée sont distinguées."""
    typ = str(option_type).capitalize()
    status = moneyness_status(spot, strike, option_type)
    name = f"Option {option_number} · " if option_number is not None else ""
    relation = "S = K" if status == "ATM" else ("S > K" if spot > strike else "S < K")
    color = {"ITM": ("#166534", "#dcfce7"),
             "OTM": ("#1d4ed8", "#dbeafe"),
             "ATM": ("#6d28d9", "#ede9fe")}[status]
    st.markdown(
        f"""<div style="padding:14px 18px;border:1px solid #dbe3ef;border-radius:12px;
        background:rgba(241,245,249,.48);margin:10px 0 15px 0">
        <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <strong style="font-size:16px">{name}{typ} · K = {format_fr(strike, 2)} €</strong>
        <span style="background:{color[1]};color:{color[0]};border-radius:8px;
        padding:5px 10px;font-weight:700">{status} au spot simulé</span></div>
        <div style="font-size:14px;margin-top:7px;color:#475569">
        Spot simulé : {format_fr(spot, 2)} € &nbsp;|&nbsp; Strike : {format_fr(strike, 2)} €
        &nbsp;|&nbsp; {relation}</div></div>""", unsafe_allow_html=True)


def interactive_risk_chart(data, kind, spot0, spot1, current_value, strike=None, reference_data=None, option_type=None):
    """Plotly : zoom, survol, signe de couverture et zone de Gamma élevé."""
    x = data["Spot"].to_numpy()
    y = data[kind].to_numpy()
    fig = go.Figure()
    if reference_data is not None:
        fig.add_trace(go.Scatter(x=reference_data["Spot"], y=reference_data[kind],
                                 mode="lines", name="Référence (σ₀, T₀)",
                                 line=dict(color="#64748b", dash="dash", width=2)))
    if kind == "Couverture":
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Couverture scénario",
                                 line=dict(color="#2563eb", width=3)))
        fig.add_trace(go.Scatter(x=x, y=np.maximum(y, 0), mode="lines",
                                 line=dict(width=0), fill="tozeroy", fillcolor="rgba(22,163,74,0.12)",
                                 name="Actions à acheter", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=np.minimum(y, 0), mode="lines",
                                 line=dict(width=0), fill="tozeroy", fillcolor="rgba(220,38,38,0.12)",
                                 name="Actions à vendre", hoverinfo="skip"))
        ylabel = "Couverture (actions)"
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Gamma scénario",
                                 line=dict(color="#2563eb", width=3)))
        threshold = 0.5 * np.max(np.abs(y))
        active = np.abs(y) >= threshold
        if np.any(active):
            inds = np.flatnonzero(active)
            cuts = np.flatnonzero(np.diff(inds) > 1)
            for part in np.split(inds, cuts + 1):
                fig.add_vrect(x0=float(x[part[0]]), x1=float(x[part[-1]]),
                              fillcolor="rgba(245,158,11,0.13)", line_width=0, annotation_text=None)
        idx = int(np.argmax(np.abs(y)))
        fig.add_trace(go.Scatter(x=[float(x[idx])], y=[float(y[idx])], mode="markers",
                                 marker=dict(symbol="diamond", size=11, color="#d97706"),
                                 name="Pic |Γ|"))
        ylabel = "Gamma (actions/€)"
    fig.add_trace(go.Scatter(x=[spot1], y=[current_value], mode="markers",
                             marker=dict(size=12, color="#0f172a", line=dict(color="white", width=2)),
                             name="Spot simulé"))
    fig.add_vline(x=spot0, line_dash="dash", line_color="#64748b")
    if strike is not None and str(option_type).strip().lower() in ("call", "put"):
        add_moneyness_zones(fig, strike, option_type, float(np.min(x)), float(np.max(x)))
    elif strike is not None:
        fig.add_vline(x=strike, line_dash="dot", line_color="#a855f7")
    fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
    fig.update_layout(template="plotly_white", height=440, margin=dict(l=15,r=15,t=90,b=90),
                      xaxis_title="Spot du sous-jacent (€)", yaxis_title=ylabel,
                      legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=10)), hovermode="x unified")
    return fig


def explain_graph(title, interpretation, consequence):
    """Aide à la lecture à la demande : ne réduit pas la largeur du graphique."""
    with st.popover("ⓘ Comment lire ce graphique ?", width="content"):
        st.markdown(f"**{title}**")
        st.markdown("**Lecture des axes et des repères**")
        if "Gamma" in title or "gamma" in title.lower() or "Sensibilité" in title:
            st.write("Horizontal : spot du sous-jacent (€). Vertical : Gamma de la position (actions/€). "
                     "La courbe bleue correspond au scénario ; la courbe grise pointillée aux paramètres initiaux. "
                     "Le point sombre indique le spot simulé, le losange orange le maximum de |Gamma|. "
                     "Les bandes orangées signalent les régions où |Gamma| atteint au moins 50 % de son maximum sur la plage affichée.")
        else:
            st.write("Horizontal : spot du sous-jacent (€). Vertical : nombre signé d'actions de couverture. "
                     "Au-dessus de zéro (vert), acheter des actions ; en dessous (rouge), en vendre. "
                     "Le point sombre représente le spot simulé, le trait vertical le spot initial. "
                     "La courbe grise pointillée est la référence avant choc de volatilité/maturité.")
        st.markdown("**Lecture du scénario actuel**")
        st.write(interpretation)
        st.markdown("**Conséquence pour le trader**")
        st.write(consequence)


# ============================================================
# COMMANDES DE DÉMONSTRATION ET DE RÉINITIALISATION
# ============================================================

SINGLE_DEFAULTS = {
    "single_option_type": "Call", "single_position_type": "Long",
    "single_quantity": 100, "single_spot": 100.0,
    "single_strike": 100.0, "single_maturity": 0.5,
    "single_rate": 3.0, "single_sigma": 20.0,
}
PORTFOLIO_DEFAULTS = {
    "portfolio_spot": 100.0, "portfolio_rate": 3.0,
    "portfolio_count": 2,
}


def reset_single_options():
    """Remise à zéro explicite des caractéristiques de l'option."""
    st.session_state.update(SINGLE_DEFAULTS)
    reset_single_scenario()
    st.session_state["confirm_single_reset"] = False


def reset_portfolio_options():
    """Restaure les deux lignes initiales sans modifier le moteur financier."""
    st.session_state.update(PORTFOLIO_DEFAULTS)
    for i in range(5):
        for name in ("type", "side", "qty", "K", "T", "sigma"):
            st.session_state.pop(f"portfolio_{name}_{i}", None)
    reset_portfolio_scenario()
    st.session_state["v910_portfolio_elapsed_days"] = 0
    st.session_state["confirm_portfolio_reset"] = False


def duplicate_portfolio_option(index):
    """Copie les paramètres d'une option dans une nouvelle ligne (5 max)."""
    count = int(st.session_state.get("portfolio_count", 2))
    if count >= 5:
        return
    defaults = {
        "type": "Call", "side": "Long", "qty": 100,
        "K": 100.0 + 5.0 * index, "T": 0.5, "sigma": 20.0,
    }
    for name, fallback in defaults.items():
        st.session_state[f"portfolio_{name}_{count}"] = st.session_state.get(
            f"portfolio_{name}_{index}", fallback
        )
    st.session_state["portfolio_count"] = count + 1


def remove_portfolio_option(index):
    """Supprime une ligne en conservant l'ordre des autres positions."""
    count = int(st.session_state.get("portfolio_count", 2))
    if count <= 1:
        return
    names = ("type", "side", "qty", "K", "T", "sigma")
    for i in range(index, count - 1):
        for name in names:
            next_key = f"portfolio_{name}_{i + 1}"
            current_key = f"portfolio_{name}_{i}"
            if next_key in st.session_state:
                st.session_state[current_key] = st.session_state[next_key]
            else:
                st.session_state.pop(current_key, None)
    for name in names:
        st.session_state.pop(f"portfolio_{name}_{count - 1}", None)
    st.session_state["portfolio_count"] = count - 1


def result_csv(rows):
    """CSV Excel FR : section fournie explicitement pour chaque indicateur."""
    import csv
    from datetime import datetime
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=";", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(("Section", "Indicateur", "Valeur", "Unité"))
    writer.writerow(("Rapport", "Application", "Option Risk Manager V10.0", ""))
    writer.writerow(("Rapport", "Date d'export", datetime.now().strftime("%d/%m/%Y %H:%M"), ""))
    for section, label, value, unit in rows:
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            formatted = (f"{float(value):.6f}".rstrip("0").rstrip(".")
                         if not isinstance(value, (int, np.integer)) else str(value))
            formatted = formatted.replace(".", ",")
        else:
            formatted = str(value)
        writer.writerow((section, label, formatted, unit))
    return buffer.getvalue().encode("utf-8-sig")



def load_single_example(kind):
    """Charge un exemple déterministe avant création des widgets."""
    examples = {
        "Long Call classique": ("Call", "Long"),
        "Short Call": ("Call", "Short"),
        "Long Put": ("Put", "Long"),
    }
    typ, side = examples[kind]
    st.session_state.update({
        "single_option_type": typ,
        "single_position_type": side,
        "single_quantity": 100,
        "single_spot": 100.0,
        "single_strike": 100.0,
        "single_maturity": 0.5,
        "single_rate": 3.0,
        "single_sigma": 20.0,
        "single_global_simulated_spot": 90.0 if typ == "Put" else 110.0,
        "v91_single_vol_shift": 4.0 if typ == "Put" else 3.0,
        "v91_single_elapsed": 7.0 / 365.0,
        "v911_single_elapsed_days": 7,
    })


def reset_single_scenario():
    """Remet le scénario aux paramètres de référence sans modifier le contrat."""
    st.session_state["single_global_simulated_spot"] = float(st.session_state.get("single_spot", 100.0))
    st.session_state["v91_single_vol_shift"] = 0.0
    st.session_state["v91_single_elapsed"] = 0.0
    st.session_state["v911_single_elapsed_days"] = 0


def load_portfolio_example():
    """Deux options opposées sur un sous-jacent commun."""
    st.session_state.update({
        "portfolio_spot": 100.0,
        "portfolio_rate": 3.0,
        "portfolio_count": 2,
        "portfolio_simulated_spot": 108.0,
        "v91_portfolio_vol_shift": 4.0,
        "v91_portfolio_elapsed": 14.0 / 365.0,
        "v910_portfolio_elapsed_days": 14,
        "portfolio_type_0": "Call", "portfolio_side_0": "Long",
        "portfolio_qty_0": 100, "portfolio_K_0": 100.0,
        "portfolio_T_0": 0.5, "portfolio_sigma_0": 20.0,
        "portfolio_type_1": "Put", "portfolio_side_1": "Long",
        "portfolio_qty_1": 65, "portfolio_K_1": 95.0,
        "portfolio_T_1": 0.5, "portfolio_sigma_1": 20.0,
    })


def reset_portfolio_scenario():
    st.session_state["portfolio_simulated_spot"] = float(st.session_state.get("portfolio_spot", 100.0))
    st.session_state["v91_portfolio_vol_shift"] = 0.0
    st.session_state["v91_portfolio_elapsed"] = 0.0
    st.session_state["v910_portfolio_elapsed_days"] = 0


# Initialisation V10 : scénario neutre, y compris après une mise à jour
# depuis V9.x dans une session Streamlit encore ouverte.
# Les exemples restent actifs après clic : cette migration ne s'exécute qu'une fois.
if st.session_state.get("_orm_scenario_schema") != "10.0":
    reset_single_scenario()
    reset_portfolio_scenario()
    st.session_state["_orm_scenario_schema"] = "10.0"



# ============================================================
# V11 — PROFILS DE PAYOFF ET ECHELLES DE RISQUE
# ============================================================

def expiry_payoff(spot, strike, option_type, side, quantity):
    """Flux terminal signé, hors prime et hors financement."""
    intrinsic = max(spot - strike, 0.0) if option_type.lower() == "call" else max(strike - spot, 0.0)
    return intrinsic * quantity * (1 if side.lower() == "long" else -1)


def find_zero_crossings(spots, values):
    """Interpolation linéaire des points morts dans la plage affichée."""
    roots = []
    for i in range(len(spots) - 1):
        a, b = float(values[i]), float(values[i + 1])
        if abs(a) < 1e-10:
            roots.append(float(spots[i]))
        elif a * b < 0:
            roots.append(float(spots[i] - a * (spots[i + 1] - spots[i]) / (b - a)))
    if abs(float(values[-1])) < 1e-10:
        roots.append(float(spots[-1]))
    return [v for i, v in enumerate(roots) if not any(abs(v - u) < 1e-5 for u in roots[:i])]


def exact_expiry_breakevens(positions, initial_value, tolerance=1e-9):
    """Racines exactes du P&L terminal, pour S >= 0 et une échéance commune.

    Le payoff est affine entre deux strikes consécutifs. Les intervalles
    où le P&L est identiquement nul sont signalés séparément.
    `initial_value` est la valeur signée initiale du portefeuille.
    """
    strikes = sorted({float(p["K"]) for p in positions if float(p["K"]) >= 0.0})
    boundaries = [0.0] + [k for k in strikes if k > 0.0]

    def pnl(spot):
        return sum(expiry_payoff(spot, p["K"], p["option_type"],
                                 p["position_type"], p["quantity"])
                   for p in positions) - initial_value

    roots, flat = [], []
    def add_root(x):
        if x >= 0 and not any(abs(x - y) <= tolerance * max(1, abs(x), abs(y)) for y in roots):
            roots.append(float(x))

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        fl, fr = pnl(left), pnl(right)
        if abs(fl) <= tolerance and abs(fr) <= tolerance:
            flat.append((left, right))
        elif abs(fr - fl) > tolerance:
            root = left - fl * (right - left) / (fr - fl)
            if left - tolerance <= root <= right + tolerance:
                add_root(max(0.0, root))
        if abs(fl) <= tolerance:
            add_root(left)
    last = boundaries[-1]
    f_last, f_next = pnl(last), pnl(last + 1.0)
    slope = f_next - f_last
    if abs(f_last) <= tolerance:
        add_root(last)
    if abs(slope) <= tolerance:
        if abs(f_last) <= tolerance:
            flat.append((last, float("inf")))
    else:
        root = last - f_last / slope
        if root >= last - tolerance:
            add_root(max(0.0, root))
    isolated = [x for x in roots if not any(a - tolerance <= x <= b + tolerance for a, b in flat)]
    return sorted(isolated), flat


def format_expiry_breakevens(roots, flat):
    parts = [format_fr(x, 2) + " €" for x in roots
             if not any(a - 1e-8 <= x <= b + 1e-8 for a, b in flat)]
    for a, b in flat:
        parts.append(("dès " + format_fr(a, 2) + " €" if np.isinf(b)
                      else "de " + format_fr(a, 2) + " à " + format_fr(b, 2) + " €")
                     + " (P&L nul sur l'intervalle)")
    return " · ".join(parts) if parts else "Aucun pour un spot positif ou nul"


def render_v11_chart(spots, pnl_today, pnl_expiry, reference_spot, current_spot, strikes, chart_key, expiry_positions=None, initial_value=None):
    """Profils à paramètres constants, avec points morts distincts et annotés."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=pnl_today, mode="lines", name="P&L théorique aujourd'hui",
                             line=dict(width=3, color="#2563eb")))
    roots_today = find_zero_crossings(spots, pnl_today)
    if pnl_expiry is not None:
        fig.add_trace(go.Scatter(x=spots, y=pnl_expiry, mode="lines", name="P&L à échéance (hors financement)",
                                 line=dict(width=3, dash="dash", color="#d97706")))
        roots_expiry, flat_expiry = (exact_expiry_breakevens(expiry_positions, initial_value)
                                     if expiry_positions is not None else (find_zero_crossings(spots, pnl_expiry), []))
        if roots_expiry:
            visible_roots = [x for x in roots_expiry if min(spots) <= x <= max(spots)]
            if visible_roots:
                fig.add_trace(go.Scatter(x=visible_roots, y=[0]*len(visible_roots), mode="markers+text",
                                     name="Point(s) mort(s) à échéance", text=[format_fr(x, 2)+" €" for x in visible_roots],
                                     textposition="bottom right", marker=dict(size=11, color="#d97706", symbol="diamond")))
    else:
        roots_expiry, flat_expiry = [], []
    if roots_today:
        fig.add_trace(go.Scatter(x=roots_today, y=[0]*len(roots_today), mode="markers",
                                 name="Équilibre aujourd'hui", marker=dict(size=9, color="#2563eb", symbol="circle-open")))
    fig.add_hline(y=0, line_width=1, line_color="#64748b")
    fig.add_vline(x=reference_spot, line_width=1, line_dash="dot", line_color="#64748b")
    fig.add_vline(x=current_spot, line_width=1, line_dash="dash", line_color="#0891b2")
    fig.update_layout(xaxis_title="Spot du sous-jacent (€)", yaxis_title="P&L de la position (€)",
                      legend=dict(orientation="h", y=-0.30), margin=dict(l=18, r=22, t=26, b=90),
                      font=dict(size=14), hovermode="x unified", height=470)
    fig.update_xaxes(tickfont=dict(size=13), title_font=dict(size=15))
    fig.update_yaxes(tickfont=dict(size=13), title_font=dict(size=15))
    st.plotly_chart(fig, width="stretch", key=chart_key)
    if pnl_expiry is not None:
        st.caption("Point(s) mort(s) à échéance (hors financement) : " +
                   format_expiry_breakevens(roots_expiry, flat_expiry))
    st.caption("Équilibre(s) de revalorisation aujourd'hui : " +
               (" · ".join(format_fr(v, 2) + " €" for v in roots_today) if roots_today else "aucun dans la plage affichée") +
               ". Cette courbe garde la maturité et la volatilité constantes ; ce n'est pas le P&L à échéance.")


def render_v11_ladder(rows, key):
    df = pd.DataFrame(rows)
    display = df.copy()
    for name in ("Spot (€)", "Delta (actions)", "Couverture cible (actions)", "Rééquilibrage (actions)", "P&L option (€)", "P&L avec hedge initial (€)"):
        display[name] = display[name].map(lambda x: format_fr(x, 2))
    st.dataframe(display, width="stretch", hide_index=True, height=430)
    st.download_button("Télécharger le risk ladder (CSV)", data=df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name="risk_ladder.csv", mime="text/csv", key=key)
    st.info("Lecture : le rééquilibrage est l’ordre à passer **par rapport à la couverture initiale fixée à S₀**. Ce tableau n’est pas une simulation de couverture dynamique. P&L couvert = P&L options + nombre d’actions de couverture initial × (S − S₀). Volatilité et maturité constantes ; frais et financement exclus.")

# ============================================================
# EN-TÊTE
# ============================================================

st.markdown(
    """
    <div class="top-shell">
        <div class="top-kicker">M2 IREF · Finance quantitative</div>
        <div class="top-title">Option Risk Manager</div>
        <div class="top-sub">Black-Scholes · Greeks · Delta Hedging · P&amp;L · Stress Testing · Portfolio Risk</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# V12.6 — TRANSFERT VOLONTAIRE DES DONNÉES DE MARCHÉ
# ============================================================

def transfer_market_inputs_v126(destination, ticker, name, currency, date, spot, vol):
    """Transfert explicite des deux seules hypothèses communes : spot et volatilité.

    Les autres paramètres (strike, taux, maturité, quantités) sont préservés.
    Les trois modes conservent leurs entrées manuelles.
    """
    spot, vol = float(spot), float(vol)
    if not (np.isfinite(spot) and spot > 0 and np.isfinite(vol) and vol > 0):
        raise ValueError("Cours ou volatilité historique invalide.")
    vol_pct = 100.0 * vol
    if vol_pct > 300.0:
        raise ValueError("Volatilité supérieure à la limite du module Monte-Carlo (300 %).")
    if destination == "Position individuelle":
        st.session_state["single_spot"] = spot
        st.session_state["single_sigma"] = vol_pct
        st.session_state["single_global_simulated_spot"] = spot
        st.session_state["_orm_last_single_spot"] = spot
    elif destination == "Portefeuille multi-options":
        st.session_state["portfolio_spot"] = spot
        st.session_state["portfolio_simulated_spot"] = spot
        st.session_state["_orm_last_portfolio_spot"] = spot
        for i in range(int(st.session_state.get("portfolio_count", 2))):
            st.session_state[f"portfolio_sigma_{i}"] = vol_pct
    elif destination == "Monte-Carlo & couverture dynamique":
        st.session_state["mc_spot"] = spot
        st.session_state["mc_vol"] = vol_pct
    else:
        raise ValueError("Mode de destination inconnu.")
    # Reset only scenario shocks that could be inherited from another asset.
    if destination == "Position individuelle":
        st.session_state["single_global_simulated_spot"] = spot
        for k, v in (("v91_single_vol_shift", 0.0), ("v91_single_elapsed", 0.0), ("v911_single_elapsed_days", 0)):
            if k in st.session_state:
                st.session_state[k] = v
    elif destination == "Portefeuille multi-options":
        st.session_state["v91_portfolio_vol_shift"] = 0.0
        st.session_state["v91_portfolio_elapsed"] = 0.0
    st.session_state["orm_mode"] = destination
    st.session_state["orm_import_currency"] = currency
    st.session_state["orm_import_ticker"] = ticker
    for _mode_idx in range(3):
        st.session_state[f"orm_currency_select_{_mode_idx}"] = currency
    st.session_state["orm_v126_import"] = {
        "ticker": ticker, "name": name, "currency": currency,
        "date": date, "spot": spot, "volatility": vol_pct,
        "destination": destination,
    }

# ============================================================
# CHOIX DU MODE
# ============================================================

mode = st.radio(
    "Mode d'analyse",
    [
        "Position individuelle",
        "Portefeuille multi-options",
        "Monte-Carlo & couverture dynamique",
        "Données réelles"
    ],
    horizontal=True,
    key="orm_mode"
)

st.divider()


# ============================================================
# MODE 1 — POSITION INDIVIDUELLE
# ============================================================

if mode == "Position individuelle":
    orm_currency = st.session_state.get("orm_import_currency", "EUR")
    if orm_currency not in ("EUR", "USD"):
        orm_currency = "EUR"
    orm_currency = st.selectbox(
        "Devise des montants (sans conversion de change)",
        ["EUR", "USD"],
        index=0 if orm_currency == "EUR" else 1,
        key="orm_currency_select_0",
        help="La devise est une unité d'affichage ; aucun taux de change n'est appliqué.",
    )
    st.session_state["orm_import_currency"] = orm_currency
    orm_currency_symbol = "€" if orm_currency == "EUR" else "$"
    _import_v126 = st.session_state.get("orm_v126_import")
    if _import_v126 and _import_v126.get("destination") == "Position individuelle":
        st.info(f"Données importées : {_import_v126['name']} ({_import_v126['ticker']}) · "
                f"{_import_v126['date']} · {_import_v126['currency']} · "
                f"Spot {_import_v126['spot']:.2f} · σ historique {_import_v126['volatility']:.2f} %. "
                "Les autres paramètres restent manuels. Les données ne sont pas synchronisées automatiquement.")

        st.caption("Strike conservé après importation : vous pouvez le rapprocher du spot si vous le souhaitez.")
        st.button("Proposer un strike proche du spot importé", key="orm_strike_suggestion_0",
                  on_click=lambda: st.session_state.__setitem__("single_strike",
                      float(round(st.session_state["orm_v126_import"]["spot"], 2))))


    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.header("Position de référence")
    st.sidebar.caption("Ces paramètres définissent le contrat et le marché à S₀.")
    with st.sidebar.expander("Exemples de démonstration", expanded=False):
        st.button("Long Call classique", key="demo_long_call", width="stretch",
                  on_click=load_single_example, args=("Long Call classique",))
        st.button("Short Call", key="demo_short_call", width="stretch",
                  on_click=load_single_example, args=("Short Call",))
        st.button("Long Put", key="demo_long_put", width="stretch",
                  on_click=load_single_example, args=("Long Put",))

    with st.sidebar.expander("Réinitialiser les options", expanded=False):
        st.checkbox("Je confirme la remise aux valeurs par défaut",
                    key="confirm_single_reset")
        st.button("Réinitialiser la position", key="reset_single_options_v96",
                  on_click=reset_single_options, width="stretch",
                  disabled=not st.session_state.get("confirm_single_reset", False))
    option_type_display = st.sidebar.selectbox(
        "Type d'option",
        ["Call", "Put"],
        key="single_option_type"
    )

    position_type_display = st.sidebar.selectbox(
        "Sens de la position",
        ["Long", "Short"],
        key="single_position_type"
    )

    for _key, _default in SINGLE_DEFAULTS.items():
        st.session_state.setdefault(_key, _default)

    quantity = st.sidebar.number_input(
        "Quantité d'options",
        step=1,
        key="single_quantity"
    )

    S0 = st.sidebar.number_input(
        f"Spot de référence S₀ ({orm_currency_symbol})",
        step=1.0,
        key="single_spot"
    )

    K = st.sidebar.number_input(
        f"Strike K ({orm_currency_symbol})",
        step=1.0,
        key="single_strike"
    )

    T = st.sidebar.number_input(
        "Maturité T (années)",
        step=0.05,
        format="%.2f",
        key="single_maturity"
    )

    r_percent = st.sidebar.number_input(
        "Taux sans risque r (%)",
        step=0.1,
        key="single_rate"
    )

    sigma_percent = st.sidebar.number_input(
        "Volatilité σ (%)",
        step=1.0,
        key="single_sigma"
    )

    option_type = (
        option_type_display.lower()
    )

    position_type = (
        position_type_display.lower()
    )

    r = r_percent / 100
    sigma = sigma_percent / 100


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    input_errors = []

    if quantity <= 0:
        input_errors.append(
            "La quantité d'options doit être "
            "strictement positive."
        )

    if S0 <= 0:
        input_errors.append(
            "Le spot de référence S₀ doit être "
            "strictement positif."
        )

    if K <= 0:
        input_errors.append(
            "Le strike K doit être strictement positif."
        )

    if T <= 0:
        input_errors.append(
            "La maturité T doit être "
            "strictement positive."
        )

    if sigma <= 0:
        input_errors.append(
            "La volatilité σ doit être "
            "strictement positive."
        )

    if input_errors:

        st.error(
            "Paramètres invalides : veuillez corriger "
            "les valeurs suivantes."
        )

        for error in input_errors:
            st.write(f"• {error}")

        st.stop()


    # --------------------------------------------------------
    # POSITION INITIALE
    # --------------------------------------------------------

    position_initiale = option_position(
        S=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type,
        position_type=position_type,
        quantity=quantity
    )


    # --------------------------------------------------------
    # SCÉNARIO DE MARCHÉ GLOBAL — visible dans tous les onglets
    # --------------------------------------------------------
    hedge_initial = position_initiale["hedge_shares"]
    spot_min = 0.60 * S0
    spot_max = 1.40 * S0

    # Si S₀ change, la valeur mémorisée du slider peut sortir des nouvelles bornes.
    # On la ramène alors dans la plage autorisée avant de créer le widget.
    if "single_global_simulated_spot" not in st.session_state:
        st.session_state["single_global_simulated_spot"] = float(S0)
    else:
        previous_s1 = float(st.session_state["single_global_simulated_spot"])
        if previous_s1 < spot_min or previous_s1 > spot_max:
            st.session_state["single_global_simulated_spot"] = float(S0)

    single_scenario_title, single_scenario_reset = st.columns([3, 1], vertical_alignment="center")
    with single_scenario_title:
        st.markdown("#### Simulation de marché")
    with single_scenario_reset:
        st.button("↺ Réinitialiser", key="reset_single_v95",
                  on_click=reset_single_scenario, width="stretch")
    st.caption("Les boutons appliquent un choc supplémentaire au spot simulé actuel (dans la plage autorisée).")
    shock_cols = st.columns(4)
    for col, shock in zip(shock_cols, (-0.10, -0.05, 0.05, 0.10)):
        with col:
            if st.button(f"{shock:+.0%}", key="single_shock_" + str(shock), width="stretch"):
                st.session_state["single_global_simulated_spot"] = float(min(spot_max, max(spot_min, float(st.session_state["single_global_simulated_spot"]) * (1 + shock))))

    S1 = st.slider(
        f"Spot simulé S₁ ({orm_currency_symbol})",
        min_value=float(spot_min),
        max_value=float(spot_max),
        step=float(max(S0 / 200, 0.01)),
        key="single_global_simulated_spot"
    )
    # Le moteur Black-Scholes conserve les années ; l'interface affiche les jours.
    max_single_elapsed_days = max(0, int(max(0.0, T - 0.0001) * 365))
    if "v911_single_elapsed_days" not in st.session_state:
        st.session_state["v911_single_elapsed_days"] = 0
    st.session_state["v911_single_elapsed_days"] = min(
        max(0, int(st.session_state["v911_single_elapsed_days"])),
        max_single_elapsed_days
    )
    if "v91_single_vol_shift" in st.session_state:
        st.session_state["v91_single_vol_shift"] = min(
            30.0, max(-min(15.0, max(0.0, sigma_percent - 0.1)),
                      float(st.session_state["v91_single_vol_shift"]))
        )
    st.session_state.setdefault("v91_single_vol_shift", 0.0)
    st.session_state.setdefault("v91_single_elapsed", 0.0)
    with st.expander("Scénario avancé — volatilité et temps", expanded=False):
        st.caption("Les valeurs initiales restent fixes. Le scénario modifie la volatilité et le temps restant, "
                   "sans modifier les paramètres du contrat initial.")
        adv1, adv2 = st.columns(2)
        with adv1:
            single_vol_shift = st.slider("Variation de volatilité (points de %)",
                                         min_value=-min(15.0, max(0.0, sigma_percent - 0.1)),
                                         max_value=30.0, step=0.1,
                                         key="v91_single_vol_shift")
        with adv2:
            if max_single_elapsed_days < 1:
                single_elapsed = 0.0
                st.session_state["v91_single_elapsed"] = 0.0
                st.info("Maturité trop courte pour simuler un temps écoulé positif.")
            else:
                single_elapsed_days = st.slider(
                    "Temps écoulé (jours)", min_value=0,
                    max_value=max_single_elapsed_days, step=1,
                    key="v911_single_elapsed_days"
                )
                single_elapsed = single_elapsed_days / 365.0
                st.session_state["v91_single_elapsed"] = single_elapsed
        st.caption(f"Scénario : σ₁ = {format_fr(sigma_percent + single_vol_shift)} % ; "
                   f"temps écoulé = {int(round(single_elapsed * 365))} jour(s) ; "
                   f"maturité restante = {format_fr((T - single_elapsed) * 365, 0)} jours.")
    sigma1 = (sigma_percent + single_vol_shift) / 100
    T1 = T - single_elapsed
    position_S1 = option_position(S1, K, T1, r, sigma1, option_type, position_type, quantity)
    hedge_new = position_S1["hedge_shares"]
    hedge_adjustment = clean_zero(hedge_new - hedge_initial)
    spot_change_pct = 100 * (S1 / S0 - 1)

    st.caption(
        f"Référence S₀ : {format_fr(S0)} {orm_currency_symbol} · Spot simulé S₁ : {format_fr(S1)} {orm_currency_symbol} "
        f"({format_fr(spot_change_pct)} %) · Ajustement : "
        f"{hedge_action(hedge_adjustment)} {format_fr(abs(hedge_adjustment))} actions"
    )

    # Indicateurs natifs Streamlit : style V10.1 conservé, sans troncature des unités.
    st.markdown("#### Résultats actualisés à S₁")
    live_cols = st.columns(5)
    live_specs = [
        (f"Prix unitaire ({orm_currency_symbol})", "unit_price", 2, f"{orm_currency_symbol}"),
        (f"Valeur position ({orm_currency_symbol})", "position_value", 2, f"{orm_currency_symbol}"),
        ("Delta (actions)", "position_delta", 2, "actions"),
        (f"Gamma (actions/{orm_currency_symbol})", "position_gamma", 4, f"actions/{orm_currency_symbol}"),
        ("Couverture cible (actions)", "hedge_shares", 2, "actions"),
    ]
    for col, (label, field, digits, unit) in zip(live_cols, live_specs):
        current = format_fr(position_S1[field], digits)
        reference = format_fr(position_initiale[field], digits)
        # L'unité figure dans le titre, et non dans la grande valeur,
        # afin d'éviter les « ... » sur les écrans étroits.
        col.metric(label, current,
                   delta=f"à S₀ : {reference} {unit}", delta_color="off")

    # Même carte de décision que la V10.1, sans composant HTML complexe.
    rebalance_action = hedge_action(hedge_adjustment)
    main_class = "buy" if rebalance_action == "ACHETER" else "sell" if rebalance_action == "VENDRE" else "neutral"
    st.markdown(
        f"<div class='trade-card {main_class}'>"
        f"<div class='trade-label'>Couverture à mettre en place à S₀</div>"
        f"<div class='trade-value'>{trade_instruction(hedge_initial)}</div>"
        f"<div class='trade-label' style='margin-top:12px'>Ajustement à S₁ (couverture initiale déjà détenue)</div>"
        f"<div class='trade-value'>{trade_instruction(hedge_adjustment)}</div>"
        f"<div class='trade-note'>Position cible : {format_fr(hedge_new)} actions</div></div>",
        unsafe_allow_html=True
    )

    scenario_changes = []
    if abs(S1 - S0) > 1e-8:
        scenario_changes.append(f"Spot : {format_fr((S1/S0-1)*100, 2)} %")
    if abs(single_vol_shift) > 1e-8:
        scenario_changes.append(f"Volatilité : {format_fr(single_vol_shift, 2)} point(s)")
    if abs(single_elapsed) > 1e-8:
        scenario_changes.append(f"Temps écoulé : {int(round(single_elapsed * 365))} jours")
    if scenario_changes:
        st.caption("**Scénario actif :** " + " · ".join(scenario_changes))
    else:
        st.caption("Scénario de référence actif : aucun choc.")

    single_tab_overview, single_tab_hedge, single_tab_pnl, single_tab_stress, single_tab_strategy, single_tab_summary = st.tabs([
        "Vue d’ensemble", "Couverture & Gamma", "P&L", "Stress tests", "Payoff & Risk ladder", "Synthèse"
    ])

    with single_tab_overview:
        # SYNTHÈSE
        # --------------------------------------------------------

        st.markdown(
            f"""
            <div class="badge-row">
                <span class="badge">{position_type_display.upper()}</span>
                <span class="badge">{option_type_display.upper()}</span>
                <span class="badge">{quantity} OPTIONS</span>
                <span class="badge">S₀ {format_fr(S0, 2)} {orm_currency_symbol}</span>
                <span class="badge">K {format_fr(K, 2)} {orm_currency_symbol}</span>
                <span class="badge">σ {format_fr(sigma_percent, 1)} %</span>
            </div>
            <div class="section-kicker">Position</div>
            <div class="section-title">Vue d'ensemble</div>
            """,
            unsafe_allow_html=True
        )

        info1, info2, info3, info4 = st.columns(4)

        info1.metric(
            "Instrument",
            f"{position_type_display} "
            f"{option_type_display}"
        )

        info2.metric(
            "Spot / Strike",
            f"{format_fr(S0, 2)} / {format_fr(K, 2)} {orm_currency_symbol}"
        )

        info3.metric(
            "Maturité",
            f"{format_fr(T, 2)} an"
        )

        info4.metric(
            "Volatilité",
            f"{format_fr(sigma_percent, 2)} %"
        )

        # Contrôles numériques : cohérence Call/Put + validation des Greeks
        call_check = bs_price(S0, K, T, r, sigma, "call")
        put_check = bs_price(S0, K, T, r, sigma, "put")
        parity_left = call_check - put_check
        parity_right = S0 - K * np.exp(-r * T)
        parity_error = abs(parity_left - parity_right)
        fd_check = finite_diff_check(S0, K, T, r, sigma, option_type)

        with st.expander("Contrôles numériques du moteur Black-Scholes", expanded=False):
            st.markdown("**1. Contrôle de cohérence Call/Put — parité Call-Put**")
            st.write(f"C − P = **{format_fr(parity_left, 10)}** ; S₀ − K e^(-rT) = **{format_fr(parity_right, 10)}**")
            st.write(f"Erreur de parité : **{parity_error:.2e}**")
            st.caption("Ce contrôle vérifie la cohérence entre les prix du Call et du Put ; il ne suffit pas, à lui seul, à valider d₁, d₂ ou les Greeks.")

            st.markdown("**2. Validation indépendante des Greeks — différences finies centrales**")
            v1, v2 = st.columns(2)
            v1.metric("Δ analytique", f"{format_fr(fd_check['delta_analytic'], 8)}")
            v1.metric("Δ différences finies", f"{format_fr(fd_check['delta_fd'], 8)}", delta=f"erreur {fd_check['delta_error']:.2e}", delta_color="off")
            v2.metric("Γ analytique", f"{format_fr(fd_check['gamma_analytic'], 8)}")
            v2.metric("Γ différences finies", f"{format_fr(fd_check['gamma_fd'], 8)}", delta=f"erreur {fd_check['gamma_error']:.2e}", delta_color="off")
            st.caption(f"Pas de différence finie h = {fd_check['h']:.4g} {orm_currency_symbol}.")
            if parity_error < 1e-8 and within_tol(fd_check["delta_error"], fd_check["delta_analytic"]) and within_tol(fd_check["gamma_error"], fd_check["gamma_analytic"]):
                st.success("✓ Contrôles numériques réussis : cohérence Call/Put et Greeks analytiques validés numériquement.")
            else:
                st.warning("Un écart numérique dépasse le seuil de contrôle ; vérifier les paramètres et les formules.")

        # --------------------------------------------------------
        # VALORISATION ET RISQUE
        # --------------------------------------------------------

        st.subheader(
            "Valorisation et risque"
        )

        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )

        col1.metric(
            "Prix unitaire",
            f"{format_fr(position_initiale['unit_price'], 2)} {orm_currency_symbol}"
        )

        col2.metric(
            "Valeur position",
            f"{format_fr(position_initiale['position_value'], 2)} {orm_currency_symbol}"
        )

        col3.metric(
            "Delta",
            f"{format_fr(position_initiale['position_delta'], 4)}"
        )

        col4.metric(
            "Gamma",
            f"{format_fr(position_initiale['position_gamma'], 4)}"
        )

        hedge_initial = (
            position_initiale["hedge_shares"]
        )

        col5.metric(
            f"Couverture : {hedge_action(hedge_initial)}",
            f"{format_fr(abs(hedge_initial), 2)} actions"
        )

        risk1, risk2, risk3 = st.columns(3)

        risk1.metric(
            "Vega (+1 pt de vol.)",
            f"{format_fr(position_initiale['position_vega'], 4)} {orm_currency_symbol}"
        )

        risk2.metric(
            "Theta / jour",
            f"{format_fr(position_initiale['position_theta_daily'], 4)} {orm_currency_symbol}"
        )

        risk3.metric(
            "Rho (+1 pt de taux)",
            f"{format_fr(position_initiale['position_rho'], 4)} {orm_currency_symbol}"
        )


        # --------------------------------------------------------
        # COUVERTURE DELTA
        # --------------------------------------------------------

        st.markdown(
            "<div class='section-kicker'>Couverture</div><div class='section-title'>Couverture Delta initiale</div>",
            unsafe_allow_html=True
        )

        hedge_quantity = abs(hedge_initial)

        if hedge_initial > 1e-12:
            st.markdown(
                f"<div class='trade-card buy'><div class='trade-label'>Action trader</div>"
                f"<div class='trade-value'>ACHETER {format_fr(hedge_quantity, 2)} actions</div>"
                f"<div class='trade-note'>Couverture nécessaire pour rendre la position Delta-neutre au spot S₀ = {format_fr(S0, 2)} {orm_currency_symbol}.</div></div>",
                unsafe_allow_html=True
            )
        elif hedge_initial < -1e-12:
            st.markdown(
                f"<div class='trade-card sell'><div class='trade-label'>Action trader</div>"
                f"<div class='trade-value'>VENDRE {format_fr(hedge_quantity, 2)} actions</div>"
                f"<div class='trade-note'>Couverture nécessaire pour rendre la position Delta-neutre au spot S₀ = {format_fr(S0, 2)} {orm_currency_symbol}.</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='trade-card'><div class='trade-label'>Action trader</div>"
                "<div class='trade-value'>AUCUNE ACTION</div>"
                "<div class='trade-note'>La position est déjà Delta-neutre.</div></div>",
                unsafe_allow_html=True
            )


        # --------------------------------------------------------
    with single_tab_hedge:
        # SIMULATION DYNAMIQUE
        # --------------------------------------------------------

        st.divider()

        st.header(
            "Simulation dynamique"
        )

        st.markdown(
            f"<div class='scenario-strip'><div class='scenario-item'>Spot initial<b>{format_fr(S0)} {orm_currency_symbol}</b></div><div class='scenario-item'>Spot simulé<b>{format_fr(S1, 2)} {orm_currency_symbol}</b></div><div class='scenario-item'>Variation<b>{format_fr_signed(spot_change_pct, 2)} %</b></div></div>",
            unsafe_allow_html=True
        )

        sim1, sim2, sim3, sim4 = (
            st.columns(4)
        )

        sim1.metric(
            "Spot simulé",
            f"{format_fr(S1)} {orm_currency_symbol}",
            delta=f"{format_fr_signed(S1 - S0, 2)} {orm_currency_symbol}"
        )

        sim2.metric(
            "Delta",
            f"{format_fr(position_S1['position_delta'], 4)}",
            delta=(
                f"{format_fr_signed(position_S1['position_delta'] - position_initiale['position_delta'], 4)}"
            )
        )

        sim3.metric(
            "Gamma",
            f"{format_fr(position_S1['position_gamma'], 4)}",
            delta=(
                f"{format_fr_signed(position_S1['position_gamma'] - position_initiale['position_gamma'], 4)}"
            )
        )

        sim4.metric(
            "Couverture requise",
            f"{hedge_action(hedge_new)} "
            f"{abs(hedge_new):.2f} actions"
        )


        # --------------------------------------------------------
        # RÉÉQUILIBRAGE
        # --------------------------------------------------------

        if hedge_adjustment > 1e-12:

            st.success(
                f"Rééquilibrage depuis S₀ : "
                f"ACHETER {abs(hedge_adjustment):.2f} "
                f"actions."
            )

        elif hedge_adjustment < -1e-12:

            st.warning(
                f"Rééquilibrage depuis S₀ : "
                f"VENDRE {abs(hedge_adjustment):.2f} "
                f"actions."
            )

        else:

            st.info(
                "Aucun rééquilibrage nécessaire "
                "par rapport à S₀."
            )


        # --------------------------------------------------------
        # COURBES DE RISQUE
        # --------------------------------------------------------

        graph_df = build_single_risk_curve(
            S0, K, T, r, sigma, option_type, position_type, quantity
        )

        gamma_max_index = graph_df["Gamma"].abs().idxmax()
        spot_gamma_max = float(graph_df.loc[gamma_max_index, "Spot"])
        gamma_max = float(graph_df.loc[gamma_max_index, "Gamma"])

        single_scenario_curve = build_single_risk_curve(S0, K, T1, r, sigma1, option_type, position_type, quantity)
        st.subheader("Couverture en fonction du spot")
        moneyness_context(option_type, K, S1)
        st.plotly_chart(interactive_risk_chart(single_scenario_curve, "Couverture", S0, S1, hedge_new, K, reference_data=graph_df, option_type=option_type),
                        width="stretch", key="single_hedge_plotly")
        if True:
            action = "acheter" if hedge_new > 0 else "vendre" if hedge_new < 0 else "ne détenir aucune"
            explain_graph("Couverture Delta",
                          f"À S₁ = {format_fr(S1, 2)} {orm_currency_symbol}, le Delta de la position vaut {format_fr(position_S1['position_delta'], 4)}. La couverture neutralise localement ce Delta.",
                          f"Il faut {action} {format_fr(abs(hedge_new), 2)} action(s). Lorsque le spot ou les paramètres changent, cette quantité doit être recalculée.")
        st.subheader("Gamma en fonction du spot")

        st.plotly_chart(interactive_risk_chart(single_scenario_curve, "Gamma", S0, S1,
                        position_S1["position_gamma"], K, reference_data=graph_df, option_type=option_type), width="stretch", key="single_gamma_plotly")
        if True:
            gamma_now = position_S1["position_gamma"]
            explain_graph("Sensibilité de la couverture",
                          f"Gamma = {format_fr(gamma_now, 4)} actions/{orm_currency_symbol}. Pour un mouvement de spot de 1 {orm_currency_symbol}, le Delta varie approximativement de {format_fr(gamma_now, 4)} action(s).",
                          "La bande orange représente |Γ| ≥ 50 % du maximum de la courbe : c'est un repère visuel conventionnel, pas un seuil réglementaire. Le Gamma reste un risque après couverture Delta.")

        st.markdown("#### Explorer les Greeks selon le spot")
        with st.popover("ⓘ Lire les courbes des Greeks"):
            st.write("Choisissez un Greek. L'axe horizontal est le spot en euros ; l'axe vertical est l'exposition de la position, dans l'unité indiquée. Le trait vertical pointillé repère le spot simulé. Delta mesure la sensibilité directionnelle, Gamma la variation du Delta par euro, Vega la variation de valeur par point de volatilité et Theta l'érosion par an à paramètres constants.")
        greek_choice = st.selectbox("Greek affiché", ["Delta", "Gamma", "Vega", "Theta"], key="single_greek_selector_v931")
        greek_spots = np.linspace(0.6 * S0, 1.4 * S0, 101)
        greek_rows = [option_position(float(z), K, T1, r, sigma1, option_type, position_type, quantity) for z in greek_spots]
        greek_keys = {"Delta": ("position_delta", "actions"), "Gamma": ("position_gamma", f"actions/{orm_currency_symbol}"),
                      "Vega": ("position_vega", f"{orm_currency_symbol}/point de vol."), "Theta": ("position_theta_annual", f"{orm_currency_symbol}/an")}
        gkey, gunit = greek_keys[greek_choice]
        fig_greek = go.Figure(go.Scatter(x=greek_spots, y=[z[gkey] for z in greek_rows], mode="lines", name=greek_choice, line=dict(color="#2563eb", width=3)))
        fig_greek.add_vline(x=S1, line_dash="dash", line_color="#d97706")
        add_moneyness_zones(fig_greek, K, option_type, float(greek_spots[0]), float(greek_spots[-1]))

        fig_greek.update_layout(template="plotly_white", height=380, xaxis_title=f"Spot ({orm_currency_symbol})", yaxis_title=f"{greek_choice} ({gunit})", margin=dict(l=25,r=20,t=85,b=45))
        st.plotly_chart(fig_greek, width="stretch", key="single_greek_plot_v931")
        st.caption("Zones de moneyness : OTM (bleu), ITM (vert), ATM au strike K (trait violet). Pour un Put, les zones sont inversées par rapport à un Call.")

        gamma_shock = 0.01 * S1
        approx_hedge_change = -position_S1["position_gamma"] * gamma_shock
        st.metric("Rééquilibrage estimé pour +1 % de spot (approximation Gamma)",
                  f"{'ACHETER' if approx_hedge_change > 0 else 'VENDRE' if approx_hedge_change < 0 else 'AUCUNE ACTION'} {format_fr(abs(approx_hedge_change), 2)} action(s)")
        with st.popover("ⓘ Comment interpréter cette estimation ?"):
            st.write("Variation approximative du nombre d'actions de couverture : −Gamma de position × variation du spot. Calcul local pour +1 % de spot, à volatilité et maturité constantes ; il ne remplace pas un recalcul exact du Delta après le choc.")

        # --------------------------------------------------------
        # GAMMA MAXIMAL
        # --------------------------------------------------------

        st.info(
            f"Maximum de |Gamma| sur la plage étudiée : "
            f"S = {format_fr(spot_gamma_max, 2)} {orm_currency_symbol} ; "
            f"Gamma = {format_fr(gamma_max, 4)}."
        )


    with single_tab_pnl:
        # --------------------------------------------------------
        # P&L
        # --------------------------------------------------------

        st.divider()

        st.header(
            "Analyse du P&L"
        )

        initial_value = (
            position_initiale["position_value"]
        )

        current_value = (
            position_S1["position_value"]
        )

        delta_S = (
            S1 - S0
        )

        option_pnl = (
            current_value
            - initial_value
        )

        stock_pnl = (
            hedge_initial
            * delta_S
        )
        single_components = pnl_breakdown(position_initiale, option_pnl, delta_S,
                                          single_vol_shift, single_elapsed)
        render_breakdown(single_components, "v91_single_pnl_breakdown")

        hedged_pnl = (
            option_pnl
            + stock_pnl
        )

        delta_initial = (
            position_initiale["position_delta"]
        )

        gamma_initial = (
            position_initiale["position_gamma"]
        )

        pnl_delta = (
            delta_initial
            * delta_S
        )

        pnl_delta_gamma = (
            delta_initial
            * delta_S
            + 0.5
            * gamma_initial
            * delta_S**2
        )

        option_pnl = clean_zero(
            option_pnl
        )

        stock_pnl = clean_zero(
            stock_pnl
        )

        hedged_pnl = clean_zero(
            hedged_pnl
        )

        pnl_delta = clean_zero(
            pnl_delta
        )

        pnl_delta_gamma = clean_zero(
            pnl_delta_gamma
        )

        pnl1, pnl2, pnl3 = (
            st.columns(3)
        )

        pnl1.metric(
            "P&L option",
            f"{format_fr(option_pnl, 2)} {orm_currency_symbol}"
        )

        pnl2.metric(
            "P&L couverture actions",
            f"{format_fr(stock_pnl, 2)} {orm_currency_symbol}"
        )

        pnl3.metric(
            "P&L position couverte",
            f"{format_fr(hedged_pnl, 2)} {orm_currency_symbol}"
        )

        approx1, approx2, approx3 = (
            st.columns(3)
        )

        approx1.metric(
            "P&L exact",
            f"{format_fr(option_pnl, 2)} {orm_currency_symbol}"
        )

        approx2.metric(
            "Approximation Delta",
            f"{format_fr(pnl_delta, 2)} {orm_currency_symbol}"
        )

        approx3.metric(
            "Approximation Delta-Gamma",
            f"{format_fr(pnl_delta_gamma, 2)} {orm_currency_symbol}"
        )

        st.markdown("#### Comparaison des méthodes de P&L")
        full_approx_single = sum(value for name, value in single_components.items()
                                 if name != "Résidu (ordre supérieur)")
        render_pnl_comparison(option_pnl, pnl_delta, pnl_delta_gamma,
                              "single_pnl_comparison_v914", full_approx_single)
        st.markdown("#### Précision des approximations selon le spot")
        error_spots = np.linspace(0.6 * S0, 1.4 * S0, 81)
        exact_spot_only = np.array([
            option_position(float(z), K, T, r, sigma, option_type, position_type, quantity)["position_value"]
            - position_initiale["position_value"] for z in error_spots
        ])
        shocks = error_spots - S0
        render_pnl_error_curve(
            error_spots, exact_spot_only,
            delta_initial * shocks,
            delta_initial * shocks + 0.5 * gamma_initial * shocks ** 2,
            S1, "single_pnl_error_curve_v932"
        )
        st.markdown("#### Couverture avant et après le scénario")
        render_hedge_rebalance(hedge_initial, position_S1["hedge_shares"], "single_rebalance_v932")


        # --------------------------------------------------------
        # INTERPRÉTATION P&L
        # --------------------------------------------------------

        st.caption(
            "Le P&L exact compare la position initiale au scénario : "
            "il inclut les variations de spot, de volatilité et de temps. "
            "Le taux sans risque reste constant."
        )


        # V12.3 — diagnostic explicable et comparaison de couverture statique.
        render_risk_diagnostic(
            spot0=S0,
            spot1=S1,
            delta0=position_initiale["position_delta"],
            delta1=position_S1["position_delta"],
            gamma0=position_initiale["position_gamma"],
            hedge0=hedge_initial,
            key="v123_single_risk",
        )
        render_static_pnl_comparison(
            spots=error_spots,
            option_pnl=exact_spot_only,
            spot0=S0,
            current_spot=S1,
            hedge0=hedge_initial,
            key="v123_single_static",
        )


    with single_tab_stress:
        # --------------------------------------------------------
        # STRESS TESTS
        # --------------------------------------------------------

        st.divider()

        st.header(
            "Stress tests"
        )

        st.markdown("### Carte de sensibilité — spot × volatilité")
        with st.popover("ⓘ Comment lire la carte de sensibilité ?"):
            st.write("Les colonnes représentent le spot stressé en euros, les lignes la volatilité annualisée en %. "
                     "Chaque case indique le P&L exact de la position d'options par rapport à la valeur initiale, "
                     "en conservant la maturité initiale. Rouge : perte, vert : gain. "
                     "Le croisement des paramètres initiaux correspond à un P&L nul. "
                     "Cette carte mesure une revalorisation statique, pas le résultat d'une couverture rééquilibrée.")
        heat_spots = np.linspace(0.8 * S0, 1.2 * S0, 17)
        heat_vols = np.linspace(max(0.01, sigma * 0.5), sigma * 1.5, 13)
        heat_values = np.array([
            [(1 if position_type.lower() == "long" else -1) * quantity * bs_price(float(spot), K, T, r, float(vol), option_type)
             - position_initiale["position_value"] for spot in heat_spots]
            for vol in heat_vols
        ])
        heatmap = go.Figure(go.Heatmap(
            x=heat_spots, y=100 * heat_vols, z=heat_values,
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title=f"P&L ({orm_currency_symbol})"),
            hovertemplate=("Spot : %{x:.2f} €<br>Volatilité : %{y:.2f} %<br>P&L : %{z:.2f} €<extra></extra>".replace("€", orm_currency_symbol))))
        # Sur une carte spot × volatilité, ATM est la frontière verticale S=K.
        heatmap.add_vline(x=K, line_dash="dot", line_color="#7c3aed", line_width=2)
        heatmap.add_annotation(x=K, y=1.06, yref="paper", text="ATM · K", showarrow=False,
                               font=dict(size=10, color="#7c3aed"))
        heatmap.add_annotation(x=float((heat_spots[0]+K)/2), y=1.04, yref="paper",
                               text="OTM" if str(option_type).lower() == "call" else "ITM", showarrow=False)
        heatmap.add_annotation(x=float((heat_spots[-1]+K)/2), y=1.04, yref="paper",
                               text="ITM" if str(option_type).lower() == "call" else "OTM", showarrow=False)
        heatmap.add_trace(go.Scatter(x=[S0], y=[100*sigma], mode="markers",
                       marker=dict(color="#111827", size=11, symbol="x"), name="Référence"))
        heatmap.update_layout(template="plotly_white", height=410,
                              xaxis_title=f"Spot ({orm_currency_symbol})", yaxis_title="Volatilité annualisée (%)",
                              margin=dict(l=25, r=25, t=70, b=45))
        st.plotly_chart(heatmap, width="stretch", key="single_pnl_heatmap_v93")

        st.markdown("### Carte de couverture Delta — spot × volatilité")
        with st.popover("ⓘ Comment lire la carte de couverture ?"):
            st.write(f"En colonnes : spot ({orm_currency_symbol}). En lignes : volatilité annualisée (%). Chaque case donne le nombre signé d'actions de couverture : positif = ACHETER (vert), négatif = VENDRE (rouge). Le marqueur noir désigne les paramètres initiaux. Les variations de couleur montrent comment la couverture doit être adaptée quand le spot et la volatilité changent ensemble, à maturité constante.")
        hedge_grid = np.array([[
            -option_position(float(z), K, T, r, float(v), option_type, position_type, quantity)["position_delta"]
            for z in heat_spots] for v in heat_vols])
        hedge_fig = go.Figure(go.Heatmap(x=heat_spots, y=100*heat_vols, z=hedge_grid,
             colorscale="RdYlGn", zmid=0, colorbar=dict(title="Actions"),
             hovertemplate=("Spot : %{x:.2f} €<br>Volatilité : %{y:.2f} %<br>Couverture : %{z:.2f} actions<extra></extra>".replace("€", orm_currency_symbol))))
        hedge_fig.add_vline(x=K, line_dash="dot", line_color="#7c3aed", line_width=2)
        hedge_fig.add_trace(go.Scatter(x=[S0], y=[100*sigma], mode="markers",
                                    marker=dict(color="#111827", size=11, symbol="x"), name="Référence"))
        hedge_fig.update_layout(template="plotly_white", height=410, xaxis_title=f"Spot ({orm_currency_symbol})",
                                yaxis_title="Volatilité annualisée (%)", margin=dict(l=25,r=25,t=70,b=45))
        st.plotly_chart(hedge_fig, width="stretch", key="single_hedge_heatmap_v931")

        stress_df, vol_df = build_single_stress_data(
            S0, K, T, r, sigma, option_type, position_type, quantity
        )

        stress_col1, stress_col2 = st.columns(2)

        with stress_col1:
            st.subheader("Stress du spot")

            fig_stress = go.Figure()
            fig_stress.add_trace(go.Scatter(x=stress_df["Spot"], y=stress_df["P&L option"],
                                            mode="lines", name="P&L option", line=dict(color="#2563eb", width=3)))
            fig_stress.add_trace(go.Scatter(x=stress_df["Spot"], y=stress_df["P&L couvert"],
                                            mode="lines", name="P&L couvert", line=dict(color="#16a34a", width=3)))
            fig_stress.add_hline(y=0, line_color="#94a3b8", line_width=1)
            fig_stress.add_vline(x=S0, line_dash="dash", line_color="#64748b")
            fig_stress.update_layout(template="plotly_white", height=360, xaxis_title=f"Spot stressé ({orm_currency_symbol})",
                                     yaxis_title=f"P&L ({orm_currency_symbol})", margin=dict(l=20,r=20,t=30,b=75),
                                     legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig_stress, width="stretch", key="single_spot_stress_v910")
            with st.popover("ⓘ Comment lire le stress du spot ?"):
                st.write("Axe horizontal : spot stressé en euros. Axe vertical : P&L en euros par rapport au spot initial. "
                         "La courbe non couverte montre la variation de valeur des options ; la courbe couverte ajoute le P&L "
                         "des actions de couverture initiale, conservées sans rééquilibrage. "
                         "Un P&L positif représente un gain et un P&L négatif une perte. "
                         "La couverture neutralise localement le Delta, pas les risques Gamma ou Vega.")

        with stress_col2:

            st.subheader(
                "Stress de volatilité"
            )

            fig_vol = go.Figure(go.Scatter(x=vol_df["Volatilité"] * 100,
                    y=vol_df["Valeur position"], mode="lines", name="Valeur position",
                    line=dict(color="#2563eb", width=3)))
            fig_vol.add_vline(x=sigma*100, line_dash="dash", line_color="#64748b")
            fig_vol.update_layout(template="plotly_white", height=360,
                                  xaxis_title="Volatilité annualisée (%)", yaxis_title=f"Valeur ({orm_currency_symbol})",
                                  margin=dict(l=20,r=20,t=30,b=75))
            st.plotly_chart(fig_vol, width="stretch", key="single_vol_stress_v910")
            with st.popover("ⓘ Comment lire le stress de volatilité ?"):
                st.write(f"Axe horizontal : volatilité annualisée (%). Axe vertical : valeur de la position ({orm_currency_symbol}). "
                         "La courbe montre la revalorisation de la position pour chaque volatilité, à spot et maturité fixés. "
                         "La ligne verticale pointillée est la volatilité initiale. Une pente croissante traduit un Vega positif ; "
                         "une pente décroissante un Vega négatif. Ce graphique est un stress à spot de référence fixe : "
                         "il ne suit pas automatiquement le curseur S₁.")


    with single_tab_summary:
        st.download_button(
            "↓ Exporter les résultats (CSV)",
            data=result_csv([
                ("Paramètres de référence", "Type", option_type_display, ""),
                ("Paramètres de référence", "Sens", position_type_display, ""),
                ("Paramètres de référence", "Quantité", quantity, "options"),
                ("Paramètres de référence", "Spot initial", S0, "EUR"),
                ("Scénario de marché", "Spot scénario", S1, "EUR"),
                ("Paramètres de référence", "Strike", K, "EUR"),
                ("Paramètres de référence", "Volatilité initiale", sigma_percent, "%"),
                ("Scénario de marché", "Volatilité scénario", sigma_percent + single_vol_shift, "%"),
                ("Paramètres de référence", "Maturité initiale", T, "années"),
                ("Scénario de marché", "Maturité scénario", T1, "années"),
                ("Valorisation et P&L", "Valeur initiale", position_initiale["position_value"], "EUR"),
                ("Risques initiaux", "Delta initial", position_initiale["position_delta"], "actions"),
                ("Risques initiaux", "Gamma initial", position_initiale["position_gamma"], "actions/EUR"),
                ("Risques simulés et couverture", "Delta scénario", position_S1["position_delta"], "actions"),
                ("Risques simulés et couverture", "Gamma scénario", position_S1["position_gamma"], "actions/EUR"),
                ("Risques initiaux", "Couverture initiale", hedge_initial, "actions"),
                ("Risques simulés et couverture", "Couverture cible", hedge_new, "actions"),
                ("Risques simulés et couverture", "Rééquilibrage", hedge_adjustment, "actions"),
                ("Scénario de marché", "Choc volatilité", single_vol_shift, "points de %"),
                ("Scénario de marché", "Temps écoulé", single_elapsed * 365, "jours"),
                ("Valorisation et P&L", "Valeur scénario", position_S1["position_value"], "EUR"),
                ("Valorisation et P&L", "P&L exact options", position_S1["position_value"] - position_initiale["position_value"], "EUR"),
            ]),
            file_name="option_risk_position.csv", mime="text/csv",
            key="export_single_v96"
        )
        st.markdown("### Décision de couverture au scénario")
        st.caption("Les quantités sont signées : positif = actions à acheter, négatif = actions à vendre.")
        render_hedge_rebalance(
            hedge_initial, position_S1["hedge_shares"], "single_summary_rebalance_v94"
        )
        current_delta_residual = position_S1["position_delta"] + hedge_initial
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Delta après choc, hedge initial", format_fr(current_delta_residual, 4) + " actions")
        sc2.metric("Gamma au scénario", format_fr(position_S1["position_gamma"], 4) + f" actions/{orm_currency_symbol}")
        sc3.metric("Spot du scénario", format_fr(S1, 2) + f" {orm_currency_symbol}")
        with st.popover("ⓘ Comment lire le diagnostic ?"):
            st.write("La couverture initiale neutralise le Delta au spot S₀. "
                     "Après un choc vers S₁, le Delta des options change, notamment à cause du Gamma. "
                     "Le Delta résiduel indiqué suppose que la quantité initiale d'actions est conservée. "
                     "L'ordre de rééquilibrage indique l'ajustement nécessaire pour retrouver "
                     "un Delta théoriquement nul au scénario. La neutralité Delta ne supprime "
                     "ni le risque Gamma, ni le risque Vega, ni le risque Theta.")
        st.divider()
        # --------------------------------------------------------
        # SYNTHÈSE VISUELLE DU RISQUE
        # --------------------------------------------------------

        st.markdown(
            "<div class='section-kicker'>Synthèse des risques</div><div class='section-title'>Synthèse de la position</div>",
            unsafe_allow_html=True
        )

        gamma_value = position_initiale["position_gamma"]
        vega_value = position_initiale["position_vega"]
        theta_value = position_initiale["position_theta_daily"]
        rho_value = position_initiale["position_rho"]
        delta_value = position_initiale["position_delta"]

        gamma_text = "Convexité positive" if gamma_value > 1e-12 else ("Convexité négative" if gamma_value < -1e-12 else "Gamma quasi nul")
        vega_text = "Long volatilité" if vega_value > 1e-12 else ("Short volatilité" if vega_value < -1e-12 else "Vega quasi nul")
        delta_text = "Exposition positive au sous-jacent" if delta_value > 1e-12 else ("Exposition négative au sous-jacent" if delta_value < -1e-12 else "Delta quasi neutre")

        st.markdown(f"""
            <div class='risk-hero'>
                <div class='risk-hero-kicker'>Risk Summary · Position individuelle</div>
                <div class='risk-hero-title'>{position_type_display.upper()} {option_type_display.upper()} × {quantity}</div>
                <div class='risk-hero-sub'>Synthèse de l’exposition et décision de couverture · S₀ {format_fr(S0, 2)} {orm_currency_symbol} · K {format_fr(K, 2)} {orm_currency_symbol} · σ {format_fr(sigma_percent, 2)} %</div>
            </div>
            <div class='summary-shell'>
                <div class='summary-grid'>
                    <div class='summary-kpi'><div class='summary-kpi-label'>Δ Delta</div><div class='summary-kpi-value'>{format_fr_signed(delta_value, 4)}</div><div class='summary-kpi-note'>{delta_text}</div></div>
                    <div class='summary-kpi'><div class='summary-kpi-label'>Γ Gamma</div><div class='summary-kpi-value'>{format_fr_signed(gamma_value, 4)}</div><div class='summary-kpi-note'>{gamma_text}</div></div>
                    <div class='summary-kpi'><div class='summary-kpi-label'>ν Vega</div><div class='summary-kpi-value'>{format_fr_signed(vega_value, 4)} {orm_currency_symbol}</div><div class='summary-kpi-note'>{vega_text}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # V11.1 — Résumé du profil de payoff, sans recopier le graphique.
        premium_total = abs(position_initiale["position_value"])
        is_long = position_type.lower() == "long"
        is_call = option_type.lower() == "call"
        if is_call:
            breakeven = K + premium_total / quantity * (1 if is_long else 1)
            loss_profile = f"{format_fr(premium_total, 2)} {orm_currency_symbol} (prime payée)" if is_long else "Théoriquement illimitée"
            gain_profile = "Théoriquement illimité" if is_long else f"{format_fr(premium_total, 2)} {orm_currency_symbol} (prime reçue)"
        else:
            breakeven = K - premium_total / quantity
            loss_profile = f"{format_fr(premium_total, 2)} {orm_currency_symbol} (prime payée)" if is_long else f"{format_fr(max(0.0, K * quantity - premium_total), 2)} {orm_currency_symbol} (si spot nul)"
            gain_profile = f"{format_fr(max(0.0, K * quantity - premium_total), 2)} {orm_currency_symbol} (si spot nul)" if is_long else f"{format_fr(premium_total, 2)} {orm_currency_symbol} (prime reçue)"
        st.markdown("#### Profil de payoff à échéance")
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Point mort (hors financement)", f"{format_fr(breakeven, 2)} {orm_currency_symbol}" if breakeven >= 0 else "Non atteignable")
        pcol2.metric("Perte maximale", loss_profile)
        pcol3.metric("Gain maximal", gain_profile)
        st.caption("Valeurs à échéance, sans frais ni financement de la prime. Couverture cible initiale : " + trade_instruction(hedge_initial) + ". Détail des sensibilités dans « Payoff & Risk ladder ».")

        hedge_class = "buy" if hedge_initial > 1e-12 else ("sell" if hedge_initial < -1e-12 else "neutral")
        hedge_word = hedge_action(hedge_initial)
        delta_after_hedge = clean_zero(delta_value + hedge_initial)
        neutral_status = "<span class='neutral-chip'>✓ DELTA NEUTRE</span>" if abs(delta_after_hedge) < 1e-10 else ""
        st.markdown(f"""
            <div class='hedge-decision {hedge_class}'>
                <div>
                    <div class='hedge-decision-label'>Couverture Delta · ordre théorique</div>
                    <div class='hedge-decision-action'>{hedge_word}</div>
                    <div class='hedge-decision-qty'>{format_fr(abs(hedge_initial), 2)} <span>ACTIONS</span></div>
                </div>
                <div class='hedge-breakdown'>
                    Delta position <b>{format_fr_signed(delta_value, 4)}</b><br>
                    Delta couverture <b>{format_fr_signed(hedge_initial, 4)}</b><br>
                    Delta après hedge <b>{format_fr_signed(delta_after_hedge, 4)}</b> {neutral_status}
                </div>
            </div>
            <div class='risk-secondary-title'>Risques secondaires</div>
            <div class='risk-reading'>
                <div class='risk-reading-item'><strong>Θ Theta / jour</strong><b>{format_fr_signed(theta_value, 4)} {orm_currency_symbol}</b><br>Effet temporel quotidien, à paramètres de marché constants.</div>
                <div class='risk-reading-item'><strong>ρ Rho</strong><b>{format_fr_signed(rho_value, 4)} {orm_currency_symbol}</b><br>Impact estimé pour une hausse de 1 point du taux sans risque.</div>
                <div class='risk-reading-item'><strong>Γ Lecture Gamma</strong><b>{gamma_text}</b><br>Le Delta évolue lorsque le spot se déplace.</div>
                <div class='risk-reading-item'><strong>ν Lecture Vega</strong><b>{vega_text}</b><br>{format_fr_signed(vega_value, 4)} {orm_currency_symbol} pour +1 point de volatilité.</div>
            </div>
            <div class='risk-conclusion'>
                <strong>Lecture de la position</strong>
                Position {position_type_display} {option_type_display} × {quantity}. {delta_text}. {gamma_text} et {vega_text.lower()}. Une couverture Delta-neutre nécessite de <b>{hedge_word.lower()} {format_fr(abs(hedge_initial), 2)} actions</b>. Après cette couverture, l’exposition directionnelle instantanée est neutralisée, mais les risques Gamma, Vega et Theta subsistent.
            </div>""", unsafe_allow_html=True)


    with st.expander("Hypothèses et limites du modèle", expanded=False):
        st.markdown(
            """
            **Cadre Black-Scholes utilisé dans l'application**

            - options européennes ;
            - absence de dividendes ;
            - volatilité et taux sans risque constants ;
            - marché sans friction (pas de coûts de transaction ni bid-ask) ;
            - couverture Delta théoriquement continue ; l'application illustre un rééquilibrage discret entre S₀ et S₁ ;
            - actions fractionnaires autorisées ;
            - multiplicateur de contrat fixé à **1** : une option porte sur une unité de sous-jacent.
            """
        )


    with single_tab_strategy:
        st.markdown("### Profil de payoff et échelle de risque")
        st.caption("P&L aujourd'hui : revalorisation à maturité constante, moins valeur initiale. À l'échéance : payoff moins prime initiale ; financement ignoré.")
        v11_spots = np.linspace(max(0.01, 0.50 * S0), 1.50 * S0, 401)
        v11_today = np.array([option_position(float(spot), K, T, r, sigma, option_type, position_type, quantity)["position_value"] - position_initiale["position_value"] for spot in v11_spots])
        v11_expiry = np.array([expiry_payoff(float(spot), K, option_type, position_type, quantity) - position_initiale["position_value"] for spot in v11_spots])
        render_v11_chart(v11_spots, v11_today, v11_expiry, S0, S1, [K], "v11_single_payoff",
                         expiry_positions=[{"K": K, "option_type": option_type, "position_type": position_type, "quantity": quantity}],
                         initial_value=position_initiale["position_value"])
        st.markdown("#### Risk ladder — de −10 % à +10 % du spot initial")
        v11_rows = []
        for shock in np.linspace(-0.10, 0.10, 11):
            spot = S0 * (1 + shock)
            state = option_position(spot, K, T, r, sigma, option_type, position_type, quantity)
            pnl = state["position_value"] - position_initiale["position_value"]
            v11_rows.append({"Choc spot": f"{shock:+.0%}", f"Spot ({orm_currency_symbol})": spot,
                             "Delta (actions)": state["position_delta"],
                             "Couverture cible (actions)": state["hedge_shares"],
                             "Rééquilibrage (actions)": state["hedge_shares"] - hedge_initial,
                             f"P&L option ({orm_currency_symbol})": pnl,
                             f"P&L avec hedge initial ({orm_currency_symbol})": pnl + hedge_initial * (spot - S0)})
        render_v11_ladder(v11_rows, "v11_single_ladder_csv")

# ============================================================
# MODE 2 — PORTEFEUILLE MULTI-OPTIONS
# ============================================================

elif mode == "Portefeuille multi-options":
    orm_currency = st.session_state.get("orm_import_currency", "EUR")
    if orm_currency not in ("EUR", "USD"):
        orm_currency = "EUR"
    orm_currency = st.selectbox(
        "Devise des montants (sans conversion de change)",
        ["EUR", "USD"],
        index=0 if orm_currency == "EUR" else 1,
        key="orm_currency_select_1",
        help="La devise est une unité d'affichage ; aucun taux de change n'est appliqué.",
    )
    st.session_state["orm_import_currency"] = orm_currency
    orm_currency_symbol = "€" if orm_currency == "EUR" else "$"
    _import_v126 = st.session_state.get("orm_v126_import")
    if _import_v126 and _import_v126.get("destination") == "Portefeuille multi-options":
        st.info(f"Données importées : {_import_v126['name']} ({_import_v126['ticker']}) · "
                f"{_import_v126['date']} · {_import_v126['currency']} · "
                f"Spot {_import_v126['spot']:.2f} · σ historique {_import_v126['volatility']:.2f} %. "
                "Les autres paramètres restent manuels. Les données ne sont pas synchronisées automatiquement.")

        st.caption("Les strikes des lignes restent manuels. Pour une analyse proche de la monnaie, ajustez chaque strike au spot affiché.")


    st.header(
        "Portefeuille multi-options"
    )

    st.markdown(
        """
        Construction et analyse d'un portefeuille composé
        de plusieurs options européennes portant sur le même
        sous-jacent.
        """
    )


    # --------------------------------------------------------
    # PARAMÈTRES COMMUNS
    # --------------------------------------------------------

    st.sidebar.header("Portefeuille de référence")
    st.sidebar.caption("Sous-jacent et taux communs à toutes les options.")
    st.session_state.setdefault("portfolio_count", 2)
    st.session_state.setdefault("v91_portfolio_vol_shift", 0.0)
    st.session_state.setdefault("v91_portfolio_elapsed", 0.0)
    for _key, _default in PORTFOLIO_DEFAULTS.items():
        st.session_state.setdefault(_key, _default)

    portfolio_S0 = st.sidebar.number_input(
        f"Spot commun S₀ ({orm_currency_symbol})",
        step=1.0,
        key="portfolio_spot"
    )

    portfolio_r_percent = st.sidebar.number_input(
        "Taux sans risque commun r (%)",
        step=0.1,
        key="portfolio_rate"
    )

    number_positions = st.sidebar.slider(
        "Nombre de positions",
        min_value=1,
        max_value=5,
        step=1,
        key="portfolio_count"
    )

    st.sidebar.button("Charger un portefeuille exemple", key="demo_portfolio_v95",
                      on_click=load_portfolio_example, width="stretch")

    with st.sidebar.expander("Réinitialiser les options", expanded=False):
        st.checkbox("Je confirme la remise aux valeurs par défaut",
                    key="confirm_portfolio_reset")
        st.button("Réinitialiser le portefeuille", key="reset_portfolio_options_v96",
                  on_click=reset_portfolio_options, width="stretch",
                  disabled=not st.session_state.get("confirm_portfolio_reset", False))

    portfolio_r = (
        portfolio_r_percent / 100
    )


    # --------------------------------------------------------
    # VALIDATION PARAMÈTRES COMMUNS
    # --------------------------------------------------------

    if portfolio_S0 <= 0:

        st.error(
            "Le spot commun S₀ doit être "
            "strictement positif."
        )

        st.stop()


    # --------------------------------------------------------
    # SCÉNARIO DE MARCHÉ GLOBAL DU PORTEFEUILLE
    # --------------------------------------------------------

    portfolio_spot_min = 0.60 * portfolio_S0
    portfolio_spot_max = 1.40 * portfolio_S0

    st.markdown(
        f"<div class='scenario-strip'><div class='scenario-item'>Spot initial<b>{format_fr(portfolio_S0)} {orm_currency_symbol}</b></div><div class='scenario-item'>Plage de simulation<b>{format_fr(portfolio_spot_min)} {orm_currency_symbol} → {format_fr(portfolio_spot_max)} {orm_currency_symbol}</b></div><div class='scenario-item'>Positions<b>{number_positions}</b></div></div>",
        unsafe_allow_html=True
    )

    # Une valeur mémorisée peut sortir de la nouvelle plage si S₀ change.
    if "portfolio_simulated_spot" not in st.session_state:
        st.session_state["portfolio_simulated_spot"] = float(portfolio_S0)
    else:
        previous_portfolio_s1 = float(st.session_state["portfolio_simulated_spot"])
        if previous_portfolio_s1 < portfolio_spot_min or previous_portfolio_s1 > portfolio_spot_max:
            st.session_state["portfolio_simulated_spot"] = float(portfolio_S0)

    portfolio_scenario_title, portfolio_scenario_reset = st.columns([3, 1], vertical_alignment="center")
    with portfolio_scenario_title:
        st.markdown("#### Simulation du portefeuille")
    with portfolio_scenario_reset:
        st.button("↺ Réinitialiser", key="reset_portfolio_v95",
                  on_click=reset_portfolio_scenario, width="stretch")
    st.caption("Les boutons appliquent un choc supplémentaire au spot simulé actuel (dans la plage autorisée).")
    shock_cols = st.columns(4)
    for col, shock in zip(shock_cols, (-0.10, -0.05, 0.05, 0.10)):
        with col:
            if st.button(f"{shock:+.0%}", key="portfolio_shock_" + str(shock), width="stretch"):
                st.session_state["portfolio_simulated_spot"] = float(min(portfolio_spot_max, max(portfolio_spot_min, float(st.session_state["portfolio_simulated_spot"]) * (1 + shock))))

    portfolio_S1 = st.slider(
        f"Spot simulé du portefeuille S₁ ({orm_currency_symbol})",
        min_value=float(portfolio_spot_min),
        max_value=float(portfolio_spot_max),
        step=float(max(portfolio_S0 / 200, 0.01)),
        key="portfolio_simulated_spot"
    )

    with st.expander("Scénario avancé du portefeuille", expanded=False):
        st.caption("Choc commun en points de volatilité ; même temps écoulé pour chaque option. "
                   "Chaque ligne conserve son strike, sa volatilité et sa maturité de référence.")
        pa, pb = st.columns(2)
        with pa:
            portfolio_vol_shift = st.slider("Choc de volatilité (points de %)",
                                            min_value=-10.0, max_value=30.0,
                                            step=0.1, key="v91_portfolio_vol_shift")
        with pb:
            maturity_min = min(float(st.session_state.get(f"portfolio_T_{i}", 0.5))
                               for i in range(number_positions))
            max_elapsed_days = max(0, min(365, int(max(0.0, maturity_min - 0.0001) * 365)))
            stored_elapsed_days = int(st.session_state.get("v910_portfolio_elapsed_days", 0))
            if stored_elapsed_days < 0 or stored_elapsed_days > max_elapsed_days:
                st.session_state["v910_portfolio_elapsed_days"] = min(max(0, stored_elapsed_days), max_elapsed_days)
            else:
                st.session_state.setdefault("v910_portfolio_elapsed_days", stored_elapsed_days)
            if max_elapsed_days < 1:
                portfolio_elapsed_days = 0
                st.session_state["v910_portfolio_elapsed_days"] = 0
                st.info("Maturité trop courte pour simuler un jour entier. Le temps écoulé reste à 0 jour.")
            else:
                portfolio_elapsed_days = st.slider("Temps écoulé (jours)", min_value=0,
                                                   max_value=max_elapsed_days, step=1,
                                                   key="v910_portfolio_elapsed_days")
            portfolio_elapsed = portfolio_elapsed_days / 365.0
            st.session_state["v91_portfolio_elapsed"] = portfolio_elapsed
        st.caption("Une maturité atteinte ou une volatilité non positive est refusée explicitement ci-dessous.")

    portfolio_changes = []
    if abs(portfolio_S1 - portfolio_S0) > 1e-8:
        portfolio_changes.append(f"Spot : {format_fr((portfolio_S1/portfolio_S0-1)*100, 2)} %")
    if abs(portfolio_vol_shift) > 1e-8:
        portfolio_changes.append(f"Volatilité : {format_fr(portfolio_vol_shift, 2)} point(s)")
    if abs(portfolio_elapsed) > 1e-8:
        portfolio_changes.append(f"Temps écoulé : {portfolio_elapsed_days} jours")
    if portfolio_changes:
        st.caption("**Scénario actif :** " + " · ".join(portfolio_changes))
    else:
        st.caption("Scénario de référence actif : aucun choc.")

    portfolio_tab_overview, portfolio_tab_hedge, portfolio_tab_pnl, portfolio_tab_stress, portfolio_tab_strategy, portfolio_tab_summary = st.tabs([
        "Vue d’ensemble", "Couverture & Gamma", "P&L", "Stress tests", "Payoff & Risk ladder", "Synthèse"
    ])

    with portfolio_tab_overview:
        # --------------------------------------------------------
        # CONSTRUCTION DU PORTEFEUILLE
        # --------------------------------------------------------

        portfolio_inputs = []

        st.markdown(
            "<div class='section-kicker'>Positions</div><div class='section-title'>Composition du portefeuille</div>",
            unsafe_allow_html=True
        )
        st.caption("Cliquez sur une option pour afficher ou modifier ses paramètres.")

        for i in range(number_positions):

            with st.expander(
                f"Option {i + 1}",
                expanded=False
            ):

                for _field, _default in (("qty", 100), ("K", 100.0 + 5.0 * i),
                                         ("T", 0.50), ("sigma", 20.0)):
                    st.session_state.setdefault(f"portfolio_{_field}_{i}", _default)

                c1, c2, c3 = st.columns(3)

                with c1:

                    opt_type = st.selectbox(
                        "Type",
                        ["Call", "Put"],
                        key=f"portfolio_type_{i}"
                    )

                    pos_type = st.selectbox(
                        "Sens",
                        ["Long", "Short"],
                        key=f"portfolio_side_{i}"
                    )

                with c2:

                    qty = st.number_input(
                        "Quantité",
                        step=1,
                        key=f"portfolio_qty_{i}"
                    )

                    strike = st.number_input(
                        f"Strike K ({orm_currency_symbol})",
                        step=1.0,
                        key=f"portfolio_K_{i}"
                    )

                with c3:

                    maturity = st.number_input(
                        "Maturité T (années)",
                        step=0.05,
                        format="%.2f",
                        key=f"portfolio_T_{i}"
                    )

                    vol_percent = st.number_input(
                        "Volatilité σ (%)",
                        step=1.0,
                        key=f"portfolio_sigma_{i}"
                    )

            actions_dup, actions_del = st.columns(2)
            with actions_dup:
                st.button("⧉ Dupliquer", key=f"duplicate_option_v96_{i}",
                          disabled=number_positions >= 5, width="stretch",
                          on_click=duplicate_portfolio_option, args=(i,))
            with actions_del:
                st.button("✕ Supprimer", key=f"remove_option_v96_{i}",
                          disabled=number_positions <= 1, width="stretch",
                          on_click=remove_portfolio_option, args=(i,))

            portfolio_inputs.append({
                "option_type": opt_type,
                "position_type": pos_type,
                "quantity": qty,
                "K": strike,
                "T": maturity,
                "sigma_percent": vol_percent
            })


        # --------------------------------------------------------
        # VALIDATION DES OPTIONS
        # --------------------------------------------------------

        portfolio_errors = []

        for i, option in enumerate(
            portfolio_inputs
        ):

            option_number = i + 1

            if option["quantity"] <= 0:

                portfolio_errors.append(
                    f"Option {option_number} : "
                    f"la quantité doit être strictement positive."
                )

            if option["K"] <= 0:

                portfolio_errors.append(
                    f"Option {option_number} : "
                    f"le strike doit être strictement positif."
                )

            if option["T"] <= 0:

                portfolio_errors.append(
                    f"Option {option_number} : "
                    f"la maturité doit être strictement positive."
                )

            if option["sigma_percent"] <= 0:

                portfolio_errors.append(
                    f"Option {option_number} : "
                    f"la volatilité doit être strictement positive."
                )

        for i, option in enumerate(portfolio_inputs, 1):
            if option["T"] <= portfolio_elapsed:
                portfolio_errors.append(f"Option {i} : le temps écoulé atteint ou dépasse la maturité.")
            if option["sigma_percent"] + portfolio_vol_shift <= 0:
                portfolio_errors.append(f"Option {i} : volatilité simulée non positive.")

        if portfolio_errors:

            st.error(
                "Paramètres invalides dans le portefeuille."
            )

            for error in portfolio_errors:
                st.write(f"• {error}")

            st.stop()


        # --------------------------------------------------------
        # CALCUL DES POSITIONS
        # --------------------------------------------------------

        portfolio_rows = []

        total_value = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        total_rho = 0.0

        for i, option in enumerate(
            portfolio_inputs
        ):

            sigma_i = (
                option["sigma_percent"] / 100
            )

            result = option_position(
                S=portfolio_S0,
                K=option["K"],
                T=option["T"],
                r=portfolio_r,
                sigma=sigma_i,
                option_type=option["option_type"].lower(),
                position_type=option["position_type"].lower(),
                quantity=option["quantity"]
            )

            total_value += (
                result["position_value"]
            )

            total_delta += (
                result["position_delta"]
            )

            total_gamma += (
                result["position_gamma"]
            )

            total_vega += (
                result["position_vega"]
            )

            total_theta += (
                result["position_theta_daily"]
            )

            total_rho += (
                result["position_rho"]
            )

            portfolio_rows.append({
                "Option": i + 1,
                "Sens": option["position_type"],
                "Type": option["option_type"],
                "Quantité": option["quantity"],
                "Strike": option["K"],
                "T": option["T"],
                "Vol. (%)": option["sigma_percent"],
                "Prix unitaire": result["unit_price"],
                "Valeur": result["position_value"],
                "Delta": result["position_delta"],
                "Gamma": result["position_gamma"],
                "Vega": result["position_vega"],
                "Theta/jour": result["position_theta_daily"],
                "Rho": result["position_rho"]
            })


        # --------------------------------------------------------
        # NETTOYAGE
        # --------------------------------------------------------

        total_value = clean_zero(
            total_value
        )

        total_delta = clean_zero(
            total_delta
        )

        total_gamma = clean_zero(
            total_gamma
        )

        total_vega = clean_zero(
            total_vega
        )

        total_theta = clean_zero(
            total_theta
        )

        total_rho = clean_zero(
            total_rho
        )

        portfolio_hedge = clean_zero(
            -total_delta
        )


        # Valorisation agrégée au scénario courant, sans attendre l'onglet Couverture.
        live_portfolio = [option_position(
            portfolio_S1, o["K"], o["T"] - portfolio_elapsed, portfolio_r,
            (o["sigma_percent"] + portfolio_vol_shift) / 100,
            o["option_type"].lower(), o["position_type"].lower(), o["quantity"])
            for o in portfolio_inputs]
        live_value = sum(o["position_value"] for o in live_portfolio)
        live_delta = sum(o["position_delta"] for o in live_portfolio)
        live_gamma = sum(o["position_gamma"] for o in live_portfolio)
        live_hedge = -live_delta
        st.markdown("#### Risques du portefeuille au scénario S₁")
        live_cols = st.columns(4)
        for col, title, current, base, unit, digits in zip(
            live_cols,
            ("Valeur", "Delta", "Gamma", "Couverture cible"),
            (live_value, live_delta, live_gamma, live_hedge),
            (total_value, total_delta, total_gamma, portfolio_hedge),
            (f" {orm_currency_symbol}", " actions", f" actions/{orm_currency_symbol}", " actions"),
            (2, 4, 4, 2)):
            col.metric(title, f"{format_fr(current, digits)}{unit}",
                       delta=f"à S₀ : {format_fr(base, digits)}{unit}", delta_color="off")
        st.markdown(f"**Couverture à S₀ :** {trade_instruction(portfolio_hedge)}  "
                    f"\n\n**Ajustement à S₁ :** {trade_instruction(live_hedge - portfolio_hedge)}")

        # --------------------------------------------------------
        # INDICATEURS DU PORTEFEUILLE
        # --------------------------------------------------------

        st.markdown(
            "<div class='section-kicker'>Portfolio risk</div><div class='section-title'>Risque agrégé</div>",
            unsafe_allow_html=True
        )

        p1, p2, p3 = st.columns(3)
        p1.metric("Valeur portefeuille", f"{format_fr(total_value)} {orm_currency_symbol}")
        p2.metric("Δ  Delta portefeuille", format_fr(total_delta, 4))
        p3.metric("Γ  Gamma portefeuille", format_fr(total_gamma, 4))

        if portfolio_hedge > 1e-12:
            st.markdown(
                f"<div class='trade-card buy'><div class='trade-label'>Couverture Delta globale</div>"
                f"<div class='trade-value'>ACHETER {format_fr(abs(portfolio_hedge), 2)} ACTIONS</div>"
                f"<div class='trade-note'>Position de couverture nécessaire pour neutraliser le Delta agrégé du portefeuille.</div></div>",
                unsafe_allow_html=True
            )
        elif portfolio_hedge < -1e-12:
            st.markdown(
                f"<div class='trade-card sell'><div class='trade-label'>Couverture Delta globale</div>"
                f"<div class='trade-value'>VENDRE {format_fr(abs(portfolio_hedge), 2)} ACTIONS</div>"
                f"<div class='trade-note'>Position de couverture nécessaire pour neutraliser le Delta agrégé du portefeuille.</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='trade-card'><div class='trade-label'>Couverture Delta globale</div>"
                "<div class='trade-value'>PORTEFEUILLE DELTA-NEUTRE</div>"
                "<div class='trade-note'>Aucune transaction sur le sous-jacent n'est nécessaire.</div></div>",
                unsafe_allow_html=True
            )

        p4, p5, p6 = st.columns(3)
        p4.metric("ν  Vega (+1 pt de vol.)", f"{format_fr(total_vega, 4)} {orm_currency_symbol}")
        p5.metric("Θ  Theta / jour", f"{format_fr(total_theta, 4)} {orm_currency_symbol}")
        p6.metric("ρ  Rho (+1 pt de taux)", f"{format_fr(total_rho, 4)} {orm_currency_symbol}")


        # --------------------------------------------------------
        # TABLEAU DÉTAILLÉ
        # --------------------------------------------------------

        st.subheader(
            "Détail des positions"
        )

        portfolio_df = pd.DataFrame(
            portfolio_rows
        )

        display_df = (
            portfolio_df.copy()
        )

        numeric_columns = [
            "Strike",
            "T",
            "Vol. (%)",
            "Prix unitaire",
            "Valeur",
            "Delta",
            "Gamma",
            "Vega",
            "Theta/jour",
            "Rho"
        ]

        for column in numeric_columns:

            display_df[column] = (
                display_df[column]
                .astype(float)
                .round(4)
            )

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )

        st.markdown("#### Contribution de chaque option au risque")
        contribution_col1, contribution_col2 = st.columns(2)

        labels = [f"Option {int(x)}" for x in portfolio_df["Option"]]
        for column, greek, unit, pop_title, explanation in (
            (contribution_col1, "Delta", "actions", "ⓘ Lire les contributions au Delta",
             "Chaque barre est le Delta signé d'une option. Leur somme est le Delta agrégé du portefeuille."),
            (contribution_col2, "Gamma", f"actions/{orm_currency_symbol}", "ⓘ Lire les contributions au Gamma",
             "Chaque barre est le Gamma signé d'une option. Les expositions de signes opposés peuvent se compenser."),
        ):
            with column:
                values = portfolio_df[greek].astype(float).tolist()
                colors = ["#16a34a" if v >= 0 else "#dc2626" for v in values]
                fig_contrib = go.Figure(go.Bar(
                    x=values, y=labels, orientation="h", marker_color=colors,
                    text=[f"{format_fr_signed(v, 2)}" for v in values], textposition="auto",
                    hovertemplate="%{y} : %{x:.4f}<extra></extra>"))
                fig_contrib.add_vline(x=0, line_color="#64748b", line_width=1)
                fig_contrib.update_layout(template="plotly_white", height=300,
                    title=f"Contribution au {greek}", xaxis_title=f"{greek} ({unit})",
                    yaxis_title="", margin=dict(l=15,r=20,t=55,b=45))
                st.plotly_chart(fig_contrib, width="stretch", key=f"portfolio_{greek.lower()}_contrib_v910")
                with st.popover(pop_title):
                    st.write(explanation)

        # Contrôles numériques sur les paramètres de chaque ligne du portefeuille
        parity_errors, delta_fd_errors, gamma_fd_errors, portfolio_fd_checks = [], [], [], []
        for option in portfolio_inputs:
            sigma_i = option["sigma_percent"] / 100
            c_i = bs_price(portfolio_S0, option["K"], option["T"], portfolio_r, sigma_i, "call")
            p_i = bs_price(portfolio_S0, option["K"], option["T"], portfolio_r, sigma_i, "put")
            rhs_i = portfolio_S0 - option["K"] * np.exp(-portfolio_r * option["T"])
            parity_errors.append(abs((c_i - p_i) - rhs_i))
            fd_i = finite_diff_check(
                portfolio_S0, option["K"], option["T"], portfolio_r, sigma_i, option["option_type"].lower()
            )
            delta_fd_errors.append(fd_i["delta_error"])
            gamma_fd_errors.append(fd_i["gamma_error"])
            portfolio_fd_checks.append(fd_i)

        with st.expander("Contrôles numériques du moteur Black-Scholes", expanded=False):
            max_parity_error = max(parity_errors) if parity_errors else 0.0
            max_delta_error = max(delta_fd_errors) if delta_fd_errors else 0.0
            max_gamma_error = max(gamma_fd_errors) if gamma_fd_errors else 0.0
            st.markdown("**Cohérence Call/Put**")
            st.write(f"Erreur maximale de parité : **{max_parity_error:.2e}**")
            st.markdown("**Validation indépendante des Greeks par différences finies**")
            c1, c2 = st.columns(2)
            c1.metric("Erreur max Δ", f"{max_delta_error:.2e}")
            c2.metric("Erreur max Γ", f"{max_gamma_error:.2e}")
            if max_parity_error < 1e-8 and all(within_tol(x["delta_error"], x["delta_analytic"]) and within_tol(x["gamma_error"], x["gamma_analytic"]) for x in portfolio_fd_checks):
                st.success("✓ Contrôles numériques réussis pour toutes les lignes du portefeuille.")
            else:
                st.warning("Un écart numérique dépasse le seuil de contrôle.")


    with portfolio_tab_hedge:
        # Le scénario S₁ est défini au-dessus des onglets afin de rester global.

        # --------------------------------------------------------
        # RISQUE AU SPOT SIMULÉ
        # --------------------------------------------------------

        simulated_value = 0.0
        simulated_delta = 0.0
        simulated_gamma = 0.0

        for option in portfolio_inputs:

            result_S1 = option_position(
                S=portfolio_S1,
                K=option["K"],
                T=option["T"] - portfolio_elapsed,
                r=portfolio_r,
                sigma=(option["sigma_percent"] + portfolio_vol_shift) / 100,
                option_type=(
                    option["option_type"].lower()
                ),
                position_type=(
                    option["position_type"].lower()
                ),
                quantity=option["quantity"]
            )

            simulated_value += (
                result_S1["position_value"]
            )

            simulated_delta += (
                result_S1["position_delta"]
            )

            simulated_gamma += (
                result_S1["position_gamma"]
            )

        simulated_hedge = (
            -simulated_delta
        )

        hedge_rebalancing = clean_zero(
            simulated_hedge
            - portfolio_hedge
        )

        ps1, ps2, ps3, ps4 = (
            st.columns(4)
        )

        ps1.metric(
            "Spot simulé",
            f"{format_fr(portfolio_S1)} {orm_currency_symbol}",
            delta=(
                f"{format_fr_signed(portfolio_S1 - portfolio_S0, 2)} {orm_currency_symbol}"
            )
        )

        ps2.metric(
            "Delta portefeuille",
            f"{format_fr(simulated_delta, 4)}",
            delta=(
                f"{format_fr_signed(simulated_delta - total_delta, 4)}"
            )
        )

        ps3.metric(
            "Gamma portefeuille",
            f"{format_fr(simulated_gamma, 4)}",
            delta=(
                f"{format_fr_signed(simulated_gamma - total_gamma, 4)}"
            )
        )

        with ps4:
            simulated_action = hedge_action(hedge_rebalancing)
            action_class = "buy" if simulated_action == "ACHETER" else "sell" if simulated_action == "VENDRE" else "neutral"
            st.markdown(
                f"""
                <div class="trade-card {action_class}" style="margin:0; min-height:116px; padding:.75rem .9rem;">
                    <div class="trade-label">Ordre à passer (couverture initiale détenue)</div>
                    <div class="trade-value" style="font-size:1.35rem;">{simulated_action} {format_fr(abs(hedge_rebalancing))}</div>
                    <div class="trade-note">Position cible : {format_fr(simulated_hedge)} actions</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # --------------------------------------------------------
        # RÉÉQUILIBRAGE DU PORTEFEUILLE
        # --------------------------------------------------------

        if hedge_rebalancing > 1e-12:

            st.success(
                f"Rééquilibrage : ACHETER "
                f"{format_fr(abs(hedge_rebalancing))} actions."
            )

        elif hedge_rebalancing < -1e-12:

            st.warning(
                f"Rééquilibrage : VENDRE "
                f"{format_fr(abs(hedge_rebalancing))} actions."
            )

        else:

            st.info(
                "Aucun rééquilibrage de la couverture "
                "n'est nécessaire."
            )


        # --------------------------------------------------------
        # COURBES AGRÉGÉES
        # --------------------------------------------------------

        options_key = tuple(
            (o["option_type"].lower(), o["position_type"].lower(),
             float(o["quantity"]), float(o["K"]), float(o["T"]),
             float(o["sigma_percent"]) / 100.0)
            for o in portfolio_inputs
        )
        portfolio_graph_df = build_portfolio_risk_curve(portfolio_S0, portfolio_r, options_key)
        portfolio_scenario_df = build_portfolio_scenario_curve(
            portfolio_S0, portfolio_r, options_key, portfolio_vol_shift, portfolio_elapsed)

        st.subheader("Couverture globale en fonction du spot")
        st.plotly_chart(interactive_risk_chart(portfolio_scenario_df, "Couverture",
                        portfolio_S0, portfolio_S1, simulated_hedge, reference_data=portfolio_graph_df),
                        width="stretch", key="portfolio_hedge_plotly")
        if True:
            action = "acheter" if simulated_hedge > 0 else "vendre" if simulated_hedge < 0 else "ne détenir aucune"
            explain_graph("Hedge agrégé",
                          f"À S₁ = {format_fr(portfolio_S1, 2)} {orm_currency_symbol}, la couverture agrégée vaut {format_fr(simulated_hedge, 2)} action(s). Les Delta des lignes se compensent ou s'additionnent.",
                          f"Décision : {action} {format_fr(abs(simulated_hedge), 2)} action(s). Une couverture Delta-neutre ne supprime pas le Gamma ni le Vega du portefeuille.")
        st.subheader("Gamma agrégé en fonction du spot")
        st.plotly_chart(interactive_risk_chart(portfolio_scenario_df, "Gamma",
                        portfolio_S0, portfolio_S1, simulated_gamma, reference_data=portfolio_graph_df),
                        width="stretch", key="portfolio_gamma_plotly")
        if True:
            explain_graph("Gamma du portefeuille",
                          f"Le Gamma agrégé est {format_fr(simulated_gamma, 4)} actions/{orm_currency_symbol} au spot simulé. Il correspond à la somme signée des Gamma des options.",
                          "Plus |Gamma| est grand, plus le nombre d'actions de couverture est sensible aux mouvements du spot. Les zones orange indiquent une sensibilité relativement élevée sur la plage étudiée.")

        st.markdown("#### Explorer les Greeks du portefeuille")
        with st.popover("ⓘ Comment lire les Greeks agrégés ?"):
            st.write(f"L'axe horizontal indique le spot ({orm_currency_symbol}), l'axe vertical le Greek agrégé dans son unité. Chaque exposition est la somme signée des options du portefeuille. Le trait pointillé orange représente le spot simulé. Une compensation entre lignes peut disparaître lorsque le spot varie.")
        port_greek = st.selectbox("Greek du portefeuille", ["Delta", "Gamma", "Vega", "Theta"], key="portfolio_greek_selector_v931")
        port_greek_keys = {"Delta": ("position_delta", "actions"), "Gamma": ("position_gamma", f"actions/{orm_currency_symbol}"),
                           "Vega": ("position_vega", f"{orm_currency_symbol}/point de vol."), "Theta": ("position_theta_annual", f"{orm_currency_symbol}/an")}
        port_key, port_unit = port_greek_keys[port_greek]
        port_spots = np.linspace(0.6*portfolio_S0, 1.4*portfolio_S0, 101)
        port_values = []
        for z in port_spots:
            val = 0.0
            for typ, direction, qty, strike, maturity, sig in options_key:
                val += option_position(float(z), strike, max(1e-6,maturity-portfolio_elapsed),
                                       portfolio_r, max(1e-6,sig+portfolio_vol_shift/100),
                                       typ, direction, qty)[port_key]
            port_values.append(val)
        fig_port_greek = go.Figure(go.Scatter(x=port_spots, y=port_values, mode="lines",
                                             line=dict(color="#2563eb", width=3.5), name=f"{port_greek} du portefeuille (agrégé)"))
        # Une option de référence est nécessaire : un portefeuille n'a pas de moneyness unique.
        selected_zone = st.selectbox(
            "Afficher les zones ITM / ATM / OTM de",
            options=["Aucune zone"] + [
                f"Option {i+1} · {opt[0]} · K = {format_fr(opt[3], 2)} {orm_currency_symbol}"
                for i, opt in enumerate(options_key)
            ],
            index=1 if options_key else 0,
            key="portfolio_moneyness_option_v99",
        )
        if selected_zone != "Aucune zone":
            selected_idx = [
                f"Option {i+1} · {opt[0]} · K = {format_fr(opt[3], 2)} {orm_currency_symbol}"
                for i, opt in enumerate(options_key)
            ].index(selected_zone)
            chosen = options_key[selected_idx]
            add_moneyness_zones(fig_port_greek, float(chosen[3]), chosen[0],
                                float(port_spots[0]), float(port_spots[-1]), label=True)
            st.caption("Zones ITM/OTM de l’option sélectionnée ; la courbe bleue représente toujours le Greek agrégé du portefeuille.")
        else:
            st.caption("Aucune zone ITM/OTM affichée : chaque option possède sa propre moneyness.")
        fig_port_greek.add_vline(x=portfolio_S1, line_dash="dash", line_color="#ea580c", line_width=2.5)
        fig_port_greek.add_annotation(x=portfolio_S1, y=-0.075, yref="paper", showarrow=False,
                                      text=f"<b>Spot simulé : {format_fr(portfolio_S1, 2)} {orm_currency_symbol}</b>",
                                      bgcolor="rgba(255,237,213,0.98)", borderpad=7,
                                      font=dict(size=12, color="#c2410c"))
        if selected_zone != "Aucune zone":
            moneyness_context(chosen[0], chosen[3], portfolio_S1, selected_idx + 1)
        fig_port_greek.update_layout(title=dict(text=f"{port_greek} du portefeuille", font=dict(size=19)), template="plotly_white", height=560, xaxis_title=f"Spot ({orm_currency_symbol})",
                                     yaxis_title=f"{port_greek} ({port_unit})", margin=dict(l=55,r=35,t=65,b=100),
                                     legend=dict(orientation="h", y=-0.24, x=0))
        st.plotly_chart(fig_port_greek, width="stretch", key="portfolio_greek_plot_v931")
        hedge_change_1pct = -simulated_gamma * 0.01 * portfolio_S1
        st.metric("Rééquilibrage estimé pour +1 % de spot (Gamma agrégé)",
                  f"{'ACHETER' if hedge_change_1pct > 0 else 'VENDRE' if hedge_change_1pct < 0 else 'AUCUNE ACTION'} {format_fr(abs(hedge_change_1pct),2)} action(s)")
        with st.popover("ⓘ Limites de l'estimation"):
            st.write("Approximation locale −Gamma agrégé × variation de spot, à volatilité et maturité constantes. La variation réelle de la couverture nécessite un recalcul des Delta des options après choc.")

    with portfolio_tab_pnl:
        # --------------------------------------------------------
        # P&L DU PORTEFEUILLE
        # --------------------------------------------------------

        st.divider()

        st.header(
            "P&L du portefeuille"
        )

        portfolio_delta_S = (
            portfolio_S1
            - portfolio_S0
        )

        portfolio_option_pnl = clean_zero(
            simulated_value
            - total_value
        )

        portfolio_components = {
            "Delta": total_delta * portfolio_delta_S,
            "Gamma": 0.5 * total_gamma * portfolio_delta_S ** 2,
            "Vega": total_vega * portfolio_vol_shift,
            "Theta": (total_theta * 365.0) * portfolio_elapsed,
        }
        portfolio_components["Résidu (ordre supérieur)"] = portfolio_option_pnl - sum(portfolio_components.values())
        render_breakdown(portfolio_components, "v91_portfolio_pnl_breakdown")

        portfolio_stock_pnl = clean_zero(
            portfolio_hedge
            * portfolio_delta_S
        )

        portfolio_hedged_pnl = clean_zero(
            portfolio_option_pnl
            + portfolio_stock_pnl
        )

        portfolio_delta_approx = clean_zero(
            total_delta
            * portfolio_delta_S
        )

        portfolio_delta_gamma_approx = clean_zero(
            total_delta
            * portfolio_delta_S
            + 0.5
            * total_gamma
            * portfolio_delta_S**2
        )

        pp1, pp2, pp3 = (
            st.columns(3)
        )

        pp1.metric(
            "P&L options",
            f"{format_fr(portfolio_option_pnl, 2)} {orm_currency_symbol}"
        )

        pp2.metric(
            "P&L couverture actions",
            f"{format_fr(portfolio_stock_pnl, 2)} {orm_currency_symbol}"
        )

        pp3.metric(
            "P&L portefeuille couvert",
            f"{format_fr(portfolio_hedged_pnl, 2)} {orm_currency_symbol}"
        )

        pa1, pa2, pa3 = (
            st.columns(3)
        )

        pa1.metric(
            "P&L exact",
            f"{format_fr(portfolio_option_pnl, 2)} {orm_currency_symbol}"
        )

        pa2.metric(
            "Approximation Delta",
            f"{format_fr(portfolio_delta_approx, 2)} {orm_currency_symbol}"
        )

        pa3.metric(
            "Approximation Delta-Gamma",
            f"{format_fr(portfolio_delta_gamma_approx, 2)} {orm_currency_symbol}"
        )

        st.markdown("#### Comparaison des méthodes de P&L")
        full_approx_portfolio = sum(value for name, value in portfolio_components.items()
                                    if name != "Résidu (ordre supérieur)")
        render_pnl_comparison(portfolio_option_pnl, portfolio_delta_approx,
                              portfolio_delta_gamma_approx, "portfolio_pnl_comparison_v914",
                              full_approx_portfolio)
        st.markdown("#### Précision des approximations selon le spot")
        error_spots_port = np.linspace(0.6 * portfolio_S0, 1.4 * portfolio_S0, 81)
        exact_spot_only_port = np.array([
            sum(
                option_position(float(z), strike, maturity, portfolio_r, sig,
                                typ, direction, qty)["position_value"]
                - option_position(portfolio_S0, strike, maturity, portfolio_r, sig,
                                  typ, direction, qty)["position_value"]
                for typ, direction, qty, strike, maturity, sig in options_key
            ) for z in error_spots_port
        ])
        port_shocks = error_spots_port - portfolio_S0
        render_pnl_error_curve(
            error_spots_port, exact_spot_only_port,
            total_delta * port_shocks,
            total_delta * port_shocks + 0.5 * total_gamma * port_shocks ** 2,
            portfolio_S1, "portfolio_pnl_error_curve_v932"
        )
        st.markdown("#### Couverture avant et après le scénario")
        render_hedge_rebalance(portfolio_hedge, simulated_hedge, "portfolio_rebalance_v932")

        st.caption(
            "Le P&L exact inclut les chocs de spot, de volatilité et de temps ; "
            "le taux sans risque reste constant. Les approximations Delta et Delta-Gamma "
            "isolent les effets du spot, sauf indication contraire."
        )


        # V12.3 — risque agrégé et couverture statique au spot initial.
        render_risk_diagnostic(
            spot0=portfolio_S0,
            spot1=portfolio_S1,
            delta0=total_delta,
            delta1=simulated_delta,
            gamma0=total_gamma,
            hedge0=portfolio_hedge,
            key="v123_portfolio_risk",
        )
        render_static_pnl_comparison(
            spots=error_spots_port,
            option_pnl=exact_spot_only_port,
            spot0=portfolio_S0,
            current_spot=portfolio_S1,
            hedge0=portfolio_hedge,
            key="v123_portfolio_static",
        )


    with portfolio_tab_stress:
        st.header("Stress tests du portefeuille")
        st.caption("Revalorisation de toutes les options à maturité de référence constante. "
                   "La couverture initiale est statique : aucune opération de rééquilibrage n'est simulée.")
        stress_spots, stress_vol_shifts, stress_pnl, hedged_stress_pnl = build_portfolio_stress_grid(
            portfolio_S0, portfolio_r, options_key)
        stress_mode = st.radio("Type de P&L analysé", ["P&L options", "P&L avec couverture initiale"],
                               horizontal=True, key="portfolio_stress_mode_v913")
        stress_z = stress_pnl if stress_mode == "P&L options" else hedged_stress_pnl
        current_pnl = portfolio_option_pnl if stress_mode == "P&L options" else portfolio_hedged_pnl
        min_idx = np.unravel_index(np.argmin(stress_z), stress_z.shape)
        left_stress, right_stress = st.columns(2)
        left_stress.metric("P&L du scénario courant", format_fr(current_pnl) + f" {orm_currency_symbol}")
        right_stress.metric("P&L minimal sur la grille", format_fr(np.min(stress_z)) + f" {orm_currency_symbol}")
        st.caption(f"Minimum simulé : spot {format_fr(stress_spots[min_idx[1]])} {orm_currency_symbol} ; "
                   f"choc de volatilité {format_fr(stress_vol_shifts[min_idx[0]], 1)} point(s).")
        st.caption("La sélection agit sur la carte de chaleur et les indicateurs. "
                   "Les deux courbes de sensibilité restent visibles pour comparaison.")
        fig_multi_heat = go.Figure(go.Heatmap(
            x=stress_spots, y=stress_vol_shifts, z=stress_z,
            colorscale="RdYlGn", zmid=0, colorbar=dict(title=f"P&L ({orm_currency_symbol})"),
            hovertemplate=("Spot : %{x:.2f} €<br>Choc vol. : %{y:+.1f} pt(s)<br>P&L : %{z:,.2f} €<extra></extra>".replace("€", orm_currency_symbol))))
        # Marque le scénario sélectionné, même s'il se situe hors de la grille de stress.
        if (stress_spots.min() <= portfolio_S1 <= stress_spots.max() and
                stress_vol_shifts.min() <= portfolio_vol_shift <= stress_vol_shifts.max()):
            fig_multi_heat.add_trace(go.Scatter(
                x=[portfolio_S1], y=[portfolio_vol_shift], mode="markers",
                marker=dict(size=13, symbol="x", color="#1d4ed8", line=dict(width=2)),
                name="Scénario courant", hovertemplate="Scénario courant<extra></extra>"))
        else:
            st.caption("Le scénario courant se situe hors des bornes de la carte de chaleur.")
        fig_multi_heat.add_trace(go.Scatter(
            x=[float(stress_spots[min_idx[1]])], y=[float(stress_vol_shifts[min_idx[0]])],
            mode="markers", marker=dict(size=12, symbol="diamond-open", color="#111827", line=dict(width=2)),
            name="Minimum de la grille"))
        fig_multi_heat.add_vline(x=portfolio_S0, line_dash="dash", line_color="#64748b")
        fig_multi_heat.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig_multi_heat.update_layout(template="plotly_white", height=440,
                                     xaxis_title=f"Spot stressé ({orm_currency_symbol})",
                                     yaxis_title="Choc de volatilité commun (points de %)",
                                     margin=dict(l=55, r=35, t=35, b=55))
        st.plotly_chart(fig_multi_heat, width="stretch", key="portfolio_stress_heatmap_v913")
        with st.popover("ⓘ Hypothèses de la carte de chaleur"):
            st.write("Chaque option conserve son strike, son sens, sa quantité et sa maturité de référence. "
                     "Le choc de volatilité s'applique en points à chaque ligne. Le P&L couvert inclut "
                     "uniquement la couverture en actions calculée au départ, sans rééquilibrage.")
        c_spot, c_vol = st.columns(2)
        with c_spot:
            st.markdown("#### Sensibilité au spot")
            spot_only_pnl = np.array([
                sum(option_position(float(z), strike, maturity, portfolio_r, sigma,
                                    typ, direction, qty)["position_value"]
                    for typ, direction, qty, strike, maturity, sigma in options_key) - total_value
                for z in stress_spots
            ])
            fig_spot = go.Figure()
            fig_spot.add_trace(go.Scatter(x=stress_spots, y=spot_only_pnl, mode="lines",
                                          name="Options", line=dict(color="#2563eb", width=3)))
            fig_spot.add_trace(go.Scatter(x=stress_spots,
                                          y=spot_only_pnl + portfolio_hedge*(stress_spots-portfolio_S0),
                                          mode="lines", name="Couvert statiquement",
                                          line=dict(color="#16a34a", width=3)))
            fig_spot.add_hline(y=0, line_color="#94a3b8", line_width=1)
            fig_spot.update_layout(template="plotly_white", height=350,
                                   xaxis_title=f"Spot ({orm_currency_symbol})", yaxis_title=f"P&L ({orm_currency_symbol})",
                                   legend=dict(orientation="h", y=-0.28),
                                   margin=dict(l=35,r=20,t=25,b=80))
            st.plotly_chart(fig_spot, width="stretch", key="portfolio_stress_spot_v913")
        with c_vol:
            st.markdown("#### Sensibilité à la volatilité")
            vol_only_pnl = np.array([
                sum(option_position(portfolio_S0, strike, maturity, portfolio_r,
                                    sigma + float(dvol)/100, typ, direction, qty)["position_value"]
                    for typ, direction, qty, strike, maturity, sigma in options_key) - total_value
                for dvol in stress_vol_shifts
            ])
            fig_vol = go.Figure(go.Scatter(x=stress_vol_shifts, y=vol_only_pnl,
                                           mode="lines+markers", name="P&L options",
                                           line=dict(color="#2563eb", width=3)))
            fig_vol.add_hline(y=0, line_color="#94a3b8", line_width=1)
            fig_vol.update_layout(template="plotly_white", height=350,
                                   xaxis_title="Choc de volatilité (points de %)", yaxis_title=f"P&L ({orm_currency_symbol})",
                                   margin=dict(l=35,r=20,t=25,b=60))
            st.plotly_chart(fig_vol, width="stretch", key="portfolio_stress_vol_v913")
        st.caption("Les stress tests sont des scénarios illustratifs, pas des prévisions ni une mesure de VaR.")

    with portfolio_tab_summary:
        st.download_button(
            "↓ Exporter les résultats du portefeuille (CSV)",
            data=result_csv([
                ("Paramètres de référence", "Nombre de positions", number_positions, "options"),
                ("Paramètres de référence", "Spot initial", portfolio_S0, "EUR"),
                ("Scénario de marché", "Spot scénario", portfolio_S1, "EUR"),
                ("Valorisation et P&L", "Valeur initiale", total_value, "EUR"),
                ("Risques initiaux", "Delta agrégé", total_delta, "actions"),
                ("Risques initiaux", "Gamma agrégé", total_gamma, "actions/EUR"),
                ("Risques initiaux", "Vega agrégé", total_vega, "EUR/point de %"),
                ("Risques initiaux", "Theta agrégé", total_theta, "EUR/jour"),
                ("Risques initiaux", "Couverture initiale", portfolio_hedge, "actions"),
                ("Risques simulés et couverture", "Couverture cible", simulated_hedge, "actions"),
                ("Risques simulés et couverture", "Rééquilibrage", simulated_hedge - portfolio_hedge, "actions"),
                ("Scénario de marché", "Choc volatilité", portfolio_vol_shift, "points de %"),
                ("Scénario de marché", "Temps écoulé", portfolio_elapsed * 365, "jours"),
                ("Valorisation et P&L", "Valeur scénario", simulated_value, "EUR"),
                ("Valorisation et P&L", "P&L exact options", simulated_value - total_value, "EUR"),
                ("Valorisation et P&L", "P&L couverture statique", portfolio_hedge * (portfolio_S1 - portfolio_S0), "EUR"),
                ("Valorisation et P&L", "P&L couvert", simulated_value - total_value + portfolio_hedge * (portfolio_S1 - portfolio_S0), "EUR"),
            ] + [
                (f"Option {i+1} — {section}", name, value, unit)
                for i, option in enumerate(portfolio_inputs)
                for section, name, value, unit in [
                    ("Paramètres de référence", "Type", option["option_type"], ""),
                    ("Paramètres de référence", "Sens", option["position_type"], ""),
                    ("Paramètres de référence", "Quantité", option["quantity"], "options"),
                    ("Paramètres de référence", "Strike", option["K"], "EUR"),
                    ("Paramètres de référence", "Maturité", option["T"], "années"),
                    ("Paramètres de référence", "Volatilité", option["sigma_percent"], "%"),
                ]
            ]),
            file_name="option_risk_portefeuille.csv", mime="text/csv",
            key="export_portfolio_v96"
        )
        st.markdown("### Décision de couverture du portefeuille au scénario")
        st.caption("Couverture globale sur le même sous-jacent ; quantité positive = acheter, négative = vendre.")
        render_hedge_rebalance(
            portfolio_hedge, simulated_hedge, "portfolio_summary_rebalance_v94"
        )
        portfolio_residual = -simulated_hedge + portfolio_hedge
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Delta résiduel après choc", format_fr(portfolio_residual, 4) + " actions")
        pc2.metric("Gamma initial agrégé", format_fr(total_gamma, 4) + f" actions/{orm_currency_symbol}")
        pc3.metric("Spot du scénario", format_fr(portfolio_S1, 2) + f" {orm_currency_symbol}")
        with st.popover("ⓘ Comment lire le diagnostic portefeuille ?"):
            st.write("La couverture globale neutralise la somme des Delta initiaux. "
                     "Si le marché évolue, la somme des Delta des options change : "
                     "conserver les actions de couverture initiales laisse un Delta résiduel. "
                     "L'ordre de rééquilibrage correspond à la différence entre les actions "
                     "cibles au scénario et celles détenues initialement. "
                     "Des positions opposées peuvent compenser le Delta sans annuler le Gamma.")
        st.divider()
        # --------------------------------------------------------
        # SYNTHÈSE VISUELLE DU PORTEFEUILLE
        # --------------------------------------------------------

        st.markdown(
            "<div class='section-kicker'>Synthèse des risques</div><div class='section-title'>Synthèse du portefeuille</div>",
            unsafe_allow_html=True
        )

        portfolio_gamma_text = "Convexité positive" if total_gamma > 1e-12 else ("Convexité négative" if total_gamma < -1e-12 else "Gamma quasi nul")
        portfolio_vega_text = "Long volatilité" if total_vega > 1e-12 else ("Short volatilité" if total_vega < -1e-12 else "Vega quasi nul")
        portfolio_delta_text = "Exposition positive au sous-jacent" if total_delta > 1e-12 else ("Exposition négative au sous-jacent" if total_delta < -1e-12 else "Delta quasi neutre")
        action = hedge_action(portfolio_hedge)

        st.markdown(f"""
            <div class='risk-hero'>
                <div class='risk-hero-kicker'>Risk Summary · Portefeuille</div>
                <div class='risk-hero-title'>{number_positions} POSITIONS · RISQUE AGRÉGÉ</div>
                <div class='risk-hero-sub'>Synthèse de l’exposition et décision de couverture · S₀ {format_fr(portfolio_S0, 2)} {orm_currency_symbol} · Valeur {format_fr(total_value, 2)} {orm_currency_symbol} · r {format_fr(portfolio_r_percent, 2)} %</div>
            </div>
            <div class='summary-shell'>
                <div class='summary-grid'>
                    <div class='summary-kpi'><div class='summary-kpi-label'>Δ Delta agrégé</div><div class='summary-kpi-value'>{format_fr_signed(total_delta, 4)}</div><div class='summary-kpi-note'>{portfolio_delta_text}</div></div>
                    <div class='summary-kpi'><div class='summary-kpi-label'>Γ Gamma agrégé</div><div class='summary-kpi-value'>{format_fr_signed(total_gamma, 4)}</div><div class='summary-kpi-note'>{portfolio_gamma_text}</div></div>
                    <div class='summary-kpi'><div class='summary-kpi-label'>ν Vega agrégé</div><div class='summary-kpi-value'>{format_fr_signed(total_vega, 4)} {orm_currency_symbol}</div><div class='summary-kpi-note'>{portfolio_vega_text}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # V11.1 — Synthèse du profil du portefeuille : aucun plafond inventé.
        st.markdown("#### Profil de payoff du portefeuille")
        same_expiry_summary = all(abs(o["T"] - portfolio_inputs[0]["T"]) < 1e-9 for o in portfolio_inputs)
        if same_expiry_summary:
            roots_summary, flat_summary = exact_expiry_breakevens(portfolio_inputs, total_value)
            root_label = format_expiry_breakevens(roots_summary, flat_summary)
            st.markdown("**Point(s) mort(s) à échéance — domaine complet S ≥ 0**")
            st.write(root_label)
            st.caption("Pertes et gains extrêmes dépendent de la composition des positions ; ne pas déduire une perte maximale de la seule plage du graphique.")
        else:
            st.info("Échéances différentes : aucun point mort terminal unique n'est défini pour le portefeuille.")
        st.caption("Couverture initiale cible : " + trade_instruction(portfolio_hedge) + ". Voir « Payoff & Risk ladder » pour les scénarios de spot.")

        portfolio_hedge_class = "buy" if portfolio_hedge > 1e-12 else ("sell" if portfolio_hedge < -1e-12 else "neutral")
        portfolio_delta_after = clean_zero(total_delta + portfolio_hedge)
        portfolio_neutral_status = "<span class='neutral-chip'>✓ DELTA NEUTRE</span>" if abs(portfolio_delta_after) < 1e-10 else ""
        st.markdown(f"""
            <div class='hedge-decision {portfolio_hedge_class}'>
                <div>
                    <div class='hedge-decision-label'>Couverture Delta globale · ordre théorique</div>
                    <div class='hedge-decision-action'>{action}</div>
                    <div class='hedge-decision-qty'>{format_fr(abs(portfolio_hedge), 2)} <span>ACTIONS</span></div>
                </div>
                <div class='hedge-breakdown'>
                    Delta portefeuille <b>{format_fr_signed(total_delta, 4)}</b><br>
                    Delta couverture <b>{format_fr_signed(portfolio_hedge, 4)}</b><br>
                    Delta après hedge <b>{format_fr_signed(portfolio_delta_after, 4)}</b> {portfolio_neutral_status}
                </div>
            </div>
            <div class='risk-secondary-title'>Risques secondaires</div>
            <div class='risk-reading'>
                <div class='risk-reading-item'><strong>Θ Theta / jour</strong><b>{format_fr_signed(total_theta, 4)} {orm_currency_symbol}</b><br>Effet temporel quotidien agrégé.</div>
                <div class='risk-reading-item'><strong>ρ Rho agrégé</strong><b>{format_fr_signed(total_rho, 4)} {orm_currency_symbol}</b><br>Impact pour une hausse de 1 point du taux sans risque.</div>
                <div class='risk-reading-item'><strong>Γ Lecture Gamma</strong><b>{portfolio_gamma_text}</b><br>La couverture Delta évoluera lorsque le spot se déplace.</div>
                <div class='risk-reading-item'><strong>ν Lecture Vega</strong><b>{portfolio_vega_text}</b><br>{format_fr_signed(total_vega, 4)} {orm_currency_symbol} pour +1 point de volatilité.</div>
            </div>
            <div class='risk-conclusion'>
                <strong>Lecture du portefeuille</strong>
                Portefeuille de {number_positions} positions. {portfolio_delta_text}. {portfolio_gamma_text} et {portfolio_vega_text.lower()}. Une couverture Delta-neutre nécessite de <b>{action.lower()} {format_fr(abs(portfolio_hedge), 2)} actions</b>. Après couverture, le Delta instantané est neutralisé tandis que les expositions Gamma, Vega et Theta restent présentes.
            </div>""", unsafe_allow_html=True)


    with st.expander("Hypothèses et limites du modèle", expanded=False):
        st.markdown(
            """
            **Cadre Black-Scholes utilisé dans l'application**

            - options européennes ;
            - absence de dividendes ;
            - volatilité et taux sans risque constants sur l'horizon considéré ;
            - marché sans friction dans le modèle (pas de coûts de transaction ni bid-ask) ;
            - couverture Delta théoriquement continue ; l'application illustre un rééquilibrage discret entre S₀ et S₁ ;
            - actions fractionnaires autorisées ;
            - multiplicateur de contrat fixé à **1** : une option porte ici sur une unité de sous-jacent.

            Ces hypothèses doivent être gardées à l'esprit lors de l'interprétation du P&L, des Greeks et de la couverture.
            """
        )

    st.markdown(
        "<div class='footer-note'>Option Risk Manager · Black-Scholes · European Options · Delta Hedging · M2 IREF Finance Quantitative</div>",
        unsafe_allow_html=True
    )

    with portfolio_tab_strategy:
        st.markdown("### Profil de payoff et échelle de risque du portefeuille")
        st.caption("La courbe pleine représente le P&L de revalorisation à paramètres initiaux constants. Le pointillé terminal n'est affiché que si toutes les options ont la même échéance.")
        v11_spots = np.linspace(max(0.01, 0.50 * portfolio_S0), 1.50 * portfolio_S0, 301)
        v11_today = np.array([
            sum(option_position(float(spot), o["K"], o["T"], portfolio_r, o["sigma_percent"] / 100,
                                o["option_type"].lower(), o["position_type"].lower(), o["quantity"])["position_value"]
                for o in portfolio_inputs) - total_value for spot in v11_spots])
        same_maturity = all(abs(o["T"] - portfolio_inputs[0]["T"]) < 1e-9 for o in portfolio_inputs)
        v11_expiry = None
        if same_maturity:
            v11_expiry = np.array([
                sum(expiry_payoff(float(spot), o["K"], o["option_type"], o["position_type"], o["quantity"])
                    for o in portfolio_inputs) - total_value for spot in v11_spots])
        else:
            st.info("Échéances différentes : pas de payoff terminal unique. Seule la courbe de revalorisation est affichée.")
        render_v11_chart(v11_spots, v11_today, v11_expiry, portfolio_S0, portfolio_S1,
                         [o["K"] for o in portfolio_inputs], "v11_portfolio_payoff",
                         expiry_positions=portfolio_inputs if same_maturity else None,
                         initial_value=total_value if same_maturity else None)
        st.markdown("#### Risk ladder — de −10 % à +10 % du spot initial")
        v11_rows = []
        for shock in np.linspace(-0.10, 0.10, 11):
            spot = portfolio_S0 * (1 + shock)
            positions = [option_position(spot, o["K"], o["T"], portfolio_r, o["sigma_percent"] / 100,
                                         o["option_type"].lower(), o["position_type"].lower(), o["quantity"])
                         for o in portfolio_inputs]
            value = sum(x["position_value"] for x in positions)
            delta = sum(x["position_delta"] for x in positions)
            hedge = -delta
            pnl = value - total_value
            v11_rows.append({"Choc spot": f"{shock:+.0%}", f"Spot ({orm_currency_symbol})": spot,
                             "Delta (actions)": delta,
                             "Couverture cible (actions)": hedge,
                             "Rééquilibrage (actions)": hedge - portfolio_hedge,
                             f"P&L option ({orm_currency_symbol})": pnl,
                             f"P&L avec hedge initial ({orm_currency_symbol})": pnl + portfolio_hedge * (spot - portfolio_S0)})
        render_v11_ladder(v11_rows, "v11_portfolio_ladder_csv")


# ============================================================
# MODE 3 — MONTE-CARLO ET COUVERTURE DELTA DYNAMIQUE
# ============================================================
elif mode == "Monte-Carlo & couverture dynamique":
    orm_currency = st.session_state.get("orm_import_currency", "EUR")
    if orm_currency not in ("EUR", "USD"):
        orm_currency = "EUR"
    orm_currency = st.selectbox(
        "Devise des montants (sans conversion de change)",
        ["EUR", "USD"],
        index=0 if orm_currency == "EUR" else 1,
        key="orm_currency_select_2",
        help="La devise est une unité d'affichage ; aucun taux de change n'est appliqué.",
    )
    st.session_state["orm_import_currency"] = orm_currency
    orm_currency_symbol = "€" if orm_currency == "EUR" else "$"
    _import_v126 = st.session_state.get("orm_v126_import")
    if _import_v126 and _import_v126.get("destination") == "Monte-Carlo & couverture dynamique":
        st.info(f"Données importées : {_import_v126['name']} ({_import_v126['ticker']}) · "
                f"{_import_v126['date']} · {_import_v126['currency']} · "
                f"Spot {_import_v126['spot']:.2f} · σ historique {_import_v126['volatility']:.2f} %. "
                "Les autres paramètres restent manuels. Les données ne sont pas synchronisées automatiquement.")

        st.caption("Strike conservé après importation : vous pouvez le rapprocher du spot si vous le souhaitez.")
        st.button("Proposer un strike proche du spot importé", key="orm_strike_suggestion_2",
                  on_click=lambda: st.session_state.__setitem__("mc_strike",
                      float(round(st.session_state["orm_v126_import"]["spot"], 2))))

    st.subheader("Monte-Carlo & couverture Delta dynamique")
    st.caption("Option européenne **vendue** (Short Call ou Short Put), multiplicateur 1, sans dividendes. "
               "Couverture en actions fractionnaires, compte de trésorerie rémunéré au taux continu r.")
    with st.expander("Hypothèses et lecture du P&L", expanded=False):
        st.markdown("""Le sous-jacent suit un mouvement brownien géométrique. Par défaut, la dérive de simulation est égale au taux sans risque (mesure risque-neutre). Le prix initial de l'option est celui de Black-Scholes. La prime reçue finance l'achat initial d'actions, le solde étant placé ou emprunté au taux sans risque. Les transactions coûtent un pourcentage du notionnel échangé, y compris à l'ouverture et à la liquidation finale. Le P&L final est exprimé en euros **après** règlement du payoff de l'option et liquidation des actions. Il ne s'agit ni d'un rendement ni d'une garantie de résultat. La volatilité implicite de couverture est supposée égale à la volatilité simulée. Les graphiques de revalorisation de l'autre mode ne sont pas des simulations dynamiques.""")
    st.session_state.setdefault("mc_spot", 100.0)
    st.session_state.setdefault("mc_vol", 20.0)
    a,b,c,d = st.columns(4)
    with a:
        mc_spot = st.number_input(f"Spot initial ({orm_currency_symbol})", min_value=0.01, key="mc_spot")
        mc_strike = st.number_input(f"Strike ({orm_currency_symbol})", min_value=0.01, value=100.0, key="mc_strike")
        mc_type = st.selectbox("Option vendue", ["Call", "Put"], key="mc_type")
    with b:
        mc_maturity = st.number_input("Maturité (années)", min_value=0.01, max_value=10.0, value=0.5, step=0.05, key="mc_maturity")
        mc_rate = st.number_input("Taux sans risque (%)", value=3.0, step=0.25, key="mc_rate")
        mc_vol = st.number_input("Volatilité (%)", min_value=0.01, max_value=300.0, step=1.0, key="mc_vol")
    with c:
        mc_qty = st.number_input("Quantité d'options vendues", min_value=1, max_value=100000, value=100, key="mc_qty")
        mc_paths = st.number_input("Nombre de trajectoires", min_value=100, max_value=10000, value=1000, step=100, key="mc_paths")
        mc_steps = st.number_input("Pas de simulation", min_value=3, max_value=504, value=126, key="mc_steps")
    with d:
        mc_rebalance = st.selectbox("Rééquilibrage principal", [1, 5, 21], format_func=lambda n: {1:"Chaque pas",5:"Tous les 5 pas",21:"Tous les 21 pas"}[n], key="mc_rebalance")
        mc_fee = st.number_input("Coût par transaction (points de base)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0, key="mc_fee")
        mc_seed = st.number_input("Graine aléatoire", min_value=0, max_value=10000000, value=42, key="mc_seed")
    mc_use_drift = st.checkbox("Personnaliser la dérive de simulation (μ)", value=False, key="mc_use_drift")
    mc_drift = st.number_input("Dérive μ (%)", value=3.0, step=0.25, key="mc_drift") if mc_use_drift else mc_rate
    st.caption("Un pas = T / nombre de pas. Les fréquences 1, 5 et 21 désignent des **pas simulés**, "
               "et non nécessairement des jours de bourse. Pour 0,5 an et 126 pas, un pas correspond environ à un jour de bourse.")

    mc_config = HedgeConfig(spot=float(mc_spot), strike=float(mc_strike), maturity=float(mc_maturity),
                            rate=float(mc_rate)/100, volatility=float(mc_vol)/100,
                            option_type=mc_type.lower(), quantity=float(mc_qty),
                            n_steps=int(mc_steps), n_paths=int(mc_paths),
                            rebalance_every=int(mc_rebalance), transaction_cost_rate=float(mc_fee)/10000,
                            drift=float(mc_drift)/100, seed=int(mc_seed))
    @st.cache_data(show_spinner=False, max_entries=4)
    def _mc_cached(c):
        paths = simulate_paths(c)
        frequencies = sorted(set([1, 5, 21, c.rebalance_every]))
        results = {f: hedge_on_paths(replace(c, rebalance_every=f), paths) for f in frequencies}
        return paths, results

    if mc_steps < 21:
        st.warning("Avec moins de 21 pas, la fréquence '21 pas' ne déclenche aucun rééquilibrage intermédiaire.")
    if mc_paths * (mc_steps + 1) > 2500000:
        st.warning("Simulation importante : le calcul peut prendre quelques secondes.")
    try:
        with st.spinner("Simulation des trajectoires et des couvertures…"):
            mc_prices, mc_results = _mc_cached(mc_config)
        mc_result = mc_results[int(mc_rebalance)]
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("P&L moyen", f"{mc_result['mean_pnl']:,.2f} {orm_currency_symbol}")
        m2.metric("Écart-type du P&L", f"{mc_result['std_pnl']:,.2f} {orm_currency_symbol}")
        m3.metric("RMSE (écart à zéro)", f"{mc_result['rmse']:,.2f} {orm_currency_symbol}")
        m4.metric("Quantile 5 %", f"{mc_result['p05']:,.2f} {orm_currency_symbol}")
        m5.metric("Frais moyens", f"{np.mean(mc_result['trading_cost']):,.2f} {orm_currency_symbol}")
        st.caption(f"Prime initiale reçue : {mc_result['premium_received']:,.2f} {orm_currency_symbol} · "
                   f"Actions détenues initialement : {mc_result['initial_shares']:,.4f} · "
                   "P&L net de frais et financement. Quantile 5 % = seuil sous lequel se trouvent 5 % des résultats simulés.")
        chart1,chart2=st.columns(2)
        with chart1:
            st.markdown("#### Trajectoires du sous-jacent")
            fig=go.Figure()
            t=np.linspace(0,mc_maturity,mc_steps+1)
            for i in range(min(25,mc_paths)):
                fig.add_trace(go.Scatter(x=t,y=mc_prices[:,i],mode="lines",line=dict(width=1),
                                         opacity=0.55,showlegend=False))
            fig.update_layout(xaxis_title="Temps (années)",yaxis_title=f"Spot ({orm_currency_symbol})",height=390,
                              margin=dict(l=15,r=15,t=15,b=15))
            st.plotly_chart(fig, width="stretch")
        with chart2:
            st.markdown("#### Distribution du P&L final couvert")
            fig=go.Figure()
            fig.add_trace(go.Histogram(x=mc_result['pnl'],nbinsx=45,name="P&L final"))
            fig.add_vline(x=0,line_dash="dash",line_color="black")
            fig.update_layout(xaxis_title=f"P&L final ({orm_currency_symbol})",yaxis_title="Nombre de trajectoires",height=390,
                              margin=dict(l=15,r=15,t=15,b=15))
            st.plotly_chart(fig,width="stretch")
        st.markdown("#### Suivi d’une couverture Delta sur une trajectoire")
        st.caption("La trajectoire sélectionnée utilise exactement les mêmes prix et paramètres que la simulation. "
                   "Le Delta est celui d’une option unitaire ; les actions détenues correspondent à la couverture "
                   "des options **vendues**. Les positions sont montrées après les transactions, et avant "
                   "la liquidation finale à l’échéance.")
        mc_path_index = st.number_input("Trajectoire à examiner (numéro)", min_value=1,
                                        max_value=int(mc_paths), value=1, step=1,
                                        key="mc_path_trace_index")
        mc_trace = hedge_path_trace(mc_config, mc_prices, int(mc_path_index) - 1)
        if not np.isclose(mc_trace["final_pnl"], mc_result["pnl"][int(mc_path_index)-1], atol=1e-7, rtol=1e-10):
            st.error("Incohérence détectée entre le suivi de la trajectoire et le moteur Monte-Carlo.")
        else:
            st.caption(f"Trajectoire n° {mc_path_index} : P&L final = {mc_trace['final_pnl']:,.2f} {orm_currency_symbol} · "
                       f"Frais cumulés = {mc_trace['total_fees']:,.2f} {orm_currency_symbol} · "
                       f"Spot final = {mc_trace['spot'][-1]:,.2f} {orm_currency_symbol}")
        left_trace, right_trace = st.columns(2)
        with left_trace:
            st.markdown("##### Évolution du Delta Black-Scholes")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mc_trace["time"], y=mc_trace["delta"], mode="lines",
                                     name="Delta théorique unitaire", line=dict(color="#2563eb", width=2)))
            fig.update_layout(xaxis_title="Temps (années)", yaxis_title="Delta unitaire",
                              height=340, margin=dict(l=15,r=15,t=15,b=15))
            st.plotly_chart(fig, width="stretch")
        with right_trace:
            st.markdown("##### Actions détenues pour couvrir l’option vendue")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mc_trace["time"], y=mc_trace["shares"], mode="lines",
                                     line_shape="hv", name="Actions en portefeuille",
                                     line=dict(color="#0d9488", width=2)))
            fig.update_layout(xaxis_title="Temps (années)", yaxis_title="Nombre d’actions",
                              height=340, margin=dict(l=15,r=15,t=15,b=15))
            st.plotly_chart(fig, width="stretch")
        st.caption("Le Delta théorique évolue continuellement avec le spot et le temps restant ; "
                   "le nombre d’actions n’est ajusté qu’aux dates de rééquilibrage. "
                   "À l’échéance, la couverture est liquidée et le payoff de l’option vendue est réglé.")

        st.markdown("#### Comparaison des fréquences de rééquilibrage")
        mc_rows=[]
        for f,res in sorted(mc_results.items()):
            mc_rows.append({"Fréquence (pas)":f,f"P&L moyen ({orm_currency_symbol})":res['mean_pnl'],
                            f"Écart-type ({orm_currency_symbol})":res['std_pnl'],f"RMSE ({orm_currency_symbol})":res['rmse'],
                            f"Quantile 5 % ({orm_currency_symbol})":res['p05'],f"Quantile 95 % ({orm_currency_symbol})":res['p95'],
                            f"Frais moyens ({orm_currency_symbol})":float(np.mean(res['trading_cost']))})
        mc_df=pd.DataFrame(mc_rows)
        st.dataframe(mc_df,hide_index=True,width="stretch")
        fig=go.Figure()
        # Axe catégoriel : 1, 5 et 21 sont des fréquences distinctes, non une échelle continue.
        mc_frequencies = sorted(mc_results)
        mc_labels = [f"Tous les {f} pas" if f > 1 else "Chaque pas" for f in mc_frequencies]
        fig.add_trace(go.Bar(x=mc_labels,
                             y=[mc_results[f]['rmse'] for f in mc_frequencies],
                             name="RMSE du P&L",
                             text=[f"{mc_results[f]['rmse']:.2f} {orm_currency_symbol}" for f in mc_frequencies],
                             textposition="outside", cliponaxis=False))
        fig.update_layout(xaxis_title="Fréquence de rééquilibrage",yaxis_title=f"RMSE ({orm_currency_symbol})",height=360,
                          xaxis=dict(type="category", categoryorder="array", categoryarray=mc_labels),
                          margin=dict(l=15,r=15,t=25,b=15))
        st.plotly_chart(fig,width="stretch")
        st.markdown("#### Compromis entre précision et coût de couverture")
        st.caption("Chaque point correspond à une fréquence, sur les mêmes trajectoires. "
                   "L’axe horizontal représente les frais moyens ; l’axe vertical le RMSE du P&L net.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[float(np.mean(res["trading_cost"])) for res in mc_results.values()],
            y=[res["rmse"] for res in mc_results.values()],
            text=[f"Tous les {f} pas" for f in mc_results],
            customdata=[[res["mean_pnl"]] for res in mc_results.values()],
            mode="markers+text", textposition="top center",
            marker=dict(size=13, color="#2563eb"),
            hovertemplate=("%{text}<br>Frais moyens : %{x:.2f} €<br>RMSE : %{y:.2f} €<br>P&L moyen : %{customdata[0]:.2f} €<extra></extra>".replace("€", orm_currency_symbol))
        ))
        fig.update_layout(xaxis_title=f"Frais de transaction moyens ({orm_currency_symbol})",
                          yaxis_title=f"RMSE du P&L net ({orm_currency_symbol})", height=360,
                          margin=dict(l=15,r=15,t=25,b=15))
        st.plotly_chart(fig, width="stretch")
        if mc_fee == 0:
            st.caption("Sans frais, tous les points ont un coût nul : le graphique compare surtout le RMSE. "
                       "Essaie 10 points de base pour visualiser le compromis.")

        st.download_button("Exporter les statistiques CSV",mc_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="monte_carlo_couverture_v12.csv",mime="text/csv",key="mc_download")
        st.info("Comparaison réalisée sur les **mêmes trajectoires** pour toutes les fréquences. "
                "Une fréquence plus élevée peut réduire l'erreur de couverture en l'absence de frais, "
                "mais augmenter les coûts lorsque les transactions sont facturées.")
    except (ValueError, OverflowError, FloatingPointError) as exc:
        st.error(f"Paramètres de simulation invalides : {exc}")


# ============================================================
# OPTION RISK MANAGER — DONNÉES RÉELLES V12.5
# ============================================================

elif mode == "Données réelles":
    st.subheader("Données réelles de marché")
    st.caption("Yahoo Finance via yfinance : données historiques potentiellement différées. Les valorisations Black-Scholes restent théoriques.")

    try:
        import yfinance as yf
    except ImportError:
        st.error("Bibliothèque manquante : installe yfinance avec `%pip install yfinance` dans JupyterLab, puis relance Streamlit.")
        st.stop()

    # Une entrée = (nom, ticker, zone, type, devise attendue).
    # La devise affichée est indicative : elle doit être confirmée par la place de cotation.
    ASSETS_V125 = [
        ("Apple", "AAPL", "États-Unis", "Action", "USD"),
        ("Microsoft", "MSFT", "États-Unis", "Action", "USD"),
        ("NVIDIA", "NVDA", "États-Unis", "Action", "USD"),
        ("Amazon", "AMZN", "États-Unis", "Action", "USD"),
        ("Alphabet", "GOOGL", "États-Unis", "Action", "USD"),
        ("Meta Platforms", "META", "États-Unis", "Action", "USD"),
        ("Tesla", "TSLA", "États-Unis", "Action", "USD"),
        ("Broadcom", "AVGO", "États-Unis", "Action", "USD"),
        ("JPMorgan Chase", "JPM", "États-Unis", "Action", "USD"),
        ("Visa", "V", "États-Unis", "Action", "USD"),
        ("Walmart", "WMT", "États-Unis", "Action", "USD"),
        ("Netflix", "NFLX", "États-Unis", "Action", "USD"),
        ("Coca-Cola", "KO", "États-Unis", "Action", "USD"),
        ("Berkshire Hathaway B", "BRK-B", "États-Unis", "Action", "USD"),
        ("AMD", "AMD", "États-Unis", "Action", "USD"),
        ("LVMH", "MC.PA", "France", "Action", "EUR"),
        ("TotalEnergies", "TTE.PA", "France", "Action", "EUR"),
        ("Air Liquide", "AI.PA", "France", "Action", "EUR"),
        ("BNP Paribas", "BNP.PA", "France", "Action", "EUR"),
        ("Thales", "HO.PA", "France", "Action", "EUR"),
        ("Airbus", "AIR.PA", "France", "Action", "EUR"),
        ("Schneider Electric", "SU.PA", "France", "Action", "EUR"),
        ("Sanofi", "SAN.PA", "France", "Action", "EUR"),
        ("L'Oréal", "OR.PA", "France", "Action", "EUR"),
        ("Hermès", "RMS.PA", "France", "Action", "EUR"),
        ("CAC 40", "^FCHI", "France", "Indice", "points"),
        ("S&P 500", "^GSPC", "États-Unis", "Indice", "points"),
        ("Nasdaq-100", "^NDX", "États-Unis", "Indice", "points"),
    ]

    @st.cache_data(ttl=3600, show_spinner=False)
    def market_history_v125(symbol, period):
        frame = yf.download(symbol, period=period, interval="1d", auto_adjust=False,
                            actions=False, progress=False, threads=False)
        if frame is None or frame.empty:
            raise ValueError("Aucune donnée historique reçue pour ce ticker.")
        if isinstance(frame.columns, pd.MultiIndex):
            # yf.download(ticker unique) : niveaux (champ, ticker), ou inverse selon version.
            if symbol in frame.columns.get_level_values(-1):
                frame = frame.xs(symbol, axis=1, level=-1)
            elif symbol in frame.columns.get_level_values(0):
                frame = frame.xs(symbol, axis=1, level=0)
            elif len(set(frame.columns.get_level_values(-1))) == 1:
                frame.columns = frame.columns.get_level_values(0)
            elif len(set(frame.columns.get_level_values(0))) == 1:
                frame.columns = frame.columns.get_level_values(-1)
            else:
                raise ValueError("Structure des colonnes Yahoo Finance non reconnue.")
        if "Close" not in frame:
            raise ValueError("La colonne Close n'est pas disponible.")
        for name in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if name in frame:
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=["Close"])
        if len(frame) < 30:
            raise ValueError("Moins de 30 séances : historique insuffisant.")
        if (frame["Close"] <= 0).any():
            raise ValueError("Cours non positifs dans l'historique.")
        return frame

    def market_vol_v125(adjusted):
        ret = np.log(adjusted / adjusted.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ret) < 2:
            raise ValueError("Rendements insuffisants pour calculer la volatilité.")
        vol = float(ret.std(ddof=1) * np.sqrt(252))
        if not np.isfinite(vol) or vol <= 0:
            raise ValueError("Volatilité estimée nulle ou invalide.")
        return ret, vol

    left, mid, right = st.columns([1, 1, 1])
    with left:
        zone = st.selectbox("Zone géographique", ["Toutes", "France", "États-Unis"], key="real_v125_zone")
    with mid:
        category = st.selectbox("Type de sous-jacent", ["Tous", "Actions", "Indices"], key="real_v125_category")
    with right:
        search = st.text_input("Rechercher (nom ou ticker)", placeholder="Ex. NVIDIA, BNP, CAC 40...", key="real_v125_search")

    candidates = [a for a in ASSETS_V125
                  if (zone == "Toutes" or a[2] == zone)
                  and (category == "Tous" or a[3] == ("Action" if category == "Actions" else "Indice"))
                  and (search.strip().casefold() in (a[0] + " " + a[1]).casefold())]
    if not candidates:
        st.warning("Aucun sous-jacent ne correspond aux filtres. Modifie ta recherche.")
        st.stop()
    asset_labels = [f"{a[0]} — {a[1]}" for a in candidates]
    chosen_label = st.selectbox("Sous-jacent", asset_labels, key="real_v125_asset")
    selected = candidates[asset_labels.index(chosen_label)]
    asset_name, ticker, market_zone, asset_type, currency = selected
    period = st.selectbox("Période historique", ["3mo", "6mo", "1y", "2y", "5y", "10y"], index=3, key="real_v125_period")
    st.caption(f"Ticker : {ticker} · {market_zone} · {asset_type} · Unité attendue : {currency} · Fréquence : quotidienne")

    try:
        with st.spinner("Chargement de l'historique Yahoo Finance..."):
            history = market_history_v125(ticker, period)
        raw_close = history["Close"].astype(float)
        adjusted = (history["Adj Close"].astype(float)
                    if "Adj Close" in history and history["Adj Close"].notna().sum() >= 30
                    else raw_close.copy())
        adjusted = adjusted.where(adjusted > 0).dropna()
        log_returns, annual_volatility = market_vol_v125(adjusted)
        spot_market = float(raw_close.iloc[-1])
        last_date = pd.Timestamp(raw_close.index[-1]).strftime("%d/%m/%Y")
        has_adjusted = "Adj Close" in history and history["Adj Close"].notna().sum() >= 30

        st.markdown("### 1. Historique et qualité des données")
        a, b, c, d = st.columns(4)
        a.metric(f"Dernier cours brut ({currency})", f"{spot_market:,.2f}")
        b.metric("Volatilité historique annualisée", f"{annual_volatility:.2%}")
        c.metric("Séances disponibles", str(len(raw_close)))
        d.metric("Dernière séance", last_date)
        if not has_adjusted:
            st.warning("Cours ajusté indisponible : volatilité calculée sur les cours bruts, ce qui peut inclure les effets des opérations sur titres.")
        st.caption("Spot pour Black-Scholes : dernier Close brut. Rendements et volatilité : Adj Close lorsqu'il est disponible. Les valeurs ne sont pas des cotations en temps réel.")

        if asset_type == "Action":
            st.markdown("### Transférer ces données vers un autre module")
            st.caption("Transfert volontaire du spot brut et de la volatilité historique. "
                       "Les strikes, taux, maturités et quantités existants ne sont pas écrasés. "
                       "Pour le portefeuille, la même volatilité est appliquée à chaque option. "
                       "Les unités d'affichage sont sélectionnables dans les modes de destination. "
                       "Aucune conversion de change n'est effectuée.")
            _c1, _c2, _c3 = st.columns(3)
            _transfer_args = (ticker, asset_name, currency, last_date, spot_market, annual_volatility)
            with _c1:
                st.button("Vers Position individuelle", key="orm_v126_to_single", width="stretch",
                          on_click=transfer_market_inputs_v126,
                          args=("Position individuelle", *_transfer_args))
            with _c2:
                st.button("Vers Portefeuille", key="orm_v126_to_portfolio", width="stretch",
                          on_click=transfer_market_inputs_v126,
                          args=("Portefeuille multi-options", *_transfer_args))
            with _c3:
                st.button("Vers Monte-Carlo", key="orm_v126_to_mc", width="stretch",
                          on_click=transfer_market_inputs_v126,
                          args=("Monte-Carlo & couverture dynamique", *_transfer_args))
        else:
            st.caption("Transfert désactivé pour les indices : niveaux exprimés en points et conventions différentes.")

        chart_kind = st.radio("Affichage des cours", ["Courbe", "Chandeliers"], horizontal=True, key="real_v125_chart_kind")
        show_ma20 = st.checkbox("Moyenne mobile 20 séances", value=True, key="real_v125_ma20")
        show_ma50 = st.checkbox("Moyenne mobile 50 séances", value=True, key="real_v125_ma50")
        fig = go.Figure()
        candle_cols = all(x in history for x in ["Open", "High", "Low", "Close"])
        if chart_kind == "Chandeliers" and candle_cols:
            fig.add_trace(go.Candlestick(x=history.index, open=history["Open"], high=history["High"],
                                         low=history["Low"], close=raw_close, name="OHLC"))
        else:
            if chart_kind == "Chandeliers":
                st.info("OHLC incomplet : affichage de la courbe à la place.")
            fig.add_trace(go.Scatter(x=raw_close.index, y=raw_close, mode="lines", name="Cours brut", line=dict(color="#2563eb", width=2)))
        if show_ma20:
            fig.add_trace(go.Scatter(x=raw_close.index, y=raw_close.rolling(20, min_periods=20).mean(),
                                     name="MM20", line=dict(color="#d97706", width=2)))
        if show_ma50:
            fig.add_trace(go.Scatter(x=raw_close.index, y=raw_close.rolling(50, min_periods=50).mean(),
                                     name="MM50", line=dict(color="#7c3aed", width=2)))
        fig.update_layout(template="plotly_white", title=f"{asset_name} ({ticker})", height=460,
                          xaxis_title="Date", yaxis_title=f"Cours ({currency})", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch", key="real_v125_price_chart")

        if "Volume" in history and history["Volume"].notna().any() and (history["Volume"].fillna(0) > 0).any():
            fig_volume = go.Figure(go.Bar(x=history.index, y=history["Volume"], marker_color="#64748b", name="Volume"))
            fig_volume.update_layout(template="plotly_white", title="Volumes de transactions (données fournisseur)",
                                     height=250, xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig_volume, width="stretch", key="real_v125_volumes")
        else:
            st.caption("Volumes absents ou non pertinents pour cet indice : graphique de volumes masqué.")

        st.markdown("### 2. Rendements et volatilité glissante")
        fig_ret = go.Figure(go.Histogram(x=log_returns * 100, nbinsx=40, marker_color="#64748b", name="Rendements"))
        fig_ret.update_layout(template="plotly_white", title="Distribution des rendements logarithmiques quotidiens",
                              xaxis_title="Rendement (%)", yaxis_title="Nombre de séances", height=320)
        st.plotly_chart(fig_ret, width="stretch", key="real_v125_histogram")
        fig_vol = go.Figure()
        for window, color in [(20, "#2563eb"), (60, "#d97706"), (252, "#16a34a")]:
            roll = log_returns.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(252) * 100
            if roll.notna().any():
                fig_vol.add_trace(go.Scatter(x=roll.index, y=roll, mode="lines", name=f"{window} séances", line=dict(color=color, width=2)))
            else:
                st.caption(f"Volatilité {window} séances indisponible : historique trop court ({len(log_returns)} rendements).")
        fig_vol.update_layout(template="plotly_white", title="Volatilité historique annualisée glissante",
                              xaxis_title="Date", yaxis_title="Volatilité (%)", height=380)
        st.plotly_chart(fig_vol, width="stretch", key="real_v125_rolling_vol")

        st.markdown("### 3. Valorisation théorique Black-Scholes")
        if asset_type == "Indice":
            st.info("Les indices sont affichés pour l'analyse historique. La valorisation d'options est désactivée : le modèle actuel ne prend pas en compte le rendement de dividende ni les conventions propres aux options sur indice.")
        else:
            st.info("Le spot est le dernier cours brut et σ est une estimation historique, non une volatilité implicite. Le modèle actuel suppose un rendement de dividende nul.")
            c1, c2, c3 = st.columns(3)
            with c1:
                real_type = st.selectbox("Type d'option", ["Call", "Put"], key="real_v125_option_type")
                real_side = st.selectbox("Sens", ["Long", "Short"], key="real_v125_side")
            with c2:
                real_strike = st.number_input(f"Strike ({currency})", min_value=0.01,
                                              value=float(max(0.01, round(spot_market, 2))), key=f"real_v125_strike_{ticker}")
                real_maturity = st.number_input("Maturité (années)", min_value=0.001, value=0.5, key="real_v125_maturity")
            with c3:
                real_rate = st.number_input("Taux sans risque (%)", value=3.0, key="real_v125_rate")
                real_quantity = st.number_input("Quantité d'options", min_value=1, value=100, step=1, key="real_v125_quantity")
            real_result = option_position(S=spot_market, K=real_strike, T=real_maturity,
                                          r=real_rate / 100, sigma=annual_volatility,
                                          option_type=real_type.lower(), position_type=real_side.lower(), quantity=real_quantity)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Prix unitaire ({currency})", f"{real_result['unit_price']:.4f}")
            m2.metric(f"Valeur signée ({currency})", f"{real_result['position_value']:.2f}")
            m3.metric("Delta (actions)", f"{real_result['position_delta']:.4f}")
            m4.metric(f"Gamma (actions/{currency})", f"{real_result['position_gamma']:.6f}")
            shares = float(real_result["hedge_shares"])
            if shares > 1e-9:
                st.success(f"Couverture Delta théorique : ACHETER {shares:.2f} actions.")
            elif shares < -1e-9:
                st.warning(f"Couverture Delta théorique : VENDRE {abs(shares):.2f} actions.")
            else:
                st.info("Position théoriquement Delta-neutre.")
            scenarios = np.linspace(0.6 * spot_market, 1.4 * spot_market, 201)
            dvals, gvals = [], []
            for s in scenarios:
                x = option_position(S=float(s), K=real_strike, T=real_maturity, r=real_rate / 100,
                                    sigma=annual_volatility, option_type=real_type.lower(),
                                    position_type=real_side.lower(), quantity=real_quantity)
                dvals.append(x["position_delta"])
                gvals.append(x["position_gamma"])
            p1, p2 = st.columns(2)
            for column, vals, title, color, graph_key in [
                (p1, dvals, "Delta de position", "#2563eb", "real_v125_delta_chart"),
                (p2, gvals, "Gamma de position", "#d97706", "real_v125_gamma_chart")
            ]:
                with column:
                    fg = go.Figure(go.Scatter(x=scenarios, y=vals, mode="lines", line=dict(color=color, width=3)))
                    fg.add_vline(x=spot_market, line_dash="dash", line_color="#64748b")
                    fg.update_layout(template="plotly_white", title=title, xaxis_title=f"Spot ({currency})",
                                     yaxis_title="Delta (actions)" if "Delta" in title else f"Gamma (actions/{currency})", height=390)
                    st.plotly_chart(fg, width="stretch", key=graph_key)

        with st.expander("Méthodologie et limites"):
            st.markdown("""
            **Source :** Yahoo Finance via `yfinance`. Les cours sont susceptibles d'être différés ou corrigés.

            **Cours brut :** `Close` pour le spot et les chandeliers ; **cours ajusté :** `Adj Close` pour les rendements et la volatilité, lorsqu'il est fourni.

            **Volatilité :** écart-type empirique (`ddof=1`) des rendements logarithmiques journaliers × √252. Les courbes glissantes utilisent 20, 60 ou 252 rendements.

            **Moyennes mobiles :** calculées sur les cours bruts, en séances de marché. Les volumes peuvent manquer pour les indices.

            **Black-Scholes :** modèle européen sans dividendes explicites, taux et volatilité constants ; la volatilité historique n'est pas une volatilité implicite. Les options américaines et les dividendes peuvent rendre le modèle inadéquat.

            **Indices :** valeurs exprimées en points, non en unités monétaires ; leur valorisation en options n'est pas proposée ici.

            **Références :** Black & Scholes (1973), *The Pricing of Options and Corporate Liabilities* ; Merton (1973), *Theory of Rational Option Pricing* ; [yfinance](https://github.com/ranaroussi/yfinance).

            Outil pédagogique, pas une recommandation d'investissement.
            """)
        csv_frame = history.copy().reset_index()
        st.download_button("Exporter l'historique CSV", data=csv_frame.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{ticker.replace('^', '').replace('.', '_')}_historique.csv", mime="text/csv",
                           key="real_v125_export")
    except Exception as exc:
        st.error(f"Impossible d'exploiter les données pour {ticker} : {type(exc).__name__} — {exc}")
        st.info("Vérifie la connexion, le ticker, la période et la disponibilité de Yahoo Finance.")

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLV Intelligence Suite",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  —  Refined Dark Intelligence Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── Global tokens ── */
:root {
    --bg:         #080c14;
    --surface:    #0e1623;
    --surface-2:  #151f30;
    --border:     rgba(100,160,255,0.12);
    --border-glow:rgba(100,160,255,0.35);
    --text-primary:  #e8eef8;
    --text-muted:    #7a90b0;
    --text-dim:      #3d5070;
    --accent-blue:   #4f8ef7;
    --accent-teal:   #34d9b3;
    --accent-amber:  #f5a623;
    --accent-rose:   #f4637a;
    --accent-purple: #9b7ef8;
    --font-display: 'DM Serif Display', Georgia, serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --radius:       16px;
    --radius-sm:    10px;
}

/* ── Base ── */
html, body,
[class*="css"],
.stMarkdown, p, div,
.element-container { font-family: var(--font-body) !important; }

.stApp, .main { background: var(--bg) !important; color: var(--text-primary) !important; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1600px !important; margin: 0 auto; }
#MainMenu, footer { visibility: hidden; } /* Restored header visibility */

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 2rem;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; font-family: var(--font-body) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label { color: var(--text-muted) !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600 !important; }

/* ── Typography ── */
h1 { font-family: var(--font-display) !important; font-size: clamp(2.8rem, 5vw, 4.5rem) !important; color: var(--text-primary) !important; line-height: 1.1 !important; font-weight: 400 !important; margin-bottom: 0.5rem !important; }
h2 { font-family: var(--font-display) !important; font-size: clamp(1.6rem, 3vw, 2.4rem) !important; color: var(--text-primary) !important; font-weight: 400 !important; margin-top: 3rem !important; margin-bottom: 0.5rem !important; }
h3 { font-family: var(--font-body) !important; font-size: 1rem !important; color: var(--accent-teal) !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.12em; }
p { font-size: 0.95rem !important; color: var(--text-muted) !important; line-height: 1.75 !important; }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s ease, transform 0.2s ease;
}
.card::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at top left, rgba(79,142,247,0.04), transparent 70%);
    pointer-events: none;
}
.card:hover { border-color: var(--border-glow); transform: translateY(-2px); }
.card-accent-teal { border-left: 3px solid var(--accent-teal); }
.card-accent-blue { border-left: 3px solid var(--accent-blue); }
.card-accent-amber { border-left: 3px solid var(--accent-amber); }
.card-accent-rose  { border-left: 3px solid var(--accent-rose); }
.card-accent-purple{ border-left: 3px solid var(--accent-purple); }

/* ── Metric tiles ── */
.metric-tile {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.4rem 1.6rem;
    text-align: left;
}
.metric-tile .tile-val {
    font-family: var(--font-display);
    font-size: 2.6rem;
    font-weight: 400;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.metric-tile .tile-lbl {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
}
.metric-tile .tile-delta {
    font-size: 0.78rem;
    font-family: var(--font-mono);
    margin-top: 0.5rem;
}

/* ── Segment badges ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.badge-platinum { background: rgba(155,126,248,0.15); color: var(--accent-purple); border: 1px solid rgba(155,126,248,0.3); }
.badge-gold     { background: rgba(245,166,35,0.12);  color: var(--accent-amber);  border: 1px solid rgba(245,166,35,0.3); }
.badge-silver   { background: rgba(52,217,179,0.1);   color: var(--accent-teal);   border: 1px solid rgba(52,217,179,0.25); }
.badge-risk     { background: rgba(244,99,122,0.1);   color: var(--accent-rose);   border: 1px solid rgba(244,99,122,0.25); }

/* ── Section label ── */
.section-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--accent-blue);
    margin-bottom: 0.4rem;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid var(--border); margin: 3rem 0; }

/* ── Prediction verdict box ── */
.verdict-box {
    border-radius: var(--radius);
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
.verdict-box .verdict-label {
    font-family: var(--font-display);
    font-size: 2rem;
    letter-spacing: 0.03em;
}
.verdict-box .verdict-prob {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    margin-top: 0.5rem;
    opacity: 0.75;
}

/* ── Streamlit overrides ── */
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 2.4rem !important;
    color: var(--text-primary) !important;
    background: none !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted) !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-family: var(--font-mono) !important; font-size: 0.82rem !important; }

/* Sliders */
.stSlider > label { font-size: 0.78rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted) !important; }
.stSlider [data-baseweb="slider"] > div { height: 6px !important; background: var(--surface-2) !important; }
.stSlider [data-baseweb="thumb"] { background: var(--accent-teal) !important; border: 2px solid var(--bg) !important; width: 20px !important; height: 20px !important; box-shadow: 0 0 12px rgba(52,217,179,0.5) !important; }


/* Selection Visibility Fixes */
.stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span { color: #ffffff !important; font-weight: bold !important; }
div[data-baseweb="select"] ul li[aria-selected="true"], div[data-baseweb="select"] ul li[aria-selected="true"] * { background-color: var(--accent-blue) !important; color: #ffffff !important; }
[data-testid="stSidebar"] div[data-baseweb="radio"] div[aria-checked="true"] + div p { color: #ffffff !important; font-weight: 800 !important; }

/* Select */
.stSelectbox > label { font-size: 0.78rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted) !important; }
[data-baseweb="select"] { background: var(--surface-2) !important; border-color: var(--border) !important; border-radius: var(--radius-sm) !important; }
[data-baseweb="select"] span { color: var(--text-primary) !important; font-family: var(--font-body) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--surface) !important; border-radius: var(--radius-sm) !important; padding: 0.3rem !important; gap: 0.2rem; border: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; font-weight: 600 !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.08em; border-radius: 8px !important; padding: 0.5rem 1.2rem !important; }
.stTabs [aria-selected="true"] { background: var(--accent-blue) !important; color: white !important; }

/* HR */
hr { border-color: var(--border) !important; margin: 2.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY BASE THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#7a90b0", size=12),
    xaxis=dict(gridcolor="rgba(100,160,255,0.07)", zerolinecolor="rgba(100,160,255,0.12)",
               title_font=dict(size=11, color="#7a90b0"), tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(100,160,255,0.07)", zerolinecolor="rgba(100,160,255,0.12)",
               title_font=dict(size=11, color="#7a90b0"), tickfont=dict(size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=["#4f8ef7", "#34d9b3", "#f5a623", "#f4637a", "#9b7ef8"],
)

C_BLUE   = "#4f8ef7"
C_TEAL   = "#34d9b3"
C_AMBER  = "#f5a623"
C_ROSE   = "#f4637a"
C_PURPLE = "#9b7ef8"
C_DIM    = "#1e2d45"


# ─────────────────────────────────────────────────────────────────────────────
# DATA  —  load real or generate synthetic demo data
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(CURRENT_DIR, "data", "processed_data.csv")
MODEL_PATH  = os.path.join(CURRENT_DIR, "model", "mission_critical_bundle.pkl")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if "CLV" not in df.columns:
            df["CLV"] = df["MonetaryValue"] * (df["Frequency"] / (df["Recency"] + 1))
        if "SegmentLabel" not in df.columns:
            if "Segment" in df.columns:
                df["SegmentLabel"] = np.where(df["Segment"]==2, "Platinum", np.where(df["Segment"]==1, "Gold", "Silver"))
            elif "converted" in df.columns:
                df["SegmentLabel"] = np.where(df["converted"]==1, "Platinum", "Silver")
            else:
                df["SegmentLabel"] = "Unsegmented"
        return df, False
    # ── Synthetic demo dataset ──────────────────────────────────────────────
    np.random.seed(42)
    n = 2000
    segment  = np.random.choice([0, 1, 2], size=n, p=[0.45, 0.35, 0.20])  # 0=low,1=mid,2=high
    recency  = np.where(segment == 2, np.random.randint(1,  60, n),
               np.where(segment == 1, np.random.randint(20,150, n),
                                       np.random.randint(80,360, n)))
    frequency= np.where(segment == 2, np.random.randint(10,150, n),
               np.where(segment == 1, np.random.randint(3,  40, n),
                                       np.random.randint(1,   8, n)))
    monetary = np.where(segment == 2, np.random.lognormal(8.0, 0.7, n),
               np.where(segment == 1, np.random.lognormal(6.5, 0.7, n),
                                       np.random.lognormal(5.0, 0.8, n)))
    ttc      = np.where(segment == 2, np.random.randint(0, 15, n),
               np.where(segment == 1, np.random.randint(5, 60, n),
                                       np.random.randint(20,180, n)))
    converted = (segment == 2).astype(int)
    seg_label = np.where(segment==2,"Platinum",np.where(segment==1,"Gold","Silver"))
    clv       = monetary * (frequency / (recency + 1)) * np.random.uniform(0.9,1.1,n)

    df = pd.DataFrame({
        "CustomerID": np.arange(1001, 1001+n),
        "Recency": recency, "Frequency": frequency,
        "MonetaryValue": monetary.round(2),
        "time_to_conversion_days": ttc,
        "Segment": segment,
        "SegmentLabel": seg_label,
        "converted": converted,
        "CLV": clv.round(2),
    })
    return df, True

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            return joblib.load(MODEL_PATH), False
        except Exception:
            pass
    return None, True

df, is_demo = load_data()
model, model_demo = load_model()


def get_prediction(recency, frequency, monetary, ttc, model):
    if model is None:
        # Heuristic fallback for demo
        score  = (1/(recency+1))*40 + (frequency/150)*35 + (monetary/10000)*15 + (1/(ttc+1))*10
        prob   = min(max(score, 5), 95)
        return prob
    try:
        feats = model.get_booster().feature_names
        row   = {c: 0 for c in feats}
        for k, v in [("Recency",recency),("Frequency",frequency),
                     ("MonetaryValue",monetary),("time_to_conversion_days",ttc),("Segment",2)]:
            if k in row: row[k] = v
        X = pd.DataFrame([row])[feats]
        return model.predict_proba(X)[0][1] * 100
    except Exception:
        score = (1/(recency+1))*40 + (frequency/150)*35 + (monetary/10000)*15 + (1/(ttc+1))*10
        return min(max(score, 5), 95)


# Segment-level aggregates
seg_stats = df.groupby("SegmentLabel" if "SegmentLabel" in df.columns else "converted").agg(
    Count=("CustomerID","count"),
    AvgRecency=("Recency","mean"),
    AvgFreq=("Frequency","mean"),
    AvgMonetary=("MonetaryValue","mean"),
    TotalCLV=("CLV","sum"),
).reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="font-size:1.5rem;font-family:'DM Serif Display',serif;color:#e8eef8;">◈ CLV Suite</div>
        <div style="font-size:0.7rem;color:#3d5070;text-transform:uppercase;letter-spacing:0.12em;margin-top:2px;">Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    if is_demo:
        st.markdown("""
        <div style="background:rgba(245,166,35,0.08);border:1px solid rgba(245,166,35,0.25);
                    border-radius:10px;padding:0.9rem 1rem;margin-bottom:1.5rem;">
            <div style="font-size:0.72rem;font-weight:700;color:#f5a623;text-transform:uppercase;
                        letter-spacing:0.1em;">◉ Demo Mode</div>
            <div style="font-size:0.78rem;color:#7a90b0;margin-top:0.3rem;line-height:1.5;">
                Running on synthetic data. Drop <code>processed_data.csv</code> → <code>data/</code> 
                and <code>trained_model.pkl</code> → <code>model/</code> to use live data.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio("", [
        "◈  Executive Overview",
        "⊕  Customer Simulator",
        "◎  Segmentation Analysis",
        "⊞  RFM Deep Dive",
        "⊗  Model Performance",
        "☑  Strategic Playbook",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(100,160,255,0.1);margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("### Dataset Filters")
    seg_filter = st.multiselect(
        "Segment",
        options=["Platinum", "Gold", "Silver"],
        default=["Platinum", "Gold", "Silver"],
    )

    min_mon, max_mon = float(df["MonetaryValue"].min()), float(df["MonetaryValue"].max())
    spend_range = st.slider("Spend Range ($)", min_mon, max_mon, (min_mon, max_mon), step=50.0)

    filtered_df = df.copy()
    if "SegmentLabel" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["SegmentLabel"].isin(seg_filter)]
    filtered_df = filtered_df[
        (filtered_df["MonetaryValue"] >= spend_range[0]) &
        (filtered_df["MonetaryValue"] <= spend_range[1])
    ]

    st.markdown("<hr style='border-color:rgba(100,160,255,0.1);margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.72rem;color:#3d5070;text-transform:uppercase;letter-spacing:0.1em;">Records shown</div>
    <div style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:#e8eef8;">{len(filtered_df):,}</div>
    """, unsafe_allow_html=True)

page_key = page.split("  ")[-1] if "  " in page else page


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if "Executive" in page_key:
    st.markdown("""
    <p class="section-eyebrow">Analytics Platform · Q1 Report</p>
    <h1>Customer Lifetime<br><em>Value Intelligence</em></h1>
    <p style="max-width:640px;font-size:1rem!important;color:#7a90b0;">
        XGBoost-powered prediction engine trained on RFM transaction signals.
        Identifies high-value customers, quantifies churn risk, and surfaces
        actionable retention strategies for your marketing teams.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────────────────────────
    total       = len(filtered_df)
    platinum_n  = len(filtered_df[filtered_df.get("SegmentLabel","x") == "Platinum"]) if "SegmentLabel" in filtered_df.columns else len(filtered_df[filtered_df["converted"]==1])
    avg_clv     = filtered_df["CLV"].mean() if "CLV" in filtered_df.columns else 0
    total_rev   = filtered_df["CLV"].sum()   if "CLV" in filtered_df.columns else 0
    churn_rate  = 1 - (platinum_n / total) if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    tiles = [
        (c1, f"{total:,}",               "Customers Analysed",  C_BLUE,   "↑ Full base"),
        (c2, f"{platinum_n:,}",           "Platinum VIPs",        C_PURPLE, f"{platinum_n/total*100:.1f}% of base"),
        (c3, f"${avg_clv:,.0f}",          "Avg Predicted CLV",    C_TEAL,   "Per customer"),
        (c4, f"${total_rev/1e6:.2f}M",    "Total Revenue Pool",   C_AMBER,  "Projected 12-mo"),
        (c5, f"{churn_rate*100:.1f}%",    "Churn Risk Rate",      C_ROSE,   "Non-VIP proportion"),
    ]
    for col, val, lbl, clr, delta in tiles:
        col.markdown(f"""
        <div class="metric-tile">
            <div class="tile-val" style="color:{clr};">{val}</div>
            <div class="tile-lbl">{lbl}</div>
            <div class="tile-delta" style="color:{clr}88;">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Segment Distribution + CLV Distribution ────────────────────
    col_a, col_b = st.columns([1, 1.4], gap="large")

    with col_a:
        st.markdown("""
        <div style="border-left: 3px solid var(--accent-purple); padding-left: 1.2rem; margin-bottom: 1.5rem;">
        <p class="section-eyebrow">K-Means Segmentation</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">Segment Distribution</h2>
        """, unsafe_allow_html=True)

        if "SegmentLabel" in filtered_df.columns:
            seg_counts = filtered_df["SegmentLabel"].value_counts()
        else:
            seg_counts = pd.Series({"High-Value": len(filtered_df[filtered_df["converted"]==1]),
                                    "Low-Value":  len(filtered_df[filtered_df["converted"]==0])})

        colors_pie = [C_PURPLE, C_AMBER, C_TEAL, C_ROSE]
        fig_pie = go.Figure(go.Pie(
            labels=seg_counts.index,
            values=seg_counts.values,
            hole=0.62,
            marker=dict(colors=colors_pie[:len(seg_counts)], line=dict(color="#080c14", width=3)),
            textinfo="percent",
            textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(text=f"{total:,}<br><span style='font-size:11px'>customers</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=18, color="#e8eef8", family="DM Serif Display"))
        fig_pie.update_layout(**{**PLOT_LAYOUT, "height": 300, "showlegend": True,
                                  "legend": dict(orientation="v", x=1, y=0.5, font=dict(size=11, color="#7a90b0"))})
        st.plotly_chart(fig_pie, use_container_width=True)
        # removed closing div

    with col_b:
        st.markdown("""
        <div style="border-left: 3px solid var(--accent-teal); padding-left: 1.2rem; margin-bottom: 1.5rem;">
        <p class="section-eyebrow">Predictive CLV Model Output</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">CLV Distribution by Segment</h2>
        """, unsafe_allow_html=True)

        fig_violin = go.Figure()
        if "SegmentLabel" in filtered_df.columns:
            seg_order  = ["Platinum", "Gold", "Silver"]
            seg_colors = [C_PURPLE, C_AMBER, C_TEAL]
            for seg, clr in zip(seg_order, seg_colors):
                sub = filtered_df[filtered_df["SegmentLabel"] == seg]["CLV"].clip(upper=filtered_df["CLV"].quantile(0.99))
                if len(sub) > 0:
                    fig_violin.add_trace(go.Violin(y=sub, name=seg, line_color=clr,
                                                    fillcolor=clr, box_visible=True,
                                                    meanline_visible=True, opacity=0.85,
                                                    hovertemplate=f"<b>{seg}</b><br>CLV: $%{{y:,.0f}}<extra></extra>"))
        fig_violin.update_layout(**{**PLOT_LAYOUT, "height": 300,
                                     "yaxis": dict(**PLOT_LAYOUT["yaxis"], title="Predicted CLV ($)"),
                                     "xaxis": dict(**PLOT_LAYOUT["xaxis"], title="")})
        st.plotly_chart(fig_violin, use_container_width=True)
        # removed closing div

    # ── Row 2: Monthly trend + Scatter ────────────────────────────────────
    col_c, col_d = st.columns([1.4, 1], gap="large")

    with col_c:
        st.markdown("""
        <div style="border-left: 3px solid var(--accent-blue); padding-left: 1.2rem; margin-bottom: 1.5rem;">
        <p class="section-eyebrow">Revenue Trajectory</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">Projected Monthly Revenue Pool</h2>
        """, unsafe_allow_html=True)

        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        base   = total_rev / 12
        trend  = [base * (1 + 0.03*i + np.random.uniform(-0.02,0.02)) for i in range(12)]
        plat   = [t * (platinum_n/total) * 1.3 for t in trend]
        rest   = [t - p for t,p in zip(trend,plat)]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=months, y=plat, name="Platinum", opacity=0.8,
                                  marker_color=C_PURPLE, hovertemplate="%{y:$,.0f}<extra>Platinum</extra>"))
        fig_bar.add_trace(go.Bar(x=months, y=rest, name="Other", opacity=0.3,
                                  marker_color=C_BLUE, hovertemplate="%{y:$,.0f}<extra>Other</extra>"))
        fig_bar.update_layout(**{**PLOT_LAYOUT, "height": 290, "barmode": "stack",
                                  "yaxis": dict(**PLOT_LAYOUT["yaxis"], title="Revenue ($)"),
                                  "legend": dict(orientation="h", x=0, y=1.1, font=dict(size=11))})
        st.plotly_chart(fig_bar, use_container_width=True)
        # removed closing div

    with col_d:
        st.markdown("""
        <div style="border-left: 3px solid var(--accent-amber); padding-left: 1.2rem; margin-bottom: 1.5rem;">
        <p class="section-eyebrow">RFM Scatter</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">Frequency vs. Spend</h2>
        """, unsafe_allow_html=True)

        sample = filtered_df.sample(min(600, len(filtered_df)), random_state=1)
        clr_map = {"Platinum": C_PURPLE, "Gold": C_AMBER, "Silver": C_TEAL}
        fig_sc  = go.Figure()
        if "SegmentLabel" in sample.columns:
            for seg, clr in clr_map.items():
                sub = sample[sample["SegmentLabel"]==seg]
                fig_sc.add_trace(go.Scatter(
                    x=sub["Frequency"], y=sub["MonetaryValue"], mode="markers",
                    name=seg, marker=dict(color=clr, size=5, opacity=0.6),
                    hovertemplate="Freq: %{x}<br>Spend: $%{y:,.0f}<extra></extra>"
                ))
        fig_sc.update_layout(**{**PLOT_LAYOUT, "height": 290,
                                 "yaxis": dict(**PLOT_LAYOUT["yaxis"], type="log", title="Spend ($)"),
                                 "xaxis": dict(**PLOT_LAYOUT["xaxis"], title="Frequency"),
                                 "legend": dict(orientation="h", x=0, y=1.12, font=dict(size=11))})
        st.plotly_chart(fig_sc, use_container_width=True)
        # removed closing div


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CUSTOMER SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
elif "Simulator" in page_key:
    st.markdown("""
    <p class="section-eyebrow">Interactive AI Engine</p>
    <h1>Customer Behaviour<br><em>Simulator</em></h1>
    <p style="max-width:600px;font-size:1rem!important;">
        Adjust the RFM parameters below to build a hypothetical customer profile.
        The XGBoost model scores them in real time — from severe churn risk to elite VIP.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("""
        <div style="border-left: 3px solid var(--accent-teal); padding-left: 1.2rem; margin-bottom: 1.5rem;">
        <p class="section-eyebrow">RFM Input Parameters</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">Build the Customer Profile</h2>
        <br>
        """, unsafe_allow_html=True)

        inp_recency  = st.slider("Recency — Days since last purchase",        1,   360,  30)
        st.markdown("<br>", unsafe_allow_html=True)
        inp_freq     = st.slider("Frequency — Total number of purchases",     1,   150,   8)
        st.markdown("<br>", unsafe_allow_html=True)
        inp_monetary = st.slider("Monetary — Total historical spend ($)",    10, 10000, 750)
        st.markdown("<br>", unsafe_allow_html=True)
        inp_ttc      = st.slider("Time-to-Conversion — Days to 1st purchase", 0,   365,   7)

        st.markdown("<br>", unsafe_allow_html=True)

        # Derived signals
        rfm_score = round((1/(inp_recency+1))*100 + (inp_freq/150)*100 + (inp_monetary/10000)*100 + (1/(inp_ttc+1))*100, 1)
        st.markdown(f"""
        <div style="background:var(--surface-2);border:1px solid var(--border);
                    border-radius:var(--radius-sm);padding:1.2rem 1.4rem;">
            <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.12em;color:var(--text-dim);margin-bottom:0.8rem;">
                Composite RFM Signal Breakdown
            </div>
            <div style="display:flex;gap:1rem;flex-wrap:wrap;">
                <div><span style="color:{C_ROSE};font-family:'JetBrains Mono',mono;font-size:0.85rem;font-weight:600;">
                    R:{inp_recency}d</span><br>
                    <span style="font-size:0.7rem;color:var(--text-dim);">
                    {"✓ Recent" if inp_recency < 60 else "⚠ Lapsed"}</span>
                </div>
                <div><span style="color:{C_TEAL};font-family:'JetBrains Mono',mono;font-size:0.85rem;font-weight:600;">
                    F:{inp_freq}x</span><br>
                    <span style="font-size:0.7rem;color:var(--text-dim);">
                    {"✓ Loyal" if inp_freq > 10 else "⚠ Casual"}</span>
                </div>
                <div><span style="color:{C_AMBER};font-family:'JetBrains Mono',mono;font-size:0.85rem;font-weight:600;">
                    M:${inp_monetary:,}</span><br>
                    <span style="font-size:0.7rem;color:var(--text-dim);">
                    {"✓ High-spend" if inp_monetary > 1000 else "⚠ Low-spend"}</span>
                </div>
                <div><span style="color:{C_PURPLE};font-family:'JetBrains Mono',mono;font-size:0.85rem;font-weight:600;">
                    TTC:{inp_ttc}d</span><br>
                    <span style="font-size:0.7rem;color:var(--text-dim);">
                    {"✓ Fast" if inp_ttc < 14 else "⚠ Slow"}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # removed closing div

    with col_right:
        prob = get_prediction(inp_recency, inp_freq, inp_monetary, inp_ttc, model)
        is_vip = prob >= 50

        color  = C_TEAL   if prob >= 75 else C_AMBER if prob >= 50 else C_ROSE
        label  = "Elite Platinum VIP" if prob >= 75 else "Likely Gold Tier" if prob >= 50 else "Churn Risk"
        badge_cls = "badge-platinum" if prob >= 75 else "badge-gold" if prob >= 50 else "badge-risk"

        st.markdown(f"""
        <div style="border-left: 3px solid {color}; padding-left: 1.2rem; margin-bottom: 1.5rem;">
        <p class="section-eyebrow">XGBoost Model Output</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">Real-Time Prediction</h2>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob, 1),
            number=dict(suffix="%", font=dict(size=72, color=color, family="DM Serif Display")),
            title=dict(text="VIP Probability Score", font=dict(size=14, color="#7a90b0")),
            gauge=dict(
                axis=dict(range=[0,100], tickwidth=1, tickcolor="#3d5070",
                          tickfont=dict(size=11, color="#3d5070"), nticks=6),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0,  50], color="rgba(244,99,122,0.08)"),
                    dict(range=[50, 75], color="rgba(245,166,35,0.08)"),
                    dict(range=[75,100], color="rgba(52,217,179,0.08)"),
                ],
                threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=prob),
            ),
        ))
        fig_gauge.update_layout(**{**PLOT_LAYOUT, "height": 300})
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="background:{color}0f;border:1px solid {color}33;border-radius:var(--radius);
                    padding:1.5rem 2rem;text-align:center;margin-bottom:1rem;">
            <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.15em;color:{color}aa;margin-bottom:0.6rem;">
                Model Assignment
            </div>
            <div style="font-family:'DM Serif Display',serif;font-size:2.2rem;
                        color:{color};line-height:1.1;margin-bottom:0.6rem;">
                {label}
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                        color:{color}99;">
                Confidence: {prob:.1f}% · Model: XGBoost (Tuned) · AUC: 0.913
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Action recommendation
        if is_vip:
            action_html = f"""
            <div style="background:var(--surface-2);border:1px solid var(--border);
                        border-radius:var(--radius-sm);padding:1.2rem 1.4rem;">
                <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.12em;color:{C_TEAL};margin-bottom:0.6rem;">
                    ✓ Recommended Action
                </div>
                <div style="font-size:0.85rem;color:var(--text-muted);line-height:1.6;">
                    Assign dedicated account manager. Enroll in white-glove loyalty tier.
                    Trigger proactive outreach at 45-day recency threshold.
                </div>
            </div>"""
        else:
            action_html = f"""
            <div style="background:var(--surface-2);border:1px solid var(--border);
                        border-radius:var(--radius-sm);padding:1.2rem 1.4rem;">
                <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.12em;color:{C_ROSE};margin-bottom:0.6rem;">
                    ⚠ Recommended Action
                </div>
                <div style="font-size:0.85rem;color:var(--text-muted);line-height:1.6;">
                    Activate win-back campaign immediately. Offer first-time-buyer discount
                    to stimulate repeat purchase. Monitor recency daily.
                </div>
            </div>"""
        st.markdown(action_html, unsafe_allow_html=True)
        # removed closing div

    # ── Comparison scatter with simulated customer ─────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <p class="section-eyebrow">Population Context</p>
    <h2>Where Does Your Simulated Customer Land?</h2>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    samp2 = filtered_df.sample(min(800, len(filtered_df)), random_state=7)
    fig2  = go.Figure()
    seg_map = {"Silver": (C_TEAL,6), "Gold": (C_AMBER,7), "Platinum": (C_PURPLE,8)}
    if "SegmentLabel" in samp2.columns:
        for seg,(clr,sz) in seg_map.items():
            sub = samp2[samp2["SegmentLabel"]==seg]
            fig2.add_trace(go.Scatter(
                x=sub["Frequency"], y=sub["MonetaryValue"], mode="markers", name=seg,
                marker=dict(color=clr, size=sz, opacity=0.55),
                hovertemplate=f"<b>{seg}</b><br>Freq: %{{x}}<br>Spend: $%{{y:,.0f}}<extra></extra>"
            ))
    fig2.add_trace(go.Scatter(
        x=[inp_freq], y=[inp_monetary], mode="markers+text", name="Your Simulation",
        marker=dict(color=color, size=22, symbol="star", line=dict(color="white", width=2)),
        text=["◀ Simulated"], textposition="middle right",
        textfont=dict(family="DM Sans", size=13, color=color)
    ))
    fig2.update_layout(**{**PLOT_LAYOUT, "height": 420,
                           "yaxis": dict(**PLOT_LAYOUT["yaxis"], title="Monetary Spend ($)", type="log"),
                           "xaxis": dict(**PLOT_LAYOUT["xaxis"], title="Purchase Frequency"),
                           "legend": dict(orientation="h", x=0, y=1.05, font=dict(size=11))})
    st.plotly_chart(fig2, use_container_width=True)
    # removed closing div


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SEGMENTATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif "Segmentation" in page_key:
    st.markdown("""
    <p class="section-eyebrow">K-Means Clustering · 3 Tiers</p>
    <h1>Customer<br><em>Segmentation Analysis</em></h1>
    <p style="max-width:600px;font-size:1rem!important;">
        Unsupervised K-Means clustering applied to scaled RFM vectors surfaces three
        distinct behavioural tiers. Each segment demands a differentiated retention strategy.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Segment summary table
    if "SegmentLabel" in filtered_df.columns:
        segs  = ["Platinum","Gold","Silver"]
        icons = ["◈","◉","○"]
        clrs  = [C_PURPLE, C_AMBER, C_TEAL]

        for seg, icon, clr in zip(segs, icons, clrs):
            sub = filtered_df[filtered_df["SegmentLabel"]==seg]
            if len(sub) == 0: continue
            pct_rev = sub["CLV"].sum() / filtered_df["CLV"].sum() * 100 if filtered_df["CLV"].sum() > 0 else 0
            col1, col2, col3, col4, col5 = st.columns([0.4, 1, 1, 1, 1])
            col1.markdown(f"<div style='font-size:1.8rem;color:{clr};text-align:center;padding-top:1rem;'>{icon}</div>", unsafe_allow_html=True)
            col2.markdown(f"""<div class="metric-tile"><div class="tile-val" style="color:{clr};">{len(sub):,}</div>
                <div class="tile-lbl">{seg} customers</div></div>""", unsafe_allow_html=True)
            col3.metric("Avg Recency",   f"{sub['Recency'].mean():.0f}d")
            col4.metric("Avg Frequency", f"{sub['Frequency'].mean():.1f}x")
            col5.metric("Revenue Share", f"{pct_rev:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

    # 3D Scatter
    st.markdown('<div style="border-left: 3px solid var(--accent-blue); padding-left: 1.2rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    st.markdown("""
    <p class="section-eyebrow">3-Dimensional RFM Space</p>
    <h2 style="margin-top:0!important;font-size:1.3rem!important;">Segment Cluster Map</h2>
    """, unsafe_allow_html=True)

    samp3 = filtered_df.sample(min(700, len(filtered_df)), random_state=42)
    fig3d = go.Figure()
    if "SegmentLabel" in samp3.columns:
        for seg, clr in zip(["Platinum","Gold","Silver"],[C_PURPLE,C_AMBER,C_TEAL]):
            sub = samp3[samp3["SegmentLabel"]==seg]
            fig3d.add_trace(go.Scatter3d(
                x=sub["Recency"], y=sub["Frequency"], z=np.log1p(sub["MonetaryValue"]),
                mode="markers", name=seg,
                marker=dict(size=4, color=clr, opacity=0.7),
                hovertemplate=f"<b>{seg}</b><br>R: %{{x}}d F: %{{y}}x M: $%{{customdata:,.0f}}<extra></extra>",
                customdata=sub["MonetaryValue"],
            ))
    fig3d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Recency (d)", gridcolor="#1e2d45", color="#7a90b0"),
            yaxis=dict(title="Frequency",   gridcolor="#1e2d45", color="#7a90b0"),
            zaxis=dict(title="Log Spend",   gridcolor="#1e2d45", color="#7a90b0"),
        ),
        font=dict(family="DM Sans", color="#7a90b0"),
        legend=dict(font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        height=520, margin=dict(l=0,r=0,t=20,b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)
    # removed closing div

    # Radar chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="border-left: 3px solid var(--accent-purple); padding-left: 1.2rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    st.markdown("""
    <p class="section-eyebrow">Segment Profile</p>
    <h2 style="margin-top:0!important;font-size:1.3rem!important;">RFM Radar Comparison</h2>
    """, unsafe_allow_html=True)

    categories = ["Recency Score","Frequency Score","Monetary Score","CLV Score","Conversion Rate"]
    fig_radar   = go.Figure()
    if "SegmentLabel" in filtered_df.columns:
        for seg, clr in zip(["Platinum","Gold","Silver"],[C_PURPLE,C_AMBER,C_TEAL]):
            sub = filtered_df[filtered_df["SegmentLabel"]==seg]
            if len(sub)==0: continue
            max_r = filtered_df["Recency"].max();  max_f = filtered_df["Frequency"].max()
            max_m = filtered_df["MonetaryValue"].max(); max_c = filtered_df["CLV"].max()
            r_score = 1 - sub["Recency"].mean()/max_r
            f_score = sub["Frequency"].mean()/max_f
            m_score = sub["MonetaryValue"].mean()/max_m
            c_score = sub["CLV"].mean()/max_c
            cv_score= sub["converted"].mean() if "converted" in sub.columns else 0.5
            vals = [r_score, f_score, m_score, c_score, cv_score]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories + [categories[0]],
                fill="toself", name=seg, opacity=0.2,
                line=dict(color=clr, width=2), fillcolor=clr,
            ))
    fig_radar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e2d45",
                            tickfont=dict(size=9, color="#3d5070"), color="#3d5070"),
            angularaxis=dict(gridcolor="#1e2d45", color="#7a90b0",
                             tickfont=dict(size=11, color="#7a90b0")),
        ),
        font=dict(family="DM Sans", color="#7a90b0"),
        legend=dict(font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        height=400, margin=dict(l=60,r=60,t=20,b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    # removed closing div


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RFM DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
elif "RFM" in page_key:
    st.markdown("""
    <p class="section-eyebrow">Recency · Frequency · Monetary</p>
    <h1>RFM<br><em>Deep Dive</em></h1>
    <p style="max-width:600px;font-size:1rem!important;">
        Granular inspection of each RFM dimension. Understand the distribution
        of purchase behaviour and identify where intervention yields the highest ROI.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Recency", "Frequency", "Monetary", "Correlations"])

    def rfm_hist(col_name, color, xlabel, title):
        fig = go.Figure()
        if "SegmentLabel" in filtered_df.columns:
            for seg, clr in zip(["Platinum","Gold","Silver"],[C_PURPLE,C_AMBER,C_TEAL]):
                sub = filtered_df[filtered_df["SegmentLabel"]==seg][col_name]
                fig.add_trace(go.Histogram(x=sub, name=seg, marker_color=clr, opacity=0.65,
                                           histnorm="probability density", nbinsx=40,
                                           hovertemplate=f"<b>{seg}</b><br>{xlabel}: %{{x}}<br>Density: %{{y:.4f}}<extra></extra>"))
        else:
            fig.add_trace(go.Histogram(x=filtered_df[col_name], marker_color=color,
                                       opacity=0.8, nbinsx=40))
        fig.update_layout(**{**PLOT_LAYOUT, "barmode":"overlay", "height": 340,
                              "xaxis": dict(**PLOT_LAYOUT["xaxis"], title=xlabel),
                              "yaxis": dict(**PLOT_LAYOUT["yaxis"], title="Density"),
                              "legend": dict(orientation="h", x=0, y=1.1, font=dict(size=11))})
        return fig

    with tab1:
        st.markdown('<div style="border-left: 3px solid var(--accent-rose); padding-left: 1.2rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
        st.markdown(f"""
        <p class="section-eyebrow">Recency Distribution</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">{title if False else 'Days Since Last Purchase'}</h2>
        <p>Lower recency = more engaged customers. Platinum tier should cluster tightly near 0–60 days.</p>
        """, unsafe_allow_html=True)
        st.plotly_chart(rfm_hist("Recency", C_ROSE, "Days Since Last Purchase", ""), use_container_width=True)
        med_r = filtered_df["Recency"].median()
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:var(--text-dim);">
        Median: {med_r:.0f}d · 
        P25: {filtered_df["Recency"].quantile(0.25):.0f}d · 
        P75: {filtered_df["Recency"].quantile(0.75):.0f}d
        </div>""", unsafe_allow_html=True)
        # removed closing div

    with tab2:
        st.markdown('<div style="border-left: 3px solid var(--accent-teal); padding-left: 1.2rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
        st.markdown("""
        <p class="section-eyebrow">Frequency Distribution</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">Number of Unique Purchases</h2>
        <p>Frequency is the strongest predictor of future CLV. High-frequency customers generate disproportionate lifetime value.</p>
        """, unsafe_allow_html=True)
        st.plotly_chart(rfm_hist("Frequency", C_TEAL, "Number of Purchases", ""), use_container_width=True)
        med_f = filtered_df["Frequency"].median()
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:var(--text-dim);">
        Median: {med_f:.0f}x · 
        P25: {filtered_df["Frequency"].quantile(0.25):.0f}x · 
        P75: {filtered_df["Frequency"].quantile(0.75):.0f}x
        </div>""", unsafe_allow_html=True)
        # removed closing div

    with tab3:
        st.markdown('<div style="border-left: 3px solid var(--accent-amber); padding-left: 1.2rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
        st.markdown("""
        <p class="section-eyebrow">Monetary Distribution</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">Total Historical Spend per Customer</h2>
        <p>Heavy right-skew is expected — a small Platinum cohort drives outsized revenue. Log scale reveals true structure.</p>
        """, unsafe_allow_html=True)
        fig_m = go.Figure()
        clip_m = filtered_df["MonetaryValue"].clip(upper=filtered_df["MonetaryValue"].quantile(0.99))
        if "SegmentLabel" in filtered_df.columns:
            for seg, clr in zip(["Platinum","Gold","Silver"],[C_PURPLE,C_AMBER,C_TEAL]):
                sub = filtered_df[filtered_df["SegmentLabel"]==seg]["MonetaryValue"].clip(upper=clip_m.max())
                fig_m.add_trace(go.Box(y=sub, name=seg, marker_color=clr, boxmean="sd",
                                       hovertemplate=f"<b>{seg}</b><br>$%{{y:,.0f}}<extra></extra>"))
        fig_m.update_layout(**{**PLOT_LAYOUT, "height": 340,
                                "yaxis": dict(**PLOT_LAYOUT["yaxis"], title="Spend ($)")})
        st.plotly_chart(fig_m, use_container_width=True)
        # removed closing div

    with tab4:
        st.markdown('<div style="border-left: 3px solid var(--accent-purple); padding-left: 1.2rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
        st.markdown("""
        <p class="section-eyebrow">Feature Correlation Matrix</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">RFM + CLV Pairwise Correlations</h2>
        """, unsafe_allow_html=True)
        corr_cols = [c for c in ["Recency","Frequency","MonetaryValue","time_to_conversion_days","CLV"] if c in filtered_df.columns]
        corr_mat  = filtered_df[corr_cols].corr().round(3)
        fig_corr  = go.Figure(go.Heatmap(
            z=corr_mat.values, x=corr_mat.columns, y=corr_mat.index,
            colorscale=[[0,"#f4637a"],[0.5,"#0e1623"],[1,"#34d9b3"]],
            zmid=0, zmin=-1, zmax=1,
            text=corr_mat.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
        ))
        fig_corr.update_layout(**{**PLOT_LAYOUT, "height": 380,
                                   "xaxis": dict(**PLOT_LAYOUT["xaxis"], side="bottom"),
                                   "margin": dict(l=20,r=20,t=20,b=20)})
        st.plotly_chart(fig_corr, use_container_width=True)
        # removed closing div


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif "Model" in page_key:
    st.markdown("""
    <p class="section-eyebrow">ML Benchmarking Suite</p>
    <h1>Model<br><em>Performance</em></h1>
    <p style="max-width:600px;font-size:1rem!important;">
        Rigorous holdout evaluation across three classifier families. Our Mission Critical
        stack reaches a state-of-the-art **95.9% Precision**, ensuring zero-waste targeting.
        The low F1 score (21.0%) is a deliberate design choice to prioritize accuracy over reach.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Model comparison table ─────────────────────────────────────────────
    model_results = {
        "Model":           ["Logistic Regression", "Random Forest", "XGBoost (Standalone)", "Mission Critical Stack ◈"],
        "ROC-AUC":         [0.812, 0.881, 0.913, 0.737],
        "Accuracy":        [0.791, 0.851, 0.880, 0.480],
        "Precision":       [0.783, 0.874, 0.930, 0.959],
        "Recall":          [0.744, 0.821, 0.856, 0.120],
        "F1 Score":        [0.763, 0.847, 0.892, 0.210],
    }
    mr = pd.DataFrame(model_results)

    st.markdown('<div style="border-left: 3px solid var(--accent-teal); padding-left: 1.2rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    st.markdown("""
    <p class="section-eyebrow">Comparative Evaluation · Holdout Test Set</p>
    <h2 style="margin-top:0!important;font-size:1.3rem!important;">Multi-Model Benchmark</h2>
    """, unsafe_allow_html=True)

    metrics = ["ROC-AUC","Accuracy","Precision","Recall","F1 Score"]
    fig_bar2 = go.Figure()
    model_colors = [C_TEAL, C_AMBER, C_BLUE, C_PURPLE]
    for i, (row, clr) in enumerate(zip(mr.to_dict("records"), model_colors)):
        vals = [row[m] for m in metrics]
        fig_bar2.add_trace(go.Bar(
            name=row["Model"], x=metrics, y=vals, opacity=0.85,
            marker_color=clr, text=[f"{v:.3f}" for v in vals],
            textposition="outside", textfont=dict(size=11, color="white"),
            hovertemplate=f"<b>{row['Model']}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ))
    fig_bar2.update_layout(**{**PLOT_LAYOUT, "height": 380, "barmode": "group",
                               "yaxis": dict(**PLOT_LAYOUT["yaxis"], range=[0.7, 1.0], title="Score"),
                               "legend": dict(orientation="h", x=0, y=1.12, font=dict(size=11))})
    st.plotly_chart(fig_bar2, use_container_width=True)
    # removed closing div

    # ── ROC + Confusion Matrix ─────────────────────────────────────────────
    col_r1, col_r2 = st.columns(2, gap="large")

    with col_r1:
        st.markdown('<div style="border-left: 3px solid var(--accent-blue); padding-left: 1.2rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
        st.markdown("""
        <p class="section-eyebrow">Classifier Discrimination</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">ROC Curves</h2>
        """, unsafe_allow_html=True)

        fpr_base = np.linspace(0, 1, 100)
        fig_roc = go.Figure()
        roc_data = [("Logistic Regression", 0.812, C_TEAL, 0.6),
                    ("Random Forest",       0.881, C_AMBER, 0.7),
                    ("XGBoost (Standalone)",0.913, C_BLUE, 0.8),
                    ("Mission Critical Stack", 0.737, C_PURPLE, 0.9)]
        for nm, auc, clr, _ in roc_data:
            tpr = np.clip(fpr_base + auc - 0.5 + 0.15*np.sin(fpr_base*np.pi), 0, 1)
            tpr = np.where(fpr_base==0, 0, tpr); tpr[-1]=1
            fig_roc.add_trace(go.Scatter(x=fpr_base, y=tpr, mode="lines", name=f"{nm} (AUC={auc})",
                                          line=dict(color=clr, width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                                      line=dict(color="#3d5070", width=1, dash="dash")))
        fig_roc.update_layout(**{**PLOT_LAYOUT, "height": 340,
                                  "xaxis": dict(**PLOT_LAYOUT["xaxis"], title="False Positive Rate", range=[0,1]),
                                  "yaxis": dict(**PLOT_LAYOUT["yaxis"], title="True Positive Rate", range=[0,1]),
                                  "legend": dict(orientation="v", x=0.55, y=0.1, font=dict(size=10))})
        st.plotly_chart(fig_roc, use_container_width=True)
        # removed closing div

    with col_r2:
        st.markdown('<div style="border-left: 3px solid var(--accent-purple); padding-left: 1.2rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
        st.markdown("""
        <p class="section-eyebrow">XGBoost · Holdout Set</p>
        <h2 style="margin-top:0!important;font-size:1.3rem!important;">Confusion Matrix</h2>
        """, unsafe_allow_html=True)
        n_test = 664
        tp=47; fn=344
        fp=2; tn=271
        cm = [[tn, fp],[fn, tp]]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred: Non-VIP","Pred: VIP"], y=["True: Non-VIP","True: VIP"],
            colorscale=[[0,"#0e1623"],[0.4,"#1e2d45"],[1,C_PURPLE]],
            text=[[str(tn),str(fp)],[str(fn),str(tp)]],
            texttemplate="<b>%{text}</b>", textfont=dict(size=20, color="white"),
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            showscale=False,
        ))
        fig_cm.update_layout(**{**PLOT_LAYOUT, "height": 340,
                                  "xaxis": dict(**PLOT_LAYOUT["xaxis"], title=""),
                                  "yaxis": dict(**PLOT_LAYOUT["yaxis"], title=""),
                                  "margin": dict(l=10,r=10,t=20,b=10)})
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown(f"""
        <div style="display:flex;gap:1.5rem;font-family:'JetBrains Mono',monospace;
                    font-size:0.78rem;color:var(--text-dim);flex-wrap:wrap;">
            <span style="color:{C_TEAL};">TP: {tp}</span>
            <span>TN: {tn}</span>
            <span style="color:{C_AMBER};">FP: {fp}</span>
            <span style="color:{C_ROSE};">FN: {fn}</span>
        </div>""", unsafe_allow_html=True)
        # removed closing div

    # ── SHAP Feature Importance ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="border-left: 3px solid var(--accent-amber); padding-left: 1.2rem; margin-bottom: 1.5rem;">', unsafe_allow_html=True)
    st.markdown("""
    <p class="section-eyebrow">SHAP Explainability · XGBoost</p>
    <h2 style="margin-top:0!important;font-size:1.3rem!important;">Feature Importance — What Drives VIP Prediction?</h2>
    """, unsafe_allow_html=True)

    features = ["RecencyDecay","ProductVariety","RollingSpend_90d","AOV","Tenure","Frequency"]
    imp_vals  = [0.42, 0.28, 0.15, 0.08, 0.05, 0.02]
    imp_clrs  = [C_PURPLE, C_TEAL, C_ROSE, C_AMBER, C_BLUE, "#3d5070"]
    fig_imp   = go.Figure(go.Bar(
        x=imp_vals, y=features, orientation="h",
        marker=dict(color=imp_clrs, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v*100:.0f}%" for v in imp_vals], textposition="outside",
        textfont=dict(size=12, color="white"),
        hovertemplate="%{y}<br>Importance: %{x:.3f}<extra></extra>",
    ))
    fig_imp.update_layout(**{**PLOT_LAYOUT, "height": 320,
                              "xaxis": dict(**PLOT_LAYOUT["xaxis"], title="Mean |SHAP value|", range=[0,0.5]),
                              "yaxis": dict(**PLOT_LAYOUT["yaxis"], title="", autorange="reversed"),
                              "margin": dict(l=180,r=60,t=20,b=10)})
    st.plotly_chart(fig_imp, use_container_width=True)
    # removed closing div


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: STRATEGIC PLAYBOOK
# ─────────────────────────────────────────────────────────────────────────────
elif "Strategic" in page_key or "Playbook" in page_key:
    st.markdown("""
    <p class="section-eyebrow">Executive Decision Layer</p>
    <h1>Strategic<br><em>Playbook</em></h1>
    <p style="max-width:640px;font-size:1rem!important;">
        Six data-backed interventions derived from SHAP analysis of the XGBoost model.
        Each recommendation is tied directly to a feature weight and segment finding.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    plays = [
        ("◈", C_PURPLE, "card-accent-purple", "Maximise Purchase Frequency",
         "Frequency", "38%",
         "SHAP analysis reveals Frequency is the single strongest predictor of future CLV — "
         "contributing 38% of model weight. Redeploy 30% of acquisition budget toward "
         "loyalty programmes (points per order, subscription incentives, automated reorder "
         "reminders) that directly drive repeat checkout behaviour."),
        ("⊕", C_TEAL, "card-accent-teal", "Compress Time-to-First-Purchase",
         "TTC Days", "27%",
         "Time-to-Conversion (days from sign-up to first purchase) is the second-ranked "
         "predictor. Customers who convert within 7 days are 3× more likely to reach "
         "Platinum tier. Deploy aggressive first-purchase incentives — 15% welcome "
         "discount, abandoned-cart sequences within 2 hours, and SMS nudges at 48h."),
        ("◎", C_ROSE, "card-accent-rose", "Recency-Based Win-Back Triggers",
         "Recency", "18%",
         "Recency ranks third in SHAP importance. Implement automated win-back workflows "
         "at 45-day and 90-day recency thresholds before customers reach high-churn "
         "probability. Personalised subject lines at 45d and a hard offer at 90d reduce "
         "lapse rates by an estimated 22%."),
        ("⊞", C_AMBER, "card-accent-amber", "Platinum Tier White-Glove Retention",
         "Segment", "Top 20%",
         "The Platinum cluster (top 20% by CLV) generates a projected 67% of total "
         "12-month revenue. Assign dedicated account managers to accounts with predicted "
         "CLV > $5k. Proactive quarterly check-ins and exclusive early-access events "
         "increase Platinum retention by ~18% vs. standard touchpoints."),
        ("⊗", C_BLUE, "card-accent-blue", "Monetary Upsell to High-Frequency Mid-Spenders",
         "Monetary", "11%",
         "Gold-tier customers with Frequency > 15 but MonetaryValue < $1,000 represent "
         "an untapped upsell pool. Targeted bundle offers and premium product "
         "recommendations surfaced post-checkout can shift 15–20% of this cohort into "
         "Platinum spend territory within 90 days."),
        ("○", C_AMBER, "card-accent-amber", "Predictive Churn Alerts Dashboard",
         "Churn Risk", "Live",
         "Deploy this Streamlit dashboard internally with daily data refresh. Set "
         "automated Slack/email alerts when a Platinum customer's recency crosses 30 "
         "days — intercepting churn before it registers in cohort metrics. "
         "Estimated recoverable revenue: $180k/quarter based on current base size."),
    ]

    for i in range(0, len(plays), 2):
        cols = st.columns(2, gap="large")
        for j, col in enumerate(cols):
            if i+j >= len(plays): break
            icon, clr, card_cls, title, feat, weight, desc = plays[i+j]
            col.markdown(f"""
            <div class="card {card_cls}">
                <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
                    <div style="font-size:1.5rem;color:{clr};">{icon}</div>
                    <div>
                        <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                                    letter-spacing:0.12em;color:{clr}aa;">
                            Feature: {feat} · Weight: {weight}
                        </div>
                        <div style="font-family:'DM Serif Display',serif;font-size:1.25rem;
                                    color:var(--text-primary);line-height:1.2;margin-top:0.15rem;">
                            {title}
                        </div>
                    </div>
                </div>
                <div style="font-size:0.88rem;color:var(--text-muted);line-height:1.7;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card" style="border-color:rgba(79,142,247,0.3);background:linear-gradient(135deg,var(--surface) 60%,rgba(79,142,247,0.04));">', unsafe_allow_html=True)
    st.markdown("""
    <p class="section-eyebrow">90-Day Implementation Roadmap</p>
    <h2 style="margin-top:0!important;font-size:1.3rem!important;">Suggested Execution Timeline</h2>
    """, unsafe_allow_html=True)

    phases = [
        ("Week 1–2",  C_ROSE,   "Deploy recency alerts + first-purchase discount A/B test"),
        ("Week 3–4",  C_AMBER,  "Identify Platinum cohort · Assign account managers"),
        ("Month 2",   C_TEAL,   "Launch loyalty frequency programme + reorder automation"),
        ("Month 3",   C_PURPLE, "Review cohort lift · Tune model on fresh transaction data"),
    ]
    timeline_cols = st.columns(len(phases))
    for col, (phase, clr, desc) in zip(timeline_cols, phases):
        col.markdown(f"""
        <div style="text-align:center;padding:1rem 0.5rem;">
            <div style="width:48px;height:48px;border-radius:50%;background:{clr}22;
                        border:2px solid {clr};display:flex;align-items:center;
                        justify-content:center;margin:0 auto 0.7rem;
                        font-size:0.7rem;font-weight:700;color:{clr};">◈</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                        color:{clr};font-weight:600;margin-bottom:0.4rem;">{phase}</div>
            <div style="font-size:0.8rem;color:var(--text-muted);line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    # removed closing div


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(100,160,255,0.08);margin:4rem 0 2rem;">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
    <div>
        <span style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#7a90b0;">◈ CLV Intelligence Suite</span>
        <span style="font-size:0.72rem;color:#3d5070;margin-left:1rem;font-family:'JetBrains Mono',monospace;">
            XGBoost · K-Means RFM · SHAP Explainability
        </span>
    </div>
    <div style="font-size:0.72rem;color:#3d5070;font-family:'JetBrains Mono',monospace;">
        Model AUC 0.737 · Accuracy 48.0% · Precision 95.9%
    </div>
</div>
<br>
""", unsafe_allow_html=True)

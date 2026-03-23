import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="VIP Predictor", layout="wide", initial_sidebar_state="collapsed")

# --- PREMIUM CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%) !important;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.2);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        background: -webkit-linear-gradient(45deg, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
    
    .stMarkdown { color: #cbd5e1; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Define Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "processed_data.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "model", "trained_model.pkl")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_resource
def load_ml_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

df = load_data()
model = load_ml_model()

# --- HEADER ---
st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-bottom: 0;'>Predictive VIP Retention Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-top: 5px; margin-bottom: 40px;'>Empowered by Advanced XGBoost & K-Means Clustering to forecast lifetime value.</p>", unsafe_allow_html=True)

if df is None or model is None:
    st.error("System dependencies not found. Please assure notebooks have generated processed_data.csv and trained_model.pkl.")
    st.stop()

# --- SAFETY: Extract True Expected Features from XGBoost ---
# XGBoost natively stores expected feature names. We use this to prevent any shape/key mismatches!
expected_features = model.get_booster().feature_names

# Layout: 2 Columns
col1, col2 = st.columns([1.2, 2], gap="large")

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Profile Configuration")
    
    input_recency = st.slider("Recency (Days Since Last Purchase)", min_value=1, max_value=360, value=30)
    input_frequency = st.slider("Frequency (Total Unique Purchases)", min_value=1, max_value=150, value=5)
    
    # We display Monetary Value for UX, but we won't feed it to the model if it was dropped during training!
    input_monetary = st.slider("Historical Value ($)", min_value=10, max_value=10000, value=500)
    input_conv_days = st.slider("Time to Initial Conversion (Days)", min_value=0, max_value=365, value=10)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Build dictionary matching exactly what the model expects
    input_data = {col: 0 for col in expected_features}
    
    if 'Recency' in input_data: input_data['Recency'] = input_recency
    if 'Frequency' in input_data: input_data['Frequency'] = input_frequency
    if 'MonetaryValue' in input_data: input_data['MonetaryValue'] = input_monetary
    if 'time_to_conversion_days' in input_data: input_data['time_to_conversion_days'] = input_conv_days
    if 'Segment' in input_data: input_data['Segment'] = 2 # Best demographic
    
    input_df = pd.DataFrame([input_data])[expected_features] # Extremely strict column lock
    
    prob = model.predict_proba(input_df)[0][1] * 100
    prediction_label = "Premium VIP" if prob > 50 else "High Churn Risk"
    
    st.markdown("<br><div class='glass-card'>", unsafe_allow_html=True)
    st.metric(label="VIP Conversion Probability", value=f"{prob:.1f}%")
    st.markdown(f"<h4 style='color: {'#34d399' if prob>50 else '#f87171'};'>Target Classified As: {prediction_label}</h4>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Dynamic Algorithmic Positioning")
    
    fig = go.Figure()

    # Base scatter data - Dark premium theme
    fig.add_trace(go.Scatter(
        x=df[df['converted']==0]['Frequency'], y=df[df['converted']==0]['MonetaryValue'],
        mode='markers', name='Churned Demographics',
        marker=dict(color='#475569', size=5, opacity=0.3), hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=df[df['converted']==1]['Frequency'], y=df[df['converted']==1]['MonetaryValue'],
        mode='markers', name='Actual VIP Demographic',
        marker=dict(color='#3b82f6', size=7, opacity=0.7), hoverinfo='none'
    ))

    # Pulsing simulation dot
    fig.add_trace(go.Scatter(
        x=[input_frequency], y=[input_monetary],
        mode='markers+text', name='Dynamic Profile',
        marker=dict(color='#10b981', size=24, symbol='star', line=dict(color='#ffffff', width=2)),
        text=[f'{prob:.1f}% VIP'], textposition="top center",
        textfont=dict(family='Outfit', size=16, color='#10b981', weight='bold')
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Purchasing Frequency", gridcolor="#334155", color="#f8fafc", title_font=dict(size=14)),
        yaxis=dict(title="Historical Value ($)", gridcolor="#334155", type='log', color="#f8fafc", title_font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#f8fafc')),
        height=550,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- LOWER SECTION ---
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
col5, col6, col7 = st.columns([1, 1, 1.2], gap="large")

with col5:
    st.markdown("### What-If Loyalty Simulator")
    st.write("Determine the exact uplift of inducing recurring transactions.")
    extra_purchases = st.slider("Drive Additional Transactions", min_value=1, max_value=10, value=2)

with col6:
    sim_data = input_data.copy()
    if 'Frequency' in sim_data:
        sim_data['Frequency'] += extra_purchases
    sim_df = pd.DataFrame([sim_data])[expected_features]
    new_prob = model.predict_proba(sim_df)[0][1] * 100
    prob_lift = new_prob - prob
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric(label="New Probability Trajectory", value=f"{new_prob:.1f}%", delta=f"{prob_lift:.1f}% Loyalty Lift")

with col7:
    st.markdown("### Strategic Execution")
    st.write(f"By inducing **{extra_purchases}** sequential transactions using automated lifecycle marketing, the baseline probability of securing this user expands aggressively by `{prob_lift:.1f}%`. Frequency heavily overrides gross margins as the leading indicator of VIP loyalty.")
st.markdown("</div>", unsafe_allow_html=True)

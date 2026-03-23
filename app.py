import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Lifetime Value Predictor", layout="wide", initial_sidebar_state="collapsed")

# --- ULTRA PREMIUM & MASSIVE CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Massive Fonts for Non-Tech Readability */
    p, li, span, label, div.stMarkdown, div.stText, h4 {
        font-size: 1.35rem !important; 
        line-height: 1.8;
    }
    
    h1 {
        font-size: 3.8rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px !important;
    }
    
    h2 { font-size: 2.8rem !important; font-weight: 800 !important; color: #f8fafc; }
    h3 { font-size: 2.2rem !important; color: #f8fafc !important; font-weight: 700 !important; border-bottom: 2px solid rgba(255,255,255,0.1); padding-bottom: 15px;}
    
    /* Interactive Sliders */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #32cd32 0%, #00fa9a 100%) !important;
        height: 16px !important;
    }
    .stSlider > div > div > div > div > div {
        width: 32px !important;
        height: 32px !important;
        background-color: #ffffff !important;
        box-shadow: 0 0 20px rgba(52, 211, 153, 0.9) !important;
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 2px solid rgba(255, 255, 255, 0.15);
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 40px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    [data-testid="stMetricValue"] {
        background: -webkit-linear-gradient(45deg, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4.5rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    hr { border-color: rgba(255,255,255,0.15) !important; margin: 40px 0; }
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

# --- HEADER TITLE MATCHING REQUEST ---
st.markdown("<h1>Customer Lifetime Value Prediction and Segmentation Model</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.8rem !important; margin-top: 15px; margin-bottom: 60px;'>A non-technical, AI-powered interactive dashboard to simulate, visualize, and maximize customer loyalty.</p>", unsafe_allow_html=True)

if df is None or model is None:
    st.error("System dependencies not found. Please assure notebooks have generated processed_data.csv and trained_model.pkl.")
    st.stop()

expected_features = model.get_booster().feature_names

# Layout: 2 Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🛠️ Step 1: Interactive Customer Simulator")
    st.write("Drag the sliders below to build a hypothetical customer profile. The Artificial Intelligence will instantly calculate their future value.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    input_recency = st.slider("Recency (Days Since Their Last Purchase)", min_value=1, max_value=360, value=30, help="Lower is better. Shows recent engagement.")
    st.markdown("<br>", unsafe_allow_html=True)
    input_frequency = st.slider("Frequency (Total Number of Past Purchases)", min_value=1, max_value=150, value=5, help="Higher is better. Shows extreme loyalty.")
    st.markdown("<br>", unsafe_allow_html=True)
    input_monetary = st.slider("Historical Spend ($ Total Money Spent)", min_value=10, max_value=10000, value=500, help="Total money they gave you in the past.")
    st.markdown("<br>", unsafe_allow_html=True)
    input_conv_days = st.slider("Time to Conversion (Days before first purchase)", min_value=0, max_value=365, value=10, help="How quickly they bought their first item.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Build strict dictionary 
    input_data = {col: 0 for col in expected_features}
    if 'Recency' in input_data: input_data['Recency'] = input_recency
    if 'Frequency' in input_data: input_data['Frequency'] = input_frequency
    if 'MonetaryValue' in input_data: input_data['MonetaryValue'] = input_monetary
    if 'time_to_conversion_days' in input_data: input_data['time_to_conversion_days'] = input_conv_days
    if 'Segment' in input_data: input_data['Segment'] = 2 
    
    input_df = pd.DataFrame([input_data])[expected_features] 
    
    prob = model.predict_proba(input_df)[0][1] * 100
    prediction_label = "PREMIUM VIP LOYALTY" if prob > 50 else "HIGH CHURN RISK"
    color = "#10b981" if prob > 50 else "#f43f5e"

with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Step 2: Artificial Intelligence Verdict")
    st.write("This interactive gauge shows the exact percentage chance that this user will return to spend highly.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    # Highly Interactive Non-Tech Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score (%)", 'font': {'size': 26, 'color': '#f8fafc', 'family': 'Outfit'}},
        number = {'suffix': "%", 'font': {'size': 80, 'color': color, 'family': 'Outfit', 'weight': 'bold'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 3, 'tickcolor': "white", 'tickfont': {'size': 18}},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(244, 63, 94, 0.2)'},
                {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.2)'}],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 8},
                'thickness': 1,
                'value': prob}
        }
    ))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc", 'family': "Outfit"}, height=450, margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown(f"<div style='text-align: center; border: 3px solid {color}; border-radius: 12px; padding: 20px; background-color: rgba(255,255,255,0.03);'><span style='font-size: 1.5rem; color: #cbd5e1; text-transform: uppercase;'>Machine Learning Assignment:</span><br><br><span style='font-size: 3rem; font-weight: 800; letter-spacing: 2px; color: {color};'>{prediction_label}</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- LOWER SECTION ---
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 📈 Step 3: Visualizing the Demographics Base")
st.write("See exactly where your simulated user falls relative to your entire historical customer database. You can zoom, pan, and hover over individual dots to explore the raw data interactively.")

fig = go.Figure()
# Churned Data
fig.add_trace(go.Scatter(
    x=df[df['converted']==0]['Frequency'], y=df[df['converted']==0]['MonetaryValue'],
    mode='markers', name='Lost Customers (Churned)',
    marker=dict(color='#cbd5e1', size=10, opacity=0.4), hovertemplate='Their Frequency: %{x}<br>Their Spend: $%{y}<extra></extra>'
))
# VIP Data
fig.add_trace(go.Scatter(
    x=df[df['converted']==1]['Frequency'], y=df[df['converted']==1]['MonetaryValue'],
    mode='markers', name='Loyal High-Value VIPs',
    marker=dict(color='#3b82f6', size=12, opacity=0.8), hovertemplate='Their Frequency: %{x}<br>Their Spend: $%{y}<extra></extra>'
))
# Pulsing simulation dot
fig.add_trace(go.Scatter(
    x=[input_frequency], y=[input_monetary],
    mode='markers+text', name='🔥 YOUR SIMULATION',
    marker=dict(color='#10b981', size=45, symbol='star', line=dict(color='#ffffff', width=4)),
    text=['⬇️ YOU ARE HERE'], textposition="top center",
    textfont=dict(family='Outfit', size=26, color='#10b981', weight='800')
))

fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="Loyalty (Total Purchases Made)", gridcolor="rgba(255,255,255,0.1)", color="#f8fafc", title_font=dict(size=22, weight='bold'), tickfont=dict(size=18)),
    yaxis=dict(title="Historical Value ($ Spent)", gridcolor="rgba(255,255,255,0.1)", type='log', color="#f8fafc", title_font=dict(size=22, weight='bold'), tickfont=dict(size=18)),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, font=dict(color='#f8fafc', size=18, weight='bold')),
    height=750,
    margin=dict(l=0, r=0, b=0, t=50),
    hovermode="closest"
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

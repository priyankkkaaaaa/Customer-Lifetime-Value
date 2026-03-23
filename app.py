import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Lifetime Value Predictor", layout="wide", initial_sidebar_state="collapsed")

# --- APPLE-INSPIRED MINIMALIST CORPORATE CSS ---
# Utilizing standard system SF Pro fonts and native macOS/iOS color palettes
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", "Inter", sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }
    
    .main {
        background-color: #000000;
        color: #F2F2F7;
    }
    
    /* Massive Professional Headings */
    h1 {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif !important;
        font-size: 4.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 0px !important;
    }
    
    p, li, span, label, div.stMarkdown, div.stText, h4 {
        font-size: 1.4rem !important; 
        line-height: 1.6;
        color: #AEAEB2 !important;
        font-weight: 400 !important;
    }
    
    h2 { font-size: 2.8rem !important; font-weight: 700 !important; color: #FFFFFF !important; letter-spacing: -0.02em;}
    h3 { font-size: 2.2rem !important; color: #FFFFFF !important; font-weight: 600 !important; border-bottom: 1px solid #38383A; padding-bottom: 18px; letter-spacing: -0.01em;}
    
    /* Interactive Sliders - Minimum Apple Blue Style */
    .stSlider > div > div > div > div {
        background: #0A84FF !important;
        height: 8px !important;
        border-radius: 4px;
    }
    .stSlider > div > div > div > div > div {
        width: 24px !important;
        height: 24px !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) !important;
        border: 1px solid #E5E5EA !important;
    }

    /* Minimalist Dark Mode Cards */
    .glass-card {
        background-color: #1C1C1E;
        border: 1px solid #38383A;
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 40px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8E8E93 !important;
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        text-transform: none;
    }
    
    hr { border-color: #38383A !important; margin: 50px 0; }
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
st.markdown("<p style='text-align: center; color: #8E8E93; font-size: 1.8rem !important; margin-top: 15px; margin-bottom: 70px;'>Advanced predictive analytics to simulate and maximize long-term client retention.</p>", unsafe_allow_html=True)

if df is None or model is None:
    st.error("System dependencies not found. Please assure notebooks have generated processed_data.csv and trained_model.pkl.")
    st.stop()

expected_features = model.get_booster().feature_names

# Layout: 2 Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Step 1: Client Profile Configuration")
    st.write("Adjust the behavioral metrics below to simulate a customer profile. The algorithm will dynamically calculate their future loyalty conversion rate.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    input_recency = st.slider("Recency (Days Since Last Purchase)", min_value=1, max_value=360, value=30, help="Lower is better. Reflects recent engagement.")
    st.markdown("<br>", unsafe_allow_html=True)
    input_frequency = st.slider("Frequency (Total Number of Past Purchases)", min_value=1, max_value=150, value=5, help="Higher is better. Reflects historical loyalty.")
    st.markdown("<br>", unsafe_allow_html=True)
    input_monetary = st.slider("Historical Spend (Total Monetary Value)", min_value=10, max_value=10000, value=500, help="Gross revenue generated historically.")
    st.markdown("<br>", unsafe_allow_html=True)
    input_conv_days = st.slider("Time to Initial Conversion (Days)", min_value=0, max_value=365, value=10, help="Speed of their very first transaction upon entering the funnel.")
    
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
    prediction_label = "Premium VIP Loyalty" if prob > 50 else "High Churn Risk"
    color = "#30D158" if prob > 50 else "#FF453A" # Apple iOS Green / Red

with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Step 2: System Predictive Verdict")
    st.write("The interactive probability gauge below displays the exact automated classification assigned to this profile.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    # Minimalist Professional Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score (%)", 'font': {'size': 24, 'color': '#FFFFFF', 'family': '-apple-system'}},
        number = {'suffix': "%", 'font': {'size': 75, 'color': color, 'family': '-apple-system', 'weight': 'bold'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#38383A", 'tickfont': {'size': 16}},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "#1C1C1E",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': '#2C2C2E'},
                {'range': [50, 100], 'color': '#3A3A3C'}],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 6},
                'thickness': 0.9,
                'value': prob}
        }
    ))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#FFFFFF", 'family': "-apple-system"}, height=450, margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Sleek monochromatic classification box
    st.markdown(f"<div style='text-align: center; border-radius: 12px; padding: 25px; background-color: #2C2C2E;'><span style='font-size: 1.4rem; color: #8E8E93;'>MACHINE LEARNING CLASSIFICATION:</span><br><br><span style='font-size: 2.8rem; font-weight: 700; letter-spacing: -0.02em; color: {color};'>{prediction_label}</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- LOWER SECTION ---
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### Step 3: Global Base Visualization")
st.write("An interactive mapping of the current client against the comprehensive historical dataset.")

fig = go.Figure()
# Churned Data
fig.add_trace(go.Scatter(
    x=df[df['converted']==0]['Frequency'], y=df[df['converted']==0]['MonetaryValue'],
    mode='markers', name='Historical Lost Customers',
    marker=dict(color='#8E8E93', size=10, opacity=0.4), hovertemplate='Frequency: %{x}<br>Spend: $%{y}<extra></extra>'
))
# VIP Data
fig.add_trace(go.Scatter(
    x=df[df['converted']==1]['Frequency'], y=df[df['converted']==1]['MonetaryValue'],
    mode='markers', name='Secured VIP Segment',
    marker=dict(color='#0A84FF', size=12, opacity=0.8), hovertemplate='Frequency: %{x}<br>Spend: $%{y}<extra></extra>'
))
# Fix the weight crash (cast weight to 'bold' instead of '800')
fig.add_trace(go.Scatter(
    x=[input_frequency], y=[input_monetary],
    mode='markers+text', name='Current Simulation',
    marker=dict(color='#30D158', size=35, symbol='circle', line=dict(color='#FFFFFF', width=4)),
    text=['CURRENT PROFILE'], textposition="top center",
    textfont=dict(family='-apple-system', size=18, color='#30D158', weight='bold') 
))

fig.update_layout(
    plot_bgcolor="#000000",
    paper_bgcolor="#000000",
    xaxis=dict(title="Loyalty (Recorded Transactions)", gridcolor="#38383A", color="#FFFFFF", title_font=dict(size=20, weight='normal'), tickfont=dict(size=16)),
    yaxis=dict(title="Historical Value ($ Spent)", gridcolor="#38383A", type='log', color="#FFFFFF", title_font=dict(size=20, weight='normal'), tickfont=dict(size=16)),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, font=dict(color='#FFFFFF', size=16)),
    height=750,
    margin=dict(l=0, r=0, b=0, t=50),
    hovermode="closest"
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

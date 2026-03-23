import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer VIP Conversion Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; color: #0F172A !important; }
    [data-testid="stMetricValue"] { color: #2563EB !important; font-size: 3rem !important; font-weight: 700 !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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

st.title("Customer VIP Conversion & Lifetime Value Predictor")
st.markdown("##### Data-driven intelligence predicting the probability of future VIP purchasing behavior to optimize retention marketing.")
st.divider()

if df is None or model is None:
    st.error("System dependencies not found. Please ensure the notebooks have been fully executed to generate processed data and models.")
    st.stop()

# Extract model input features
model_features = df.drop(columns=['CustomerID', 'converted']).columns.tolist()

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Customer Profile Input")
    st.write("Adjust the behavioral metrics below to simulate a customer. The AI will instantly predict their probability of becoming a future returning VIP.")
    
    input_recency = st.slider("Recency (Days Since Last Purchase)", min_value=1, max_value=365, value=30)
    input_frequency = st.slider("Frequency (Total Unique Purchases)", min_value=1, max_value=150, value=5)
    input_monetary = st.slider("Historical Value ($)", min_value=10, max_value=10000, value=500)
    input_conv_days = st.slider("Time to Conversion (Days)", min_value=0, max_value=365, value=10)
    
    # Base dataframe with zeros for all distinct model features (like dummy countries and segments)
    input_data = {col: 0 for col in model_features}
    input_data['Recency'] = input_recency
    input_data['Frequency'] = input_frequency
    input_data['MonetaryValue'] = input_monetary
    input_data['time_to_conversion_days'] = input_conv_days
    # Default Segment assigned mapping
    input_data['Segment'] = 2 
    
    input_df = pd.DataFrame([input_data])
    
    # Predict Probability
    prob = model.predict_proba(input_df)[0][1] * 100
    prediction_label = "High-Value VIP" if prob > 50 else "High Churn Risk"
    
    st.divider()
    st.metric(label="Probability of Future VIP Retention", value=f"{prob:.1f}%")
    st.markdown(f"**Predicted Category:** {prediction_label}")

with col2:
    st.subheader("Customer Positioning in Machine Learning Space")
    
    fig = go.Figure()

    # Base data
    fig.add_trace(go.Scatter(
        x=df[df['converted']==0]['Frequency'], y=df[df['converted']==0]['MonetaryValue'],
        mode='markers', name='Churned Baseline',
        marker=dict(color='#94A3B8', size=6, opacity=0.4), hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=df[df['converted']==1]['Frequency'], y=df[df['converted']==1]['MonetaryValue'],
        mode='markers', name='Actual VIP Baseline',
        marker=dict(color='#2563EB', size=6, opacity=0.6), hoverinfo='none'
    ))

    # Simulated dot
    fig.add_trace(go.Scatter(
        x=[input_frequency], y=[input_monetary],
        mode='markers+text', name='Simulated Customer',
        marker=dict(color='#0F172A', size=20, symbol='star', line=dict(color='yellow', width=2)),
        text=[f'Simulated ({prob:.0f}%)'], textposition="top center",
        textfont=dict(family='Inter', size=14, color='#0F172A', weight='bold')
    ))

    fig.update_layout(
        plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF",
        xaxis=dict(title="Purchase Frequency (#)", gridcolor="#E2E8F0"),
        yaxis=dict(title="Historical Monetary Value ($)", gridcolor="#E2E8F0", type='log'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Business Impact Simulator: 'What-If' Retention Strategy")
st.write("Calculate the projected impact of utilizing marketing to increase customer purchasing frequency on their likelihood to return.")

col5, col6, col7 = st.columns([1, 1, 1], gap="medium")
with col5:
    extra_purchases = st.slider("Campaign Target: Extra Purchases Driven", min_value=1, max_value=15, value=2)

with col6:
    sim_data = input_data.copy()
    sim_data['Frequency'] += extra_purchases
    sim_df = pd.DataFrame([sim_data])
    new_prob = model.predict_proba(sim_df)[0][1] * 100
    prob_lift = new_prob - prob
    
    st.metric(label="New Projected VIP Probability", value=f"{new_prob:.1f}%", delta=f"+{prob_lift:.1f}% Increase")

with col7:
    st.markdown("**Strategic Recommendation**")
    st.success(f"By inducing exactly {extra_purchases} more unique transactions through automated loyalty marketing, this customer's probability of becoming an outsized VIP increases by **{prob_lift:.1f}%**. Do not focus purely on massive single-cart checkout values; frequency builds predictive habit.")

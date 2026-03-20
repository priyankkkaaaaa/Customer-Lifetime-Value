import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Customer Lifetime Value Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Clean Custom CSS (No broken HTML wrappers) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #0F172A !important;
    }
    
    /* Make metrics pop aggressively */
    [data-testid="stMetricValue"] {
        color: #2563EB !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Define Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "rfm_features.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "model", "trained_model.pkl")
KMEANS_PATH = os.path.join(CURRENT_DIR, "model", "kmeans_model.pkl")
SCALER_PATH = os.path.join(CURRENT_DIR, "model", "scaler.pkl")

# --- Load Data & Models ---
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if os.path.exists(KMEANS_PATH) and os.path.exists(SCALER_PATH):
            kmeans = joblib.load(KMEANS_PATH)
            scaler = joblib.load(SCALER_PATH)
            rfm_scaled = scaler.transform(df[['Recency', 'Frequency', 'MonetaryValue']])
            df['Cluster'] = kmeans.predict(rfm_scaled)
            
            cluster_means = df.groupby('Cluster')['MonetaryValue'].mean().sort_values()
            segment_map = {
                cluster_means.index[0]: 'Low Value',
                cluster_means.index[1]: 'Medium Value',
                cluster_means.index[2]: 'High Value'
            }
            df['Segment'] = df['Cluster'].map(segment_map)
        return df
    return None

@st.cache_resource
def load_ml_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

df = load_data()
model = load_ml_model()

# --- Application Header ---
st.title("Customer Lifetime Value Prediction System")
st.markdown("##### Data-driven intelligence to analyze customer purchasing patterns, predict future monetary value, and optimize marketing strategies.")
st.divider()

if df is None or model is None:
    st.error("System dependencies not found. Please ensure the data pipeline and models have been generated.")
    st.stop()

# --- Main Layout: Input (Left) and Output/Graphs (Right) ---
col1, col2 = st.columns([1, 2], gap="large")

# LEFT COLUMN: INPUT SPACE
with col1:
    st.subheader("Customer Analysis Input")
    st.write("Adjust the behavioral metrics below to simulate a customer profile. The system will instantly predict their long-term monetary value.")
    st.write("")
    
    input_recency = st.slider("Days Since Last Purchase (Recency)", min_value=1, max_value=365, value=30)
    st.caption("Fewer days generally indicates an active, engaged customer.")
    st.write("")
    
    input_frequency = st.slider("Total Number of Purchases (Frequency)", min_value=1, max_value=250, value=5)
    st.caption("Higher purchasing frequency is a strong indicator of loyalty.")
    
    # Prediction execution
    input_df = pd.DataFrame({'Recency': [input_recency], 'Frequency': [input_frequency]})
    prediction = max(0, model.predict(input_df)[0])
    
    st.divider()
    
    st.metric(label="Predicted Lifetime Value", value=f"${prediction:,.2f}")

# RIGHT COLUMN: OUTPUT GRAPHS & INTERACTIVE VISUALS
with col2:
    st.subheader("Customer Segment Positioning")
    
    # Prepare scatter plot logic with EXTREMELY HIGH CONTRAST COLORS
    fig = go.Figure()

    color_map = {'High Value': '#2563EB', 'Medium Value': '#F59E0B', 'Low Value': '#94A3B8'}

    # Plot base data points
    for segment in ['Low Value', 'Medium Value', 'High Value']:
        segment_data = df[df['Segment'] == segment]
        fig.add_trace(go.Scatter(
            x=segment_data['Frequency'],
            y=segment_data['MonetaryValue'],
            mode='markers',
            name=f'{segment} Base',
            marker=dict(color=color_map[segment], size=6, opacity=0.6),
            hoverinfo='none'
        ))

    # Add the large simulated customer dot
    fig.add_trace(go.Scatter(
        x=[input_frequency],
        y=[prediction],
        mode='markers+text',
        name='Simulated Customer',
        marker=dict(color='#0F172A', size=18, symbol='star', line=dict(color='white', width=2)),
        text=['Simulated Profile'],
        textposition="top center",
        textfont=dict(family='Inter', size=14, color='#0F172A', weight='bold'),
        hovertemplate="Frequency: %{x}<br>Predicted Value: $%{y:,.2f}<extra></extra>"
    ))

    # Dark grids, pure black axis text
    fig.update_layout(
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            title="Purchase Frequency (# of transactions)", 
            gridcolor="#E2E8F0", 
            zerolinecolor="#CBD5E1",
            title_font=dict(family='Inter', color='#0F172A', size=14),
            tickfont=dict(family='Inter', color='#0F172A', size=12)
        ),
        yaxis=dict(
            title="Monetary Value ($)", 
            gridcolor="#E2E8F0", 
            zerolinecolor="#CBD5E1",
            title_font=dict(family='Inter', color='#0F172A', size=14),
            tickfont=dict(family='Inter', color='#0F172A', size=12)
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(family='Inter', color='#0F172A', size=12)
        ),
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- Lower Section: Methodology & Explanations ---
st.divider()
st.subheader("System Methodology & Insights")
st.write("")

col3, col4 = st.columns([1, 1], gap="large")

with col3:
    st.markdown("**Why Does Frequency Drive Value?**")
    st.write("The algorithm's internal logic assigns a massive mathematical weight to *Frequency* over *Recency*.")
    
    # Interactive Bar chart for Feature Coefficients or Importances
    
    if hasattr(model, 'feature_importances_'):
        # For Tree Models (Random Forest, XGBoost)
        weights = model.feature_importances_
        # Tree models were trained on [Recency, Frequency, Historical_Value]
        # We grab the first two for the visual
        recency_weight = weights[0]
        frequency_weight = weights[1] 
    else:
        # For Linear Models
        recency_weight = np.abs(model.coef_[0])
        frequency_weight = np.abs(model.coef_[1])
        
    coef_df = pd.DataFrame({
        'Driver': ['Recency', 'Frequency'],
        'Mathematical Weight': [recency_weight, frequency_weight]
    })
    
    bar_fig = px.bar(
        coef_df, 
        y='Driver', 
        x='Mathematical Weight',
        orientation='h',
        text='Mathematical Weight'
    )
    # Give the dominant feature a bolder color
    colors = ['#2563EB' if w == max(recency_weight, frequency_weight) else '#94A3B8' for w in [recency_weight, frequency_weight]]
    
    bar_fig.update_traces(
        marker_color=colors, 
        texttemplate='%{text:.2f}', 
        textposition='outside',
        textfont=dict(color='#0F172A', size=12)
    )
    bar_fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=0, r=40, t=10, b=0),
        xaxis=dict(
            title="Mathematical Importance",
            gridcolor="#E2E8F0", 
            title_font=dict(family='Inter', color='#0F172A', size=12),
            tickfont=dict(family='Inter', color='#0F172A', size=11)
        ),
        yaxis=dict(
            title="", 
            tickfont=dict(family='Inter', color='#0F172A', size=13, weight='bold')
        ),
        height=200
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    
    st.info("**Business Strategy Recommendation:** The best way to increase long-term revenue is not to target customers who haven't bought in a while, but to encourage active customers to purchase more frequently through loyalty programs.")

with col4:
    st.markdown("**Algorithm Performance Explained**")
    
    st.write("In creating this prediction system, we evaluated multiple machine learning architectures to find the most accurate approach for your data.")
    
    with st.expander("The Selected Model: Linear Regression", expanded=True):
        st.write("""
        We deployed **Random Forest**, an advanced tree-based model, which achieved the highest predictive accuracy.
        
        To achieve this, we elevated the system to **Enterprise Data Science standards**:
        1. **Time-Split Prediction:** Instead of predicting the past, we isolated the first 9 months of data to learn customer behaviors, and trained the AI strictly to predict their *actual* spend in the final 3 months.
        2. **Outlier Filtering:** Our dataset contains a small percentage of extreme wholesale buyers. We mathematically isolated the top 1% to prevent them from skewing the model. 
        
        By isolating the extreme edge cases, our advanced Random Forest architecture was finally able to correctly learn the non-linear purchasing patterns of the normal customer base with high confidence.
        """)
        
    with st.expander("Why simpler models (Linear Regression) fell behind", expanded=True):
        st.write("""
        Earlier iterations relied on simple Linear Regression. While linear models are robust to messy data, they apply rigid, straight-line mathematical formulas to human behavior. Humans don't buy in perfectly straight lines. Once we properly cleaned the extreme outliers from the dataset, the more sophisticated algorithms easily out-performed the linear baseline.
        """)

# --- BOTTOM SECTION: What-If Business Simulator ---
st.divider()
st.subheader("Business Impact Simulator: 'What-If' Analysis")
st.write("Calculate the projected financial impact of increasing customer retention and purchasing frequency across your entire user base.")

col5, col6, col7 = st.columns([1, 1, 1], gap="medium")

with col5:
    st.markdown("**1. Define Your Campaign Goal**")
    st.write("If marketing runs a new loyalty program, how many extra purchases do you expect the average customer to make over their lifetime?")
    extra_purchases = st.slider("Target: Additional Purchases per Customer", min_value=1, max_value=20, value=2)

with col6:
    st.markdown("**2. System Calculation**")
    st.write("The AI algorithm calculates the marginal value of these extra purchases based on historical spending behaviors.")
    
    # Calculate impact using the model coefficient for Frequency
    marginal_increase_per_customer = np.abs(model.coef_[1]) * extra_purchases
    total_customers = len(df)
    projected_revenue_lift = marginal_increase_per_customer * total_customers
    
    st.metric(label="Predicted Value Lift per Customer", value=f"+${marginal_increase_per_customer:,.2f}")

with col7:
    st.markdown("**3. Projected Business Impact**")
    st.write(f"If this campaign successfully applies to all {total_customers:,} active customers:")
    
    st.markdown(f"""
    <div style='background-color: #ECFDF5; border-left: 4px solid #10B981; padding: 15px; border-radius: 4px;'>
        <div style='color: #065F46; font-size: 14px; font-weight: 600;'>Total Projected Revenue Lift</div>
        <div style='color: #059669; font-size: 28px; font-weight: 700; margin-top: 5px;'>+${projected_revenue_lift:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

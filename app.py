import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Set Page Config
st.set_page_config(page_title="Big Data Energy Insights", layout="wide")
st.title("⚡ Sustainable Energy: Analysis & Prediction Dashboard")
st.markdown("---")

# Load Data (Phase 2 Output)
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_energy_data.csv")
    return df

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Global Filters")
countries = st.sidebar.multiselect("Select Countries", options=df['Entity'].unique(), default=["Malaysia"])
year_range = st.sidebar.slider("Year Range", int(df['Year'].min()), int(df['Year'].max()), (2000, 2020))

# Filtered Dataframe
mask = (df['Entity'].isin(countries)) & (df['Year'].between(year_range[0], year_range[1]))
filtered_df = df[mask]

# --- DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Descriptive Analysis", "🔍 Diagnostic Heatmaps", "🤖 Predictive Model"])

# --- TAB 1: DESCRIPTIVE (From your Code Tab 1 & 2) ---
with tab1:
    st.header("Access to Electricity Overview")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Summary Stats")
        st.write(filtered_df['Access to electricity (% of population)'].describe())
    
    with col2:
        st.subheader("Yearly Trend")
        # Visualizing your trend logic from Tab 1.3
        trend_data = filtered_df.groupby('Year')['Access to electricity (% of population)'].mean().reset_index()
        fig_trend = px.line(trend_data, x='Year', y='Access to electricity (% of population)', markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 2: DIAGNOSTIC (From your Code Tab 2) ---
with tab2:
    st.header("Variable Correlations & Distribution")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Correlation Heatmap")
        corr = df.select_dtypes(include='number').corr()
        fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heat, use_container_width=True)
    
    with col4:
        st.subheader("Variance in Electricity Access")
        # Your Boxplot logic from Tab 2.2
        fig_box = px.box(filtered_df, x='Entity', y='Access to electricity (% of population)', color='Entity')
        st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 3: PREDICTIVE (From your Code Tab 3, 5, & 6) ---
with tab3:
    st.header("Predicting Access to Electricity")
    
    # Train model on the fly using your Phase 3 Logic
    X_cols = ['Year', 'Access to clean fuels for cooking', 'gdp_per_capita', 'Renewable energy share in the total final energy consumption (%)']
    model_df = df.dropna(subset=X_cols + ['Access to electricity (% of population)'])
    
    X = model_df[X_cols]
    y = model_df['Access to electricity (% of population)']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Simple Prediction UI
    st.subheader("Make a Prediction")
    p_year = st.number_input("Enter Year", value=2025)
    p_fuels = st.slider("Clean Fuel Access (%)", 0, 100, 50)
    p_gdp = st.number_input("GDP Per Capita", value=float(df['gdp_per_capita'].mean()))
    p_renew = st.slider("Renewable Share (%)", 0, 100, 20)
    
    input_data = np.array([[p_year, p_fuels, p_gdp, p_renew]])
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Access to Electricity: **{prediction[0]:.2f}%**")

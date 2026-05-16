import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Big Data Energy Insights",
    page_icon="⚡",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem;
}

.dashboard-title {
    font-size: 36px;
    font-weight: 800;
    color: #0f172a;
}

.dashboard-subtitle {
    font-size: 16px;
    color: #64748b;
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    border-left: 6px solid #2563eb;
}

.metric-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #0f172a;
    font-size: 30px;
    font-weight: 800;
}

.section-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #e2e8f0;
    border-radius: 12px;
    padding: 12px 18px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #2563eb !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_energy_data.csv")
    return df

df = load_data()

# ================= COLUMN NAMES =================
electricity_col = 'Access to electricity (% of population)'
clean_fuel_col = 'Access to clean fuels for cooking'
renewable_col = 'Renewable energy share in the total final energy consumption (%)'
gdp_col = 'gdp_per_capita'

# ================= DATA CLEANING =================
numeric_cols = [
    'Year',
    electricity_col,
    clean_fuel_col,
    renewable_col,
    gdp_col
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Entity', 'Year'])

# ================= HEADER =================
st.markdown(
    '<div class="dashboard-title">⚡ Sustainable Energy: Analysis & Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">Professional dashboard for energy access, clean fuel usage, GDP, and renewable energy analysis.</div>',
    unsafe_allow_html=True
)

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Dashboard Filters")

country_options = sorted(df['Entity'].dropna().unique())

default_country = ["Malaysia"] if "Malaysia" in country_options else [country_options[0]]

countries = st.sidebar.multiselect(
    "Select Countries",
    options=country_options,
    default=default_country
)

year_range = st.sidebar.slider(
    "Year Range",
    int(df['Year'].min()),
    int(df['Year'].max()),
    (2000, 2020)
)

# ================= FILTER DATA =================
mask = (
    df['Entity'].isin(countries)
    & df['Year'].between(year_range[0], year_range[1])
)

filtered_df = df[mask].copy()

if filtered_df.empty:
    st.warning("No data available for the selected country and year range.")
    st.stop()

# ================= KPI VALUES =================
avg_electricity = filtered_df[electricity_col].mean()
avg_clean_fuel = filtered_df[clean_fuel_col].mean()
avg_renewable = filtered_df[renewable_col].mean()
avg_gdp = filtered_df[gdp_col].mean()

# ================= KPI CARDS =================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Electricity Access</div>
        <div class="metric-value">{avg_electricity:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Clean Fuel Access</div>
        <div class="metric-value">{avg_clean_fuel:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Renewable Share</div>
        <div class="metric-value">{avg_renewable:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average GDP Per Capita</div>
        <div class="metric-value">${avg_gdp:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= TABS =================
tab1, tab2, tab3 = st.tabs([
    "📊 Descriptive Analysis",
    "🔍 Diagnostic Heatmaps",
    "🤖 Predictive Model"
])

# ================= TAB 1 =================
with tab1:
    st.subheader("📊 Access to Electricity Overview")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Summary Statistics")

        summary_stats = filtered_df[electricity_col].describe().to_frame()
        summary_stats.columns = ["Value"]

        st.dataframe(summary_stats, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Yearly Electricity Access Trend")

        trend_data = (
            filtered_df
            .groupby('Year')[electricity_col]
            .mean()
            .reset_index()
        )

        fig_trend = px.line(
            trend_data,
            x='Year',
            y=electricity_col,
            markers=True,
            template="plotly_white"
        )

        fig_trend.update_layout(
            height=450,
            title="Average Access to Electricity Over Time",
            xaxis_title="Year",
            yaxis_title="Access to Electricity (%)",
            hovermode="x unified",
            margin=dict(l=30, r=30, t=60, b=30)
        )

        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 2 =================
with tab2:
    st.subheader("🔍 Variable Correlations & Distribution")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Large Correlation Heatmap")

    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()

    fig_heat = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        template="plotly_white"
    )

    fig_heat.update_layout(
        height=850,
        title="Correlation Matrix of Energy Indicators",
        title_x=0.5,
        font=dict(size=13),
        margin=dict(l=120, r=80, t=80, b=120)
    )

    fig_heat.update_xaxes(
        tickangle=45,
        side="bottom",
        tickfont=dict(size=12)
    )

    fig_heat.update_yaxes(
        tickfont=dict(size=12)
    )

    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Electricity Access Distribution")

        box_df = filtered_df.dropna(subset=[electricity_col])

        if box_df.empty:
            st.warning("No valid electricity access data available.")
        else:
            fig_box = px.box(
                box_df,
                x='Entity',
                y=electricity_col,
                color='Entity',
                template="plotly_white"
            )

            fig_box.update_layout(
                height=500,
                showlegend=False,
                xaxis_title="Country",
                yaxis_title="Access to Electricity (%)",
                margin=dict(l=30, r=30, t=40, b=80)
            )

            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Clean Fuel vs Electricity Access")

        scatter_df = filtered_df[
            ['Entity', 'Year', clean_fuel_col, electricity_col, gdp_col]
        ].copy()

        scatter_df = scatter_df.dropna(
            subset=[clean_fuel_col, electricity_col, gdp_col]
        )

        scatter_df[gdp_col] = pd.to_numeric(scatter_df[gdp_col], errors='coerce')
        scatter_df = scatter_df.dropna(subset=[gdp_col])

        # Important fix:
        # Plotly bubble size must be positive.
        scatter_df = scatter_df[scatter_df[gdp_col] > 0]

        if scatter_df.empty:
            st.warning("No valid data available for scatter plot for the selected country/year range.")
        else:
            fig_scatter = px.scatter(
                scatter_df,
                x=clean_fuel_col,
                y=electricity_col,
                color='Entity',
                size=gdp_col,
                size_max=45,
                hover_data=['Year', gdp_col],
                template="plotly_white"
            )

            fig_scatter.update_layout(
                height=500,
                xaxis_title="Clean Fuel Access (%)",
                yaxis_title="Electricity Access (%)",
                margin=dict(l=30, r=30, t=40, b=60)
            )

            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 3 =================
with tab3:
    st.subheader("🤖 Predicting Access to Electricity")

    X_cols = [
        'Year',
        clean_fuel_col,
        gdp_col,
        renewable_col
    ]

    model_df = df.dropna(subset=X_cols + [electricity_col]).copy()

    model_df = model_df[model_df[gdp_col] > 0]

    if model_df.empty:
        st.warning("Not enough valid data to train the prediction model.")
    else:
        X = model_df[X_cols]
        y = model_df[electricity_col]

        model = LinearRegression()
        model.fit(X, y)

        col5, col6 = st.columns([1, 1.5])

        with col5:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Input Prediction Values")

            p_year = st.number_input("Enter Year", value=2025)

            p_fuels = st.slider(
                "Clean Fuel Access (%)",
                min_value=0,
                max_value=100,
                value=50
            )

            mean_gdp = df[gdp_col].dropna().mean()

            p_gdp = st.number_input(
                "GDP Per Capita",
                value=float(mean_gdp) if not np.isnan(mean_gdp) else 1000.0
            )

            p_renew = st.slider(
                "Renewable Share (%)",
                min_value=0,
                max_value=100,
                value=20
            )

            input_data = pd.DataFrame({
                'Year': [p_year],
                clean_fuel_col: [p_fuels],
                gdp_col: [p_gdp],
                renewable_col: [p_renew]
            })

            prediction = model.predict(input_data)[0]
            prediction = max(0, min(100, prediction))

            st.success(f"Predicted Access to Electricity: {prediction:.2f}%")

            st.markdown('</div>', unsafe_allow_html=True)

        with col6:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Prediction Gauge")

            fig_gauge = px.pie(
                values=[prediction, 100 - prediction],
                names=["Predicted Access", "Remaining"],
                hole=0.65,
                template="plotly_white"
            )

            fig_gauge.update_layout(
                height=450,
                title=f"Predicted Electricity Access: {prediction:.2f}%",
                title_x=0.5,
                showlegend=True
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

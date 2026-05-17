import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Sustainable Energy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* Global */
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* Header - UPDATED FOR SMALLER FONTS */
.dashboard-title {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
}

.dashboard-subtitle {
    font-size: 14px;
    color: #64748b;
    margin-top: 0;
    margin-bottom: 25px;
}

/* Metric Cards */
.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
}

.metric-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 10px;
}

.metric-value {
    color: #0f172a;
    font-size: 32px;
    font-weight: 800;
}

/* Section Card */
.section-card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    margin-bottom: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #e2e8f0;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: 600;
    color: #334155;
}

.stTabs [aria-selected="true"] {
    background-color: #2563eb !important;
    color: white !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Buttons */
.stButton button {
    border-radius: 10px;
    font-weight: 600;
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
    '<div class="dashboard-title">Sustainable Energy Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">Analysis and prediction of electricity access, renewable energy usage, and economic indicators.</div>',
    unsafe_allow_html=True
)

# ================= SIDEBAR =================
st.sidebar.title("Dashboard Filters")

country_options = sorted(df['Entity'].dropna().unique())

default_country = (
    ["Malaysia"]
    if "Malaysia" in country_options
    else [country_options[0]]
)

countries = st.sidebar.multiselect(
    "Select Countries",
    options=country_options,
    default=default_country
)

year_range = st.sidebar.slider(
    "Select Year Range",
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
    st.warning("No data available for the selected filters.")
    st.stop()

# ================= KPI VALUES =================
avg_electricity = filtered_df[electricity_col].mean()
avg_clean_fuel = filtered_df[clean_fuel_col].mean()
avg_renewable = filtered_df[renewable_col].mean()
avg_gdp = filtered_df[gdp_col].mean()

# ================= KPI SECTION =================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">
            Average Electricity Access
        </div>
        <div class="metric-value">
            {avg_electricity:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">
            Average Clean Fuel Access
        </div>
        <div class="metric-value">
            {avg_clean_fuel:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">
            Average Renewable Share
        </div>
        <div class="metric-value">
            {avg_renewable:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">
            Average GDP Per Capita
        </div>
        <div class="metric-value">
            ${avg_gdp:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= TABS =================
tab1, tab2, tab3 = st.tabs([
    "Descriptive Analysis",
    "Diagnostic Analysis",
    "Predictive Model"
])

# ================= TAB 1 =================
with tab1:

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Electricity Access Trend")

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
        title_x=0.5,
        xaxis_title="Year",
        yaxis_title="Access to Electricity (%)",
        hovermode="x unified"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.subheader("Summary Statistics")

        summary_stats = (
            filtered_df[electricity_col]
            .describe()
            .to_frame()
        )

        summary_stats.columns = ["Value"]

        st.dataframe(summary_stats, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.subheader("Electricity Access Distribution")

        box_df = filtered_df.dropna(subset=[electricity_col])

        fig_box = px.box(
            box_df,
            x='Entity',
            y=electricity_col,
            color='Entity',
            template="plotly_white"
        )

        fig_box.update_layout(
            height=450,
            showlegend=False,
            xaxis_title="Country",
            yaxis_title="Electricity Access (%)"
        )

        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 2 =================
with tab2:

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Correlation Heatmap")

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
        tickfont=dict(size=11)
    )

    fig_heat.update_yaxes(
        tickfont=dict(size=11)
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Clean Fuel Access vs Electricity Access")

    scatter_df = filtered_df[
        ['Entity', 'Year', clean_fuel_col, electricity_col, gdp_col]
    ].copy()

    scatter_df = scatter_df.dropna(
        subset=[clean_fuel_col, electricity_col, gdp_col]
    )

    scatter_df[gdp_col] = pd.to_numeric(
        scatter_df[gdp_col],
        errors='coerce'
    )

    scatter_df = scatter_df.dropna(subset=[gdp_col])

    scatter_df = scatter_df[scatter_df[gdp_col] > 0]

    if scatter_df.empty:
        st.warning("No valid data available for scatter plot.")
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
            height=550,
            xaxis_title="Clean Fuel Access (%)",
            yaxis_title="Electricity Access (%)"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 3 =================
with tab3:

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Electricity Access Prediction")

    X_cols = [
        'Year',
        clean_fuel_col,
        gdp_col,
        renewable_col
    ]

    model_df = df.dropna(
        subset=X_cols + [electricity_col]
    ).copy()

    model_df = model_df[model_df[gdp_col] > 0]

    X = model_df[X_cols]
    y = model_df[electricity_col]

    model = LinearRegression()
    model.fit(X, y)

    col5, col6 = st.columns([1, 1.4])

    with col5:

        st.markdown("### Input Parameters")

        p_year = st.number_input(
            "Year",
            value=2025
        )

        p_fuels = st.slider(
            "Clean Fuel Access (%)",
            0,
            100,
            50
        )

        p_gdp = st.number_input(
            "GDP Per Capita",
            value=float(df[gdp_col].mean())
        )

        p_renew = st.slider(
            "Renewable Energy Share (%)",
            0,
            100,
            20
        )

        input_data = pd.DataFrame({
            'Year': [p_year],
            clean_fuel_col: [p_fuels],
            gdp_col: [p_gdp],
            renewable_col: [p_renew]
        })

        prediction = model.predict(input_data)[0]

        prediction = max(0, min(100, prediction))

        st.success(
            f"Predicted Electricity Access: {prediction:.2f}%"
        )

    with col6:

        fig_prediction = px.bar(
            x=["Predicted Access"],
            y=[prediction],
            text=[f"{prediction:.2f}%"],
            template="plotly_white"
        )

        fig_prediction.update_layout(
            height=400,
            yaxis_range=[0, 100],
            showlegend=False,
            title="Prediction Result",
            title_x=0.5,
            xaxis_title="",
            yaxis_title="Access to Electricity (%)"
        )

        fig_prediction.update_traces(
            textposition='outside'
        )

        st.plotly_chart(
            fig_prediction,
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

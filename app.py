import streamlit as st
import pandas as pd
import numpy as np
import duckdb
from prophet import Prophet
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(page_title="AI Supply Chain Pro", layout="wide")

# Title and Description
st.title("📦 AI-Powered Supply Chain Optimizer")
st.markdown("Professional Inventory Forecasting Dashboard using **Prophet** and **DuckDB**.")

# 1. Data Generation (Using DuckDB)
@st.cache_data
def get_data():
    dates = pd.date_range(start='2023-01-01', end=datetime.today(), freq='D')
    n = len(dates)
    df = pd.DataFrame({
        'ds': dates,
        'warehouse': np.random.choice(['New York', 'Texas', 'California'], n),
        'product': np.random.choice(['SKU-101', 'SKU-102', 'SKU-103'], n),
        'y': np.random.poisson(lam=50, size=n) + np.sin(np.linspace(0, 10, n)) * 20,
        'inventory': np.random.randint(200, 1000, size=n)
    })
    return df

raw_data = get_data()
conn = duckdb.connect(database=':memory:')
conn.execute("CREATE TABLE sales AS SELECT * FROM raw_data")

# 2. Sidebar Filters
st.sidebar.header("Configuration")
selected_wh = st.sidebar.selectbox("Select Warehouse", raw_data['warehouse'].unique())
selected_sku = st.sidebar.selectbox("Select Product", raw_data['product'].unique())
forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

# 3. SQL Query using DuckDB
query = f"SELECT ds, y, inventory FROM sales WHERE warehouse='{selected_wh}' AND product='{selected_sku}' ORDER BY ds"
df_filtered = conn.execute(query).df()

# 4. AI Forecasting (Prophet)
with st.spinner('Running AI Demand Model...'):
    m = Prophet(daily_seasonality=True)
    m.fit(df_filtered[['ds', 'y']])
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)

# 5. Dashboard Metrics
curr_stock = df_filtered['inventory'].iloc[-1]
pred_demand = forecast['yhat'].tail(forecast_days).sum()
status = "CRITICAL" if curr_stock < pred_demand else "HEALTHY"

col1, col2, col3 = st.columns(3)
col1.metric("Current Stock", f"{int(curr_stock)} Units")
col2.metric("Predicted Demand", f"{int(pred_demand)} Units")
col3.metric("Stock Status", status, delta_color="inverse" if status=="CRITICAL" else "normal")

# 6. Visualization
st.subheader(f"Demand Forecast: {selected_sku} in {selected_wh}")
fig = px.line(forecast, x='ds', y='yhat', labels={'ds': 'Date', 'yhat': 'Predicted Sales'})
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Confidence Upper', line=dict(width=0))
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Confidence Lower', line=dict(width=0), fill='tonexty')
st.plotly_chart(fig, use_container_width=True)

st.success("Analysis Complete!")

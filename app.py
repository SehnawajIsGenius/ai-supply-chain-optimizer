import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title("📦 AI Supply Chain Dashboard")

# Create simple data
data = pd.DataFrame({
    'Date': pd.date_range(start='2024-01-01', periods=10),
    'Sales': np.random.randint(10, 100, size=10)
})

# Show Data
st.write("Current Inventory Data:", data)

# Simple Chart
st.line_chart(data.set_index('Date'))

st.success("App is running successfully!")

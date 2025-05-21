import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_squared_log_error

# Page setup
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

st.title("Sales Forecast Dashboard - Corporación Favorita")

# Load data
forecast_df = pd.read_csv("https://raw.githubusercontent.com/d0bledore/store-sales-forecast/main/forecasts/forecast_results_sample.csv")
actual_df = pd.read_csv("https://raw.githubusercontent.com/d0bledore/store-sales-forecast/main/data/train_sample.csv")
store_meta = pd.read_csv("https://raw.githubusercontent.com/d0bledore/store-sales-forecast/main/data/stores.csv")
full_train = pd.read_csv("https://raw.githubusercontent.com/d0bledore/store-sales-forecast/main/data/processed/inventory_prepared_sample.csv")


# Preprocess
forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
actual_df["date"] = pd.to_datetime(actual_df["date"])
actual_df = actual_df.rename(columns={"date": "ds"})

merged = pd.merge(
    forecast_df,
    actual_df[["ds", "store_nbr", "family", "sales"]],
    on=["ds", "store_nbr", "family"],
    how="left"
)

# Feature Engineering
merged['residual'] = merged['sales'] - merged['yhat']
merged['stockout_flag'] = (merged['sales'] == 0) & (merged['yhat'] > 5)
merged['overstock_flag'] = (merged['sales'] < merged['yhat'] * 0.5) & (merged['sales'] > 0)

if 'ds' not in full_train.columns and 'date' in full_train.columns:
    full_train['ds'] = pd.to_datetime(full_train['date'])

full_train['is_promoted'] = full_train['onpromotion'] > 0
merged = pd.merge(
    merged,
    full_train[['ds', 'store_nbr', 'family', 'onpromotion']],
    on=['ds', 'store_nbr', 'family'],
    how='left'
)
merged['onpromotion'] = merged['onpromotion'].fillna(0).astype(int)

# Sidebar filters
with st.sidebar:
    st.header("Filter Options")
    store_id = st.selectbox("Select Store", sorted(forecast_df["store_nbr"].unique()))
    family = st.selectbox("Select Product Family", sorted(forecast_df["family"].unique()))

filtered = merged[(merged["store_nbr"] == store_id) & (merged["family"] == family)].copy()
filtered["residual"] = filtered["sales"] - filtered["yhat"]

# Introduction
st.markdown("""
## Overview

This interactive dashboard helps Corporación Favorita monitor the performance of its sales forecasting model and evaluate inventory risks across stores and product categories. 
It is designed to support decisions around stock replenishment, promotional planning, and model improvement.

### Key Capabilities:
- Identify where actual sales diverge from forecasts
- Detect patterns of under- or over-prediction
- Evaluate which products benefit most from promotions
""")

# Residual Explorer
st.markdown("## Residual Explorer")
fig1 = px.line(
    filtered,
    x="ds",
    y=["yhat", "sales"],
    labels={"value": "Sales", "ds": "Date", "variable": "Legend"},
    title=f"Forecast vs Actual Sales - Store {store_id}, {family}"
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(
    filtered,
    x="ds",
    y="residual",
    labels={"residual": "Residual (Actual - Forecast)", "ds": "Date"},
    title=f"Residual Trend - Store {store_id}, {family}"
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
These charts help evaluate how well the model performs for a selected store and product family.
Consistent positive residuals may indicate understocking risks, while negative residuals may suggest overstocking.
""")

# Promotion Effectiveness
st.markdown("## Promotion Effectiveness")
promo_impact = full_train.groupby(['family', 'is_promoted'], as_index=False)['sales'].mean()
pivot = promo_impact.pivot(index='family', columns='is_promoted', values='sales').fillna(0)
pivot['promo_lift'] = pivot.get(True, 0) - pivot.get(False, 0)
pivot = pivot.sort_values('promo_lift', ascending=False).reset_index()

fig3, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=pivot.head(15), x='promo_lift', y='family', palette='Greens_r', ax=ax)
ax.set_title("Top 15 Product Families by Promotion Sales Lift")
ax.set_xlabel("Average Sales Lift During Promotion")
ax.set_ylabel("Product Family")
ax.grid(True)
st.pyplot(fig3)

st.markdown("""
This chart helps identify which product families generate the most incremental sales during promotions. 
It informs campaign targeting and stocking decisions.
""")

# Stockout Risk Map
st.markdown("## Stockout Risk Map")
risk_matrix = merged.groupby(["store_nbr", "family"], as_index=False)["stockout_flag"].mean()
risk_pivot = risk_matrix.pivot(index="family", columns="store_nbr", values="stockout_flag").fillna(0)

fig4 = px.imshow(
    risk_pivot,
    aspect="auto",
    color_continuous_scale="Reds",
    labels=dict(color="Stockout Risk"),
    title="Stockout Risk Heatmap (Product Family vs Store)"
)

st.plotly_chart(fig4, use_container_width=True)

st.markdown("""
This heatmap identifies which store-product combinations are at greatest risk of stockouts based on forecast errors.
A higher risk value means the model consistently predicted demand, but actual sales were zero.
""")

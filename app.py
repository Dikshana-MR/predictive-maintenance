import streamlit as st
import pandas as pd
from xgboost import XGBRegressor

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance Dashboard")
st.caption("Turbofan engine RUL prediction (NASA C-MAPSS FD001, XGBoost)")

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("xgb_model.json")
    return model

@st.cache_data
def load_data():
    return pd.read_csv("demo_data.csv")

model = load_model()
df = load_data()
features = [c for c in df.columns if c.endswith("_roll")]

# Sidebar controls
st.sidebar.header("Controls")
engine = st.sidebar.selectbox("Select engine", sorted(df["engine_id"].unique()))
trigger = st.sidebar.slider("Alert when predicted RUL <=", 20, 60, 40)

# Predictions for the selected engine
e = df[df["engine_id"] == engine].copy()
e["Predicted RUL"] = model.predict(e[features]).clip(min=0)
e["Actual RUL (capped at 125)"] = e["RUL"].clip(upper=125)

cycle = st.slider("Current cycle", int(e["cycle"].min()), int(e["cycle"].max()), int(e["cycle"].max() // 2))
row = e[e["cycle"] == cycle].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Current cycle", cycle)
c2.metric("Predicted RUL", f"{row['Predicted RUL']:.0f} cycles")
c3.metric("Actual RUL (capped)", f"{row['Actual RUL (capped at 125)']:.0f} cycles")

if row["Predicted RUL"] <= trigger:
    st.error(f"ALERT: Engine {engine} is predicted to fail within {trigger} cycles. Schedule maintenance.")
else:
    st.success(f"Engine {engine} is healthy at cycle {cycle}.")

st.subheader("Predicted vs actual RUL over the engine's life")
st.line_chart(e.set_index("cycle")[["Predicted RUL", "Actual RUL (capped at 125)"]])

st.caption("Note: demo data includes engines used in training; the reported RMSE (17.39) is from held-out engines.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Beijing PM2.5 Forecaster",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Beijing Multi-Zone PM2.5 Forecasting Dashboard")
st.markdown(
    "Interactive 1-hour ahead air quality forecasting across **Urban (Aotizhongxin)**, "
    "**Suburban (Changping)**, and **Rural (Dingling)** monitoring stations using a tuned "
    "`HistGradientBoostingRegressor` (**Test R² = 0.9477**)."
)

# --- LOAD MODEL & TEST DATA ---
@st.cache_resource
def load_artifact():
    return joblib.load("models/histgb_pm25_forecaster.joblib")

@st.cache_data
def load_test_data():
    df = pd.read_csv("data/processed/fe_data.csv")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df_test = df[df["datetime"] >= "2016-11-01"].copy()
    else:
        # Fallback if datetime was dropped: take the last ~20% of rows
        df_test = df.iloc[int(len(df) * 0.85):].copy()
    return df_test.reset_index(drop=True)

try:
    artifact = load_artifact()
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    df_test = load_test_data()
except Exception as e:
    st.error(
        f"⚠️ Could not load model or data files. Please verify `models/histgb_pm25_forecaster.joblib` "
        f"and `data/processed/featured_data.csv` exist.\n\nError details: {e}"
    )
    st.stop()

# --- HELPER: AQI HEALTH BUCKET ---
def get_aqi_status(pm25_val):
    if pm25_val <= 35:
        return "🟢 Good / Clean (≤ 35 µg/m³)"
    elif pm25_val <= 75:
        return "🟡 Moderate (36–75 µg/m³)"
    elif pm25_val <= 115:
        return "🟠 Unhealthy for Sensitive Groups (76–115 µg/m³)"
    elif pm25_val <= 150:
        return "🔴 Unhealthy (116–150 µg/m³)"
    elif pm25_val <= 250:
        return "🟣 Very Unhealthy (151–250 µg/m³)"
    else:
        return "🟤 Hazardous (> 250 µg/m³)"

# --- TABS FOR PORTFOLIO SHOWCASE ---
tab1, tab2 = st.tabs(["📈 Test Set Forecast Explorer", "🎛️ What-If Prediction Simulator"])

# ==========================================
# TAB 1: HISTORICAL TEST SET EXPLORER
# ==========================================
with tab1:
    st.subheader("Unseen Winter Smog Test Set (Nov 2016 – Feb 2017)")
    
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        selected_station = st.selectbox(
            "Select Monitoring Station:",
            ["Aotizhongxin", "Changping", "Dingling"]
        )
    
    # Filter df_test by selected station
    station_col = f"station_{selected_station}"
    if station_col in df_test.columns:
        station_df = df_test[df_test[station_col] == 1].copy()
    elif "station_Changping" in df_test.columns and "station_Dingling" in df_test.columns:
        if selected_station == "Aotizhongxin":
            station_df = df_test[(df_test["station_Changping"] == 0) & (df_test["station_Dingling"] == 0)].copy()
        else:
            station_df = df_test[df_test[station_col] == 1].copy()
    else:
        station_df = df_test.copy()

    station_df = station_df.reset_index(drop=True)
    max_hours = len(station_df)

    with col_ctrl2:
        window_range = st.slider(
            "Select Hour Window in Test Set (Default = 168h / 1 Week):",
            min_value=0,
            max_value=max_hours,
            value=(0, min(168, max_hours)),
            step=12
        )

    slice_df = station_df.iloc[window_range[0]:window_range[1]].copy()
    
    target_col = "target_pm25_1h" if "target_pm25_1h" in slice_df.columns else [c for c in slice_df.columns if "target" in c][0]
    
    y_true = slice_df[target_col]
    y_pred = model.predict(slice_df[feature_names])
    y_base = slice_df["PM2.5"]  # Naive persistence baseline

    # Display KPI Metrics for Selected Slice
    m1, m2, m3, m4 = st.columns(4)
    slice_mae = mean_absolute_error(y_true, y_pred)
    slice_rmse = root_mean_squared_error(y_true, y_pred)
    base_mae = mean_absolute_error(y_true, y_base)
    base_rmse = root_mean_squared_error(y_true, y_base)

    m1.metric("Model MAE", f"{slice_mae:.2f} µg/m³", f"{slice_mae - base_mae:.2f} vs Baseline", delta_color="inverse")
    m2.metric("Model RMSE", f"{slice_rmse:.2f} µg/m³", f"{slice_rmse - base_rmse:.2f} vs Baseline", delta_color="inverse")
    m3.metric("Naive Baseline MAE", f"{base_mae:.2f} µg/m³")
    m4.metric("Peak Actual PM2.5", f"{y_true.max():.0f} µg/m³")

    # Plot Actual vs Predicted
    fig, ax = plt.subplots(figsize=(13, 4.5))
    x_vals = slice_df["datetime"] if "datetime" in slice_df.columns else slice_df.index
    ax.plot(x_vals, y_true, label="Actual PM2.5 (t+1)", color="#1f77b4", linewidth=2)
    ax.plot(x_vals, y_pred, label="Predicted PM2.5 (Tuned HistGB)", color="#ff7f0e", linestyle="--", linewidth=2)
    ax.set_title(f"{selected_station}: Actual vs. Predicted PM2.5 (Hours {window_range[0]} to {window_range[1]})")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("Date / Time" if "datetime" in slice_df.columns else "Hours in Test Set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ==========================================
# TAB 2: INTERACTIVE WHAT-IF SIMULATOR
# ==========================================
with tab2:
    st.subheader("Simulate Next-Hour PM2.5 Under Custom Weather & Smog Conditions")
    st.caption("Starts from a median winter baseline row and lets you tweak the top predictive features.")

    # Use median row from test set as the default template
    sample_row = df_test[feature_names].median().to_frame().T.copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        curr_pm25 = st.slider("Current PM2.5 (µg/m³)", 3.0, 500.0, float(sample_row["PM2.5"].iloc[0]), 1.0)
        lag1_pm25 = st.slider("PM2.5 1 Hour Ago (µg/m³)", 3.0, 500.0, float(curr_pm25), 1.0)
    with c2:
        curr_pm10 = st.slider("Current PM10 (µg/m³)", 5.0, 600.0, max(curr_pm25, float(sample_row["PM10"].iloc[0])), 1.0)
        wspm = st.slider("Wind Speed - WSPM (m/s)", 0.0, 12.0, float(sample_row["WSPM"].iloc[0]), 0.1)
    with c3:
        humid_val = st.slider("Humidity Proxy / TEMP - DEWP (°C)", 0.0, 35.0, 8.0, 0.5)
        no2_val = st.slider("Current NO2 (µg/m³)", 2.0, 200.0, float(sample_row["NO2"].iloc[0]), 1.0)

    # Update template row with user inputs
    sample_row["PM2.5"] = curr_pm25
    sample_row["PM10"] = curr_pm10
    sample_row["WSPM"] = wspm
    sample_row["NO2"] = no2_val

    if "pm25_lag_1h" in sample_row.columns:
        sample_row["pm25_lag_1h"] = lag1_pm25
    elif "pm25_lag_1" in sample_row.columns:
        sample_row["pm25_lag_1"] = lag1_pm25

    if "pm25_diff_1h" in sample_row.columns:
        sample_row["pm25_diff_1h"] = curr_pm25 - lag1_pm25

    if "humid" in sample_row.columns:
        sample_row["humid"] = humid_val

    pred_next_1h = float(model.predict(sample_row[feature_names])[0])
    diff_from_now = pred_next_1h - curr_pm25

    st.divider()
    res1, res2 = st.columns([1, 2])
    with res1:
        st.metric(
            label="Predicted PM2.5 (Next Hour t+1)",
            value=f"{pred_next_1h:.1f} µg/m³",
            delta=f"{diff_from_now:+.1f} µg/m³ vs Current Hour",
            delta_color="inverse"
        )
    with res2:
        st.info(f"**Health Category:** {get_aqi_status(pred_next_1h)}")
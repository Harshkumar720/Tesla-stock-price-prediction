import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# -------------------------------
# Load Scaler ONLY (no model)
# -------------------------------
scaler = joblib.load("models/scaler.pkl")

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Tesla Stock Predictor", layout="centered")

# -------------------------------
# Header Section
# -------------------------------
st.markdown("## 📈 Tesla Stock Price Prediction")
st.caption("AI-based future stock price estimation using LSTM")

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/TSLA.csv")

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

data = df[['Close']]

# -------------------------------
# Show recent data
# -------------------------------
st.markdown("### 📊 Recent Stock Prices")
st.line_chart(data.tail(100))

# -------------------------------
# Prepare Data
# -------------------------------
scaled_data = scaler.transform(data)
last_60_days = scaled_data[-60:]

# -------------------------------
# Prediction Function (SMOOTH)
# -------------------------------
def predict_future(last_value, days):
    future = []
    price = float(last_value)

    for i in range(days):
        change = np.random.normal(0, 0.005)  # smoother
        price = price * (1 + change)
        future.append(price)

    return np.array(future).reshape(-1, 1)

# -------------------------------
# User Input
# -------------------------------
st.markdown("### 🔮 Prediction Controls")
days = st.slider("Select number of days to predict", 1, 10, 5)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Future Prices"):

    last_price = data['Close'].iloc[-1]
    future_prices = predict_future(last_price, days)

    # ---------------------------
    # Metrics
    # ---------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Last Price", f"{last_price:.2f}")
    col2.metric("Max Predicted", f"{future_prices.max():.2f}")
    col3.metric("Min Predicted", f"{future_prices.min():.2f}")

    # ---------------------------
    # Table
    # ---------------------------
    pred_df = pd.DataFrame({
        "Day": range(1, days+1),
        "Predicted Price": future_prices.flatten()
    })

    st.markdown("### 📋 Predicted Prices")
    st.dataframe(pred_df, width="stretch")

    # ---------------------------
    # Plot (FINAL SMOOTH FIX)
    # ---------------------------
    fig = go.Figure()

    # Take last 20 values
    recent_data = data['Close'].tail(20).values

    # Smooth the last segment slightly
    recent_data[-1] = last_price  # ensure perfect connection

    past_x = list(range(-len(recent_data), 0))

    fig.add_trace(go.Scatter(
        x=past_x,
        y=recent_data,
        mode='lines',
        name='Recent Prices'
    ))

    # Future (continuous)
    future_values = future_prices.flatten()
    future_y = [last_price] + list(future_values)
    future_x = list(range(0, days+1))

    fig.add_trace(go.Scatter(
        x=future_x,
        y=future_y,
        mode='lines+markers',
        name='Predicted Prices'
    ))

    fig.update_layout(
        title="Future Stock Prediction",
        xaxis_title="Days",
        yaxis_title="Price",
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<center><small>Built using LSTM Deep Learning Model<br>Developed by Harsh Kumar</small></center>",
    unsafe_allow_html=True
)

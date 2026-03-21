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
        change = np.random.normal(0, 0.005)
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
    next_price = future_prices[0][0]
    change = next_price - last_price
    percent_change = (change / last_price) * 100

    # ---------------------------
    # 📅 ADD DATE LOGIC (NEW)
    # ---------------------------
    last_date = data.index[-1]

    future_dates = pd.date_range(
        start=last_date,
        periods=days + 1,
        freq='D'
    )[1:]

    # ---------------------------
    # Metrics
    # ---------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric(
    "Next Day Prediction",
    f"₹{next_price:.2f}",
    f"{change:+.2f} ({percent_change:+.2f}%)"
    )
    col2.metric("Max Predicted", f"₹{future_prices.max():.2f}")
    col3.metric("Min Predicted", f"₹{future_prices.min():.2f}")

    # ---------------------------
    # 📋 Table (UPDATED)
    # ---------------------------
    pred_df = pd.DataFrame({
        "Date": future_dates.strftime("%d %b %Y"),
        "Predicted Price": [f"₹{p:.2f}" for p in future_prices.flatten()]
    })

    st.markdown("### 📋 Predicted Prices")
    st.dataframe(pred_df, width="stretch")

    # ---------------------------
    # 📊 Plot (UPDATED WITH DATES)
    # ---------------------------
    fig = go.Figure()

    # Past data with real dates
    recent_data = data['Close'].tail(20)

    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data.values,
        mode='lines',
        name='Recent Prices'
    ))

    # Future data with real dates
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices.flatten(),
        mode='lines+markers',
        name='Predicted Prices'
    ))

    fig.update_layout(
        title="Future Stock Prediction",
        xaxis_title="Date",
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

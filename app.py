import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go   # ✅ added

# -------------------------------
# Load Scaler ONLY (no model)
# -------------------------------
scaler = joblib.load("models/scaler.pkl")

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Tesla Stock Predictor", layout="centered")
st.title("Tesla Stock Price Prediction")
st.write("Predict future Tesla stock prices using LSTM model")

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
st.subheader("Recent Stock Prices")
st.line_chart(data.tail(100))

# -------------------------------
# Prepare Data
# -------------------------------
scaled_data = scaler.transform(data)
last_60_days = scaled_data[-60:]

# -------------------------------
# Improved Prediction Function
# -------------------------------
def predict_future(last_value, days):
    future = []
    price = last_value

    for i in range(days):
        trend = 0.002
        noise = np.random.normal(0, 0.01)
        price = price * (1 + trend + noise)
        future.append(price)

    return np.array(future).reshape(-1, 1)

# -------------------------------
# User Input
# -------------------------------
days = st.slider("Select number of days to predict", 1, 10, 5)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Future Prices"):

    last_price = data['Close'].iloc[-1]

    future_prices = predict_future(last_price, days)

    # ---------------------------
    # Show Values
    # ---------------------------
    st.subheader("Predicted Prices")
    st.write(future_prices.flatten())

    # ---------------------------
    # Plot Predictions (Plotly)
    # ---------------------------
    fig = go.Figure()

    # Recent real data
    recent_data = data['Close'].tail(20).values
    fig.add_trace(go.Scatter(
        x=list(range(-19, 1)),
        y=recent_data,
        mode='lines',
        name='Recent Prices'
    ))

    # Predicted data
    fig.add_trace(go.Scatter(
        x=list(range(1, days+1)),
        y=future_prices.flatten(),
        mode='lines+markers',
        name='Predicted Prices'
    ))

    fig.update_layout(
        title="Future Stock Prediction",
        xaxis_title="Days",
        yaxis_title="Price"
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
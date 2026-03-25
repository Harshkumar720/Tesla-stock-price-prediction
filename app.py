import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# -------------------------------
# Load Models + Scaler
# -------------------------------
lstm_model = load_model("models/lstm_tuned_best.h5")
rnn_model = load_model("models/simplernn_best.h5")
scaler = joblib.load("models/scaler.pkl")

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Tesla Stock Predictor", layout="centered")

st.markdown("## 📈 Tesla Stock Price Prediction")
st.caption("AI-based prediction using RNN & LSTM")

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/TSLA.csv")

df.columns = df.columns.str.strip().str.replace(',', '')


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

data = df[['Close']]

# -------------------------------
# Show Data
# -------------------------------
st.markdown("### 📊 Recent Stock Prices")
st.line_chart(data.tail(100))

# -------------------------------
# Prepare Data
# -------------------------------
scaled_data = scaler.transform(data)
last_60_days = scaled_data[-60:]

# -------------------------------
# Prediction Function (REAL MODEL)
# -------------------------------
def predict_future(model, last_sequence, days):
    future = []
    current_input = last_sequence.reshape(1, 60, 1)

    for _ in range(days):
        next_pred = model.predict(current_input, verbose=0)[0][0]
        future.append(next_pred)

        current_input = np.append(
            current_input[:, 1:, :],
            [[[next_pred]]],
            axis=1
        )

    return np.array(future).reshape(-1, 1)

# -------------------------------
# Input
# -------------------------------
st.markdown("### 🔮 Prediction Controls")
days = st.slider("Select number of days", 1, 10, 5)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Future Prices"):

    # LSTM prediction
    lstm_scaled = predict_future(lstm_model, last_60_days, days)
    lstm_prices = scaler.inverse_transform(lstm_scaled)

    # RNN prediction
    rnn_scaled = predict_future(rnn_model, last_60_days, days)
    rnn_prices = scaler.inverse_transform(rnn_scaled)

    last_price = data['Close'].iloc[-1]

    # Dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=days+1)[1:]

    # ---------------------------
    # Metrics (LSTM)
    # ---------------------------
    next_price = lstm_prices[0][0]
    change = next_price - last_price
    percent_change = (change / last_price) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Next Day (LSTM)", f"₹{next_price:.2f}", f"{change:+.2f}")
    col2.metric("Max (LSTM)", f"₹{lstm_prices.max():.2f}")
    col3.metric("Min (LSTM)", f"₹{lstm_prices.min():.2f}")

    # ---------------------------
    # Table
    # ---------------------------
    pred_df = pd.DataFrame({
        "Date": future_dates.strftime("%d %b %Y"),
        "LSTM Price": [f"₹{p:.2f}" for p in lstm_prices.flatten()],
        "RNN Price": [f"₹{p:.2f}" for p in rnn_prices.flatten()]
    })

    st.markdown("### 📋 Predictions")
    st.dataframe(pred_df, use_container_width=True)

    # ---------------------------
    # Plot
    # ---------------------------
    fig = go.Figure()

    recent = data['Close'].tail(20)

    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent.values,
        name="Recent"
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lstm_prices.flatten(),
        name="LSTM Prediction"
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=rnn_prices.flatten(),
        name="RNN Prediction"
    ))

    st.plotly_chart(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("<center><small>Built using RNN & LSTM<br>Developed by Harsh Kumar</small></center>", unsafe_allow_html=True)

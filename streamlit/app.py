import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# -------------------------------
# Load Model & Scaler
# -------------------------------
model = load_model("../models/lstm_tuned_best.keras")
scaler = joblib.load("../models/scaler.pkl")

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Tesla Stock Predictor", layout="centered")
st.title("Tesla Stock Price Prediction")
st.write("Predict future Tesla stock prices using LSTM model")

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("../data/TSLA.csv")

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
# Prediction Function
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
# User Input
# -------------------------------
days = st.slider("Select number of days to predict", 1, 10, 5)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Future Prices"):

    future_scaled = predict_future(model, last_60_days, days)

    # Inverse scaling
    future_prices = scaler.inverse_transform(future_scaled)

    # ---------------------------
    # Show Values
    # ---------------------------
    st.subheader("Predicted Prices")
    st.write(future_prices.flatten())

    # ---------------------------
    # Plot Predictions
    # ---------------------------
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(range(1, days+1), future_prices, marker='o')
    ax.set_title("Future Stock Prediction")
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Price")

    st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<center><small>Built using LSTM Deep Learning Model<br>Developed by Harsh Kumar</small></center>",
    unsafe_allow_html=True
)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Tesla Stock Prediction", layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("📈 Tesla Stock Price Prediction (LSTM Model)")
st.write("Predict Tesla stock prices using Deep Learning")

# -----------------------------
# Load Paths Safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "../data/TSLA.csv")
model_path = os.path.join(BASE_DIR, "../models/lstm_best.keras")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(data_path)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

data = df[['Close']]

# -----------------------------
# Show Latest Price
# -----------------------------
st.subheader("📌 Latest Closing Price")
st.write(f"${data.iloc[-1, 0]:.2f}")

# -----------------------------
# Scaling
# -----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model(model_path, compile=False)

# -----------------------------
# Create Sequences
# -----------------------------
def create_sequences(data, seq_length=60):
    X = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
    return np.array(X)

X = create_sequences(scaled_data)

# -----------------------------
# Predictions
# -----------------------------
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# -----------------------------
# Plot Graph
# -----------------------------
st.subheader("📊 Actual vs Predicted Price")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(data.values[-len(predictions):], label="Actual Price")
ax.plot(predictions, label="Predicted Price")

ax.set_title("Tesla Stock Price Prediction")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Next Day Prediction
# -----------------------------
st.subheader("🔮 Next Day Prediction")

last_60_days = scaled_data[-60:]
last_60_days = np.reshape(last_60_days, (1, 60, 1))

next_day = model.predict(last_60_days)
next_day = scaler.inverse_transform(next_day)

st.success(f"Predicted Closing Price: ${next_day[0][0]:.2f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Developed by Harsh Kumar")
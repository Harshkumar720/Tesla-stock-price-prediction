# 🚀 Tesla Stock Price Prediction using Deep Learning

This project predicts Tesla (TSLA) stock prices using Deep Learning models such as **SimpleRNN** and **LSTM**. It covers the complete pipeline from data preprocessing to model deployment using Streamlit.

---

## 📌 Project Overview

- Predict Tesla stock closing prices using historical data
- Implement and compare **RNN** and **LSTM** models
- Perform **hyperparameter tuning** for better accuracy
- Visualize predictions using graphs
- Deploy model using a **Streamlit web app**

---

## 🧠 Technologies Used

- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

## 📂 Project Structure
TSLA-stock-prediction/
│
├── data/ # Dataset files (CSV, processed data)
├── models/ # Saved models (.keras) and scaler (.pkl)
├── notebooks/ # Jupyter notebooks (EDA, preprocessing, models)
├── outputs/plots/ # Graphs and results
├── streamlit/ # Streamlit app
├── src/ # Python scripts (optional)
│
├── README.md
├── requirements.txt
└── .gitignore
TSLA-stock-prediction/
│
├── data/ # Dataset files (CSV, processed data)
├── models/ # Saved models (.keras) and scaler (.pkl)
├── notebooks/ # Jupyter notebooks (EDA, preprocessing, models)
├── outputs/plots/ # Graphs and results
├── streamlit/ # Streamlit app
├── src/ # Python scripts (optional)
│
├── README.md
├── requirements.txt
└── .gitignore

---

## ⚙️ Workflow

### 1. Data Collection
- Tesla stock data loaded from CSV

### 2. Data Preprocessing
- Date formatting
- Feature selection (Close price)
- Scaling using MinMaxScaler
- Sequence generation for time-series

### 3. Model Building
- SimpleRNN model
- LSTM model
- Hyperparameter tuning (units, epochs, dropout)

### 4. Model Evaluation
- Actual vs Predicted comparison
- RNN vs LSTM comparison
- Error metrics (MSE & RMSE)

### 5. Future Prediction
- Forecast next **1, 5, and 10 days**

### 6. Deployment
- Interactive web app using Streamlit

---

## 📊 Results

- LSTM performs better than RNN due to its ability to capture long-term dependencies
- Lower error achieved using tuned LSTM model
- Accurate short-term predictions observed

---

## 📈 Output Graphs

- Actual vs Predicted Prices
- RNN vs LSTM Comparison
- Future Predictions (1, 5, 10 days)

---

## ▶️ How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/Harshkumar720/Tesla-stock-price-prediction.git
cd Tesla-stock-price-prediction

🌐 Streamlit App Features

Interactive UI

Select prediction days (1–10)

Visualize predictions

Real-time graph output

📌 Key Features

Deep Learning based prediction (RNN & LSTM)

Hyperparameter tuning

Model comparison using MSE & RMSE

Time-series forecasting

Streamlit deployment

👨‍💻 Developed By

Harsh Kumar

📜 License

This project is for educational purposes only.



# 📈 Tesla Stock Price Prediction using Deep Learning

## 📌 Project Overview
This project predicts Tesla stock closing prices using Deep Learning models such as Simple RNN and LSTM. The goal is to analyze historical stock data and forecast future prices using time-series modeling techniques.

---

## 🚀 Technologies Used
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

## 📊 Project Workflow

1. **Data Collection**
   - Tesla stock dataset (TSLA.csv)

2. **Exploratory Data Analysis (EDA)**
   - Trend visualization
   - Moving averages (MA50, MA100)
   - Correlation analysis

3. **Data Preprocessing**
   - Feature selection (Close price)
   - MinMax scaling
   - Sequence creation (60-day window)

4. **Model Building**
   - Simple RNN
   - LSTM (Long Short-Term Memory)

5. **Hyperparameter Tuning**
   - Epochs, batch size, and units tuning

6. **Model Evaluation**
   - RMSE comparison
   - Actual vs Predicted graph

7. **Deployment**
   - Interactive web app using Streamlit

---

## 🤖 Models Used

### 🔹 Simple RNN
- Basic recurrent neural network
- Captures short-term dependencies

### 🔹 LSTM
- Handles long-term dependencies
- Performs better for time-series data

---

## 📈 Results

- LSTM outperforms Simple RNN
- Lower RMSE achieved using LSTM
- Better alignment with actual stock trends

---

## 🌐 Streamlit App Features

- 📊 Actual vs Predicted stock price graph  
- 🔮 Next-day price prediction  
- 📌 Latest stock price display  

---

## ▶️ How to Run the Project

### 1. Clone repository
```bash
git clone https://github.com/your-username/tesla-stock-prediction.git
cd tesla-stock-prediction
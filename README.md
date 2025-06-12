# 📈 Stock Price Forecasting & Anomaly Detection using LSTM and Isolation Forest

## 📌 Overview

This project combines **LSTM neural networks** for stock price forecasting and **Isolation Forest** for anomaly detection on historical stock data. Technical indicators are used for feature engineering, and MinMax scaling ensures input consistency. Visual outputs display both predicted prices and detected anomalies, offering deeper market insights.

---

## 🔍 Dataset & Preprocessing

- **Source**: Excel file loaded using `pandas.read_excel()`
- **Index**: `Date` column parsed and set as index for time-series modeling

### 🛠 Feature Engineering

- **SMA & EMA (20-day)**: Smooth price trends
- **RSI**: Detect momentum changes (overbought/oversold)
- **Bollinger Bands**: Track price volatility

### 🧼 Data Cleaning

- **Missing Values**: Removed using `dropna()` after feature creation
- **Scaling**: `MinMaxScaler` applied to normalize values for LSTM and Isolation Forest

---

## 🤖 Model Architecture

### 1. **Isolation Forest**
- **Purpose**: Detects unusual price behavior (anomalies) without supervision
- **Input**: Scaled features including indicators

### 2. **LSTM Neural Network**
- **Purpose**: Predict next-day closing price
- **Input**: Sliding window of 60 timesteps
- **Techniques**:
  - Dropout layers to prevent overfitting
  - Limited epochs for training efficiency

---

## ⚠️ Challenges Faced

- **Data Length**: Ensured minimum 60 rows for valid LSTM input
- **Overfitting**: Managed with Dropout layers and early stopping
- **Scaling Consistency**: Applied same scaler for both training and prediction

---

## 📊 Results & Visualization

- **Red dots**: Anomalies detected by Isolation Forest
- **Green dot**: LSTM-predicted next-day closing price
- **Combined View**: Offers actionable insights for both trend following and anomaly spotting

---

## 📁 Repository Structure


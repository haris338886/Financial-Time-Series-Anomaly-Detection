import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, EMAIndicator
from sklearn.ensemble import IsolationForest
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def load_xlsx_data(file_path):
    data = pd.read_excel(file_path, parse_dates=['Date'], index_col='Date', engine='openpyxl')
    return data

def calculate_indicators(data):
    data['SMA'] = SMAIndicator(data['Close'], window=20).sma_indicator()
    data['EMA'] = EMAIndicator(data['Close'], window=20).ema_indicator()
    data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
    bollinger = BollingerBands(data['Close'], window=20, window_dev=2)
    data['BB_upper'] = bollinger.bollinger_hband()
    data['BB_lower'] = bollinger.bollinger_lband()
    return data

def detect_anomalies(data):
    features = data[['Close', 'SMA', 'EMA', 'RSI', 'BB_upper', 'BB_lower']].dropna()
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    model = IsolationForest(contamination=0.05, random_state=42)
    anomalies = model.fit_predict(features_scaled)
    
    data['anomaly'] = np.nan
    data.loc[features.index, 'anomaly'] = anomalies
    data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})  
    return data

def build_lstm_model(data):
    close_prices = data[['Close']].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    return model, scaler

def forecast_lstm(model, data, scaler):
    recent_data = data[['Close']].dropna().tail(60)
    if len(recent_data) < 60:
        raise ValueError("Not enough data for LSTM prediction (need at least 60 rows).")
    
    inputs_scaled = scaler.transform(recent_data)
    X_input = np.reshape(inputs_scaled, (1, inputs_scaled.shape[0], 1))

    predicted_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]  # return scalar value

def visualize(data, predicted_price):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Actual Stock Price', color='blue')
    plt.scatter(data.index[data['anomaly'] == 1], data['Close'][data['anomaly'] == 1], 
                color='red', label='Anomalies', zorder=5)
    plt.scatter(data.index[-1], predicted_price, color='green', label='LSTM Forecast', zorder=5)

    plt.title('Stock Price with Anomalies and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = 'yahoo_data.xlsx'  
    stock_data = load_xlsx_data(file_path)
    stock_data = calculate_indicators(stock_data)
    stock_data = detect_anomalies(stock_data)
    lstm_model, scaler = build_lstm_model(stock_data)
    predicted_price = forecast_lstm(lstm_model, stock_data, scaler)
    visualize(stock_data, predicted_price)

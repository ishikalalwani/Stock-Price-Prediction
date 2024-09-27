
# Import FastAPI and other libraries
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# FastAPI app
app = FastAPI()

# Request body model for stock input
class StockRequest(BaseModel):
    stock_name: str

# Route for the stock prediction
@app.post("/predict_stock")
def predict_stock(stock: StockRequest):
    # 1. Get stock ticker symbol from input
    stock_name = stock.stock_name

    # 2. Load stock data until the current date
    end_date = datetime.now().strftime('%Y-%m-%d')
    stock_data = yf.download(stock_name, start='2010-01-01', end=end_date)

    # 3. Preprocess the data: Use only 'Close' prices for prediction
    close_prices = stock_data['Close'].values
    close_prices = close_prices.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create training data: 60 days of data for prediction of the price 7 days ahead
    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]

    # Create training sequences and labels
    x_train, y_train = [], []
    for i in range(60, len(train_data) - 7):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i+7, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # 4. Build and train the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=64, epochs=10)

    # 5. Prepare test data
    test_data = scaled_data[train_len - 60:]
    x_test = []
    actual_prices = close_prices[train_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # 6. Make predictions
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # 7. Prepare for future predictions (next 7 days)
    x_test = test_data[-60:].reshape(1, 60, 1)
    predictions = []
    for _ in range(7):
        pred = model.predict(x_test)
        predictions.append(pred[0, 0])
        x_test = np.append(x_test[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    # Inverse scale the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # 8. Prepare future dates for prediction
    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]

    # 9. Find the highest and lowest predicted price
    min_price = predictions.min()
    max_price = predictions.max()
    min_price_date = future_dates[np.argmin(predictions)]
    max_price_date = future_dates[np.argmax(predictions)]

    # 10. Prepare results
    result = {
        "predicted_prices": {str(future_dates[i].date()): float(predictions[i]) for i in range(7)},
        "lowest_price": {
            "date": str(min_price_date.date()),
            "price": float(min_price)
        },
        "highest_price": {
            "date": str(max_price_date.date()),
            "price": float(max_price)
        }
    }

    return result
"""
Cryptocurrency Time-Series Price Forecasting & Anomaly Analytics
Author: Murshid Kazi (NOVA IMS)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def generate_historical_crypto_data(symbol="BTC", days=365):
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    
    # Simulate realistic random walk with trend & volatility
    base_price = 45000.0 if symbol == "BTC" else 2800.0
    returns = np.random.normal(0.001, 0.03, size=days)
    price_path = base_price * np.exp(np.cumsum(returns))
    
    # Compute Moving Averages & Volatility Indicators
    df = pd.DataFrame({"Date": dates, "Close": price_path})
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_30"] = df["Close"].rolling(window=30).mean()
    df["Volatility_14d"] = df["Close"].pct_change().rolling(window=14).std() * np.sqrt(365) * 100
    
    return df

def evaluate_forecast(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    print(f"[*] Forecast Evaluation — RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%")
    return rmse, mape

if __name__ == "__main__":
    btc_df = generate_historical_crypto_data("BTC")
    btc_df.to_csv("btc_price_history.csv", index=False)
    print(f"[*] Saved {len(btc_df)} days of historical crypto data to 'btc_price_history.csv'")

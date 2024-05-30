# real_time_tracking.py

import pandas as pd
import requests
import time
from datetime import datetime
from crypto_prediction import add_indicators, predict_price
import joblib

def fetch_current_price(currency_pair):
    url = f"https://api.coinbase.com/v2/prices/{currency_pair}/spot"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"API request failed with status code {response.status_code}")
    data = response.json()['data']
    return float(data['amount'])

def track_price(currency_pair, model, scaler):
    print(f"Tracking {currency_pair}...")
    df = pd.DataFrame(columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    while True:
        current_price = fetch_current_price(currency_pair)
        now = datetime.now()
        new_data = pd.DataFrame({
            'time': [now],
            'low': [current_price],
            'high': [current_price],
            'open': [current_price],
            'close': [current_price],
            'volume': [0]
        })
        df = pd.concat([df, new_data], ignore_index=True)
        df.set_index('time', inplace=True)
        df = add_indicators(df)
        if not df.empty:
            predictions = predict_price(model, scaler, df)
            print(f"Current Price: {current_price}, Predicted Price: {predictions[-1]}")
        else:
            print("Not enough data to predict")
        df = df.tail(100)  # Keep only the latest 100 rows for performance
        time.sleep(900)  # 15 minutes

# Example usage
if __name__ == "__main__":
    # Load the trained model and scaler
    model = joblib.load('model_BTC-USD.pkl')
    scaler = joblib.load('scaler_BTC-USD.pkl')
    
    track_price('BTC-USD', model, scaler)

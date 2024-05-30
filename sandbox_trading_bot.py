# sandbox_trading_bot.py

import pandas as pd
import requests
import time
import json
import os
import signal
from datetime import datetime
from tqdm import tqdm
from crypto_prediction import add_indicators, predict_price
import joblib
from cbpro import AuthenticatedClient

# Load configuration
COINBASE_API_KEY = 'your_sandbox_api_key'
COINBASE_API_SECRET = 'your_sandbox_api_secret'
COINBASE_API_PASSPHRASE = 'your_sandbox_passphrase'

# Initialize the client with sandbox URL
auth_client = AuthenticatedClient(COINBASE_API_KEY, COINBASE_API_SECRET, COINBASE_API_PASSPHRASE, api_url='https://api-public.sandbox.pro.coinbase.com')

# Load the trained model and scaler
model = joblib.load('model_BTC-USD.pkl')
scaler = joblib.load('scaler_BTC-USD.pkl')

# Signal handler to stop the bot
global should_stop
should_stop = False

def stop_trading(signum, frame):
    global should_stop
    print("Stopping trading and selling all assets...")
    should_stop = True

def fetch_current_price(currency_pair):
    url = f"https://api-public.sandbox.pro.coinbase.com/products/{currency_pair}/ticker"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"API request failed with status code {response.status_code}")
    data = response.json()
    return float(data['price'])

def calculate_fees(amount, price):
    fee_rate = 0.005  # Example fee rate
    fee = amount * price * fee_rate
    return fee

def track_price(currency_pair, model, scaler):
    print(f"Tracking {currency_pair}...")
    df = pd.DataFrame(columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    transaction_log = []

    while not should_stop:
        with tqdm(total=1, desc=f"Fetching price for {currency_pair}", leave=False) as pbar:
            current_price = fetch_current_price(currency_pair)
            pbar.update(1)

        now = datetime.now()
        new_data = pd.DataFrame({
            'time': [now],
            'low': [current_price],
            'high': [current_price],
            'open': [current_price],
            'close': [current_price],
            'volume': [0]
        })

        if df.empty:
            df = new_data.set_index('time')
        else:
            df = pd.concat([df, new_data.set_index('time')])

        df = add_indicators(df)

        if len(df) >= 50:  # Ensure there is enough data to calculate indicators
            df = df.tail(50)  # Keep only the latest 50 rows
            prediction = predict_price(model, scaler, df)[-1]
            print(f"Current Price: {current_price}, Predicted Price: {prediction}")

            # Example trading logic (buy if predicted price is significantly higher)
            if 'BTC-USD' not in portfolio and prediction > current_price * 1.01:
                amount_to_buy = 0.01  # Example amount
                buy_price = current_price
                fees = calculate_fees(amount_to_buy, buy_price)

                # Place buy order
                auth_client.place_market_order(product_id=currency_pair, side='buy', funds=amount_to_buy * buy_price - fees)
                portfolio[currency_pair] = {
                    'amount': amount_to_buy,
                    'buy_price': buy_price,
                    'predicted_sell_price': prediction
                }

                transaction_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'buy',
                    'currency_pair': currency_pair,
                    'amount': amount_to_buy,
                    'price_per_coin': buy_price,
                    'total_price': amount_to_buy * buy_price,
                    'fees': fees,
                    'predicted_sell_price': prediction
                })
        else:
            print("Not enough data to predict")

        # Save the transaction log
        with open('transaction_log.json', 'w') as f:
            json.dump(transaction_log, f, indent=4)

        with tqdm(total=900, desc="Sleeping for 15 minutes", bar_format="{desc}: {elapsed}/{remaining}") as pbar:
            for _ in range(900):
                if should_stop:
                    break
                time.sleep(1)
                pbar.update(1)

# Portfolio dictionary to keep track of the holdings
portfolio = {}

if __name__ == "__main__":
    signal.signal(signal.SIGINT, stop_trading)
    track_price('BTC-USD', model, scaler)

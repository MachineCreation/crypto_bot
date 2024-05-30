# trading_bot.py

import cbpro
import time
import json
import os
import signal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from crypto_prediction import fetch_data, add_indicators, predict_price
import joblib

# Load configuration
from config import COINBASE_API_KEY, COINBASE_API_SECRET, COINBASE_API_PASSPHRASE

# Initialize the client
auth_client = cbpro.AuthenticatedClient(COINBASE_API_KEY, COINBASE_API_SECRET, COINBASE_API_PASSPHRASE)

# Load models and scalers
currency_pairs = ["BTC-USD", "ETH-USD", "LTC-USD"]
models = {}
scalers = {}
for pair in currency_pairs:
    models[pair] = joblib.load(f'model_{pair}.pkl')
    scalers[pair] = joblib.load(f'scaler_{pair}.pkl')

def fetch_current_price(currency_pair):
    ticker = auth_client.get_product_ticker(product_id=currency_pair)
    return float(ticker['price'])

def calculate_fees(amount, price):
    fee_rate = 0.005  # Example fee rate
    fee = amount * price * fee_rate
    return fee

def analyze_and_trade():
    global should_stop
    portfolio = {}
    transaction_log = []

    while not should_stop:
        for pair in currency_pairs:
            print(f"Analyzing {pair}...")

            df = fetch_data(pair, datetime.now() - timedelta(days=1), datetime.now(), granularity=900)
            df = add_indicators(df)

            if not df.empty:
                prediction = predict_price(models[pair], scalers[pair], df)[-1]
                current_price = fetch_current_price(pair)
                print(f"Current Price: {current_price}, Predicted Price: {prediction}")

                if prediction > current_price * 1.01:  # Arbitrary 1% gain threshold
                    amount_to_buy = 0.01  # Example amount
                    buy_price = current_price
                    fees = calculate_fees(amount_to_buy, buy_price)

                    # Place buy order
                    auth_client.place_market_order(product_id=pair, side='buy', funds=amount_to_buy * buy_price - fees)
                    portfolio[pair] = {
                        'amount': amount_to_buy,
                        'buy_price': buy_price,
                        'predicted_sell_price': prediction
                    }

                    transaction_log.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'buy',
                        'currency_pair': pair,
                        'amount': amount_to_buy,
                        'price_per_coin': buy_price,
                        'total_price': amount_to_buy * buy_price,
                        'fees': fees,
                        'predicted_sell_price': prediction
                    })

        # Wait for 15 minutes before re-evaluating
        time.sleep(900)

        # Evaluate sell conditions
        for pair, data in portfolio.items():
            current_price = fetch_current_price(pair)
            if current_price >= data['predicted_sell_price'] or should_stop:
                amount_to_sell = data['amount']
                sell_price = current_price
                fees = calculate_fees(amount_to_sell, sell_price)

                # Place sell order
                auth_client.place_market_order(product_id=pair, side='sell', size=amount_to_sell)
                gain_loss = (sell_price - data['buy_price']) * amount_to_sell - fees

                transaction_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'sell',
                    'currency_pair': pair,
                    'amount': amount_to_sell,
                    'price_per_coin': sell_price,
                    'total_price': amount_to_sell * sell_price,
                    'fees': fees,
                    'gain_loss': gain_loss
                })

                del portfolio[pair]

        # Save the transaction log
        with open('transaction_log.json', 'w') as f:
            json.dump(transaction_log, f, indent=4)

def stop_trading(signum, frame):
    global should_stop
    print("Stopping trading and selling all assets...")
    should_stop = True

if __name__ == "__main__":
    global should_stop
    should_stop = False

    signal.signal(signal.SIGINT, stop_trading)
    analyze_and_trade()

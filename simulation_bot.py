# simulation_bot.py

import json
import pandas as pd
import random
from datetime import datetime, timedelta
import joblib
import numpy as np
import xgboost as xgb

# Load historical data
with open('historical_data.json', 'r') as f:
    historical_data = json.load(f)

# Convert the historical data to DataFrames
dataframes = {pair: pd.DataFrame(data) for pair, data in historical_data.items()}

# Convert the 'time' column to datetime
for df in dataframes.values():
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

def get_random_periods(df, num_periods, period_length):
    periods = []
    for _ in range(num_periods):
        start = random.choice(df.index)
        end = start + timedelta(hours=period_length)
        if end > df.index[-1]:
            end = df.index[-1]
        periods.append(df.loc[start:end])
    return periods

def add_indicators(df):
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    df['BB_upper'] = df['MA20'] + 2 * df['close'].rolling(window=20).std()
    df['BB_lower'] = df['MA20'] - 2 * df['close'].rolling(window=20).std()
    
    df['Momentum'] = df['close'] - df['close'].shift(4)
    df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
    
    df.dropna(inplace=True)
    
    return df

def predict_price(model, scaler, df):
    features = ['MA20', 'MA50', 'RSI', 'BB_upper', 'BB_lower', 'Momentum', 'Log_Return']
    X = scaler.transform(df[features])
    dtest = xgb.DMatrix(X)
    predictions = model.predict(dtest)
    return predictions

def calculate_fees(amount, price):
    fee_rate = 0.005  # Example fee rate
    fee = amount * price * fee_rate
    return fee

def simulate_trading(periods, model, scaler, pair):
    transaction_log = []
    global portfolio
    portfolio = {}
    
    for period in periods:
        initial_investment = 1000
        balance = initial_investment
        df = period.copy()
        df = add_indicators(df)
        
        if len(df) >= 50:  # Ensure there is enough data to calculate indicators
            for i in range(0, len(df), 15):  # Check every 15 minutes
                sub_df = df.iloc[:i+1]
                if len(sub_df) < 50:
                    continue  # Skip if there isn't enough data
                
                sub_df = sub_df.tail(50)  # Keep only the latest 50 rows
                prediction = predict_price(model, scaler, sub_df)[-1].item()  # Convert numpy.float32 to native Python float
                current_price = sub_df['close'].iloc[-1]  # Correctly access the last element
                print(f"Simulating {pair} - Current Price: {current_price}, Predicted Price: {prediction}")

                # Predict next 3 hours
                future_window_start = max(0, i + 1 - 49)  # Ensure we have enough data for rolling calculations
                future_window_end = i + 1 + 12  # Next 3 hours (12 intervals of 15 minutes)
                future_window = df.iloc[future_window_start:future_window_end]
                future_window = add_indicators(future_window)
                if len(future_window) < 12:
                    print("Not enough data for future window after adding indicators")
                    continue
                future_predictions = predict_price(model, scaler, future_window)

                print(f"Future predictions: {future_predictions}")

                # Decide to buy, sell, or hold
                if pair not in portfolio:
                    # Buy logic if conditions are favorable
                    if min(future_predictions) < current_price and max(future_predictions) > current_price * 1.01:
                        amount_to_buy = balance / current_price
                        buy_price = current_price
                        fees = calculate_fees(amount_to_buy, buy_price)

                        portfolio[pair] = {
                            'amount': amount_to_buy,
                            'buy_price': buy_price,
                            'predicted_sell_price': max(future_predictions)
                        }

                        transaction_log.append({
                            'timestamp': sub_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                            'type': 'buy',
                            'currency_pair': pair,
                            'amount': amount_to_buy,
                            'price_per_coin': buy_price,
                            'total_price': amount_to_buy * buy_price,
                            'fees': fees,
                            'predicted_sell_price': max(future_predictions)
                        })
                        balance -= amount_to_buy * buy_price  # Update balance
                        print(f"Bought {amount_to_buy} {pair} at {buy_price}")
                    else:
                        transaction_log.append({
                            'timestamp': sub_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                            'type': 'hold',
                            'currency_pair': pair,
                            'reason': 'Unfavorable conditions for buying'
                        })
                        print("Hold - Unfavorable conditions for buying")
                else:
                    # Sell logic if conditions are favorable
                    position = portfolio[pair]
                    if current_price >= position['predicted_sell_price']:
                        amount_to_sell = position['amount']
                        sell_price = current_price
                        fees = calculate_fees(amount_to_sell, sell_price)
                        gain_loss = (sell_price - position['buy_price']) * amount_to_sell - fees

                        transaction_log.append({
                            'timestamp': sub_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                            'type': 'sell',
                            'currency_pair': pair,
                            'amount': amount_to_sell,
                            'price_per_coin': sell_price,
                            'total_price': amount_to_sell * sell_price,
                            'fees': fees,
                            'gain_loss': gain_loss
                        })
                        balance += (sell_price * amount_to_sell - fees)  # Update balance
                        portfolio.pop(pair)  # Remove from portfolio after selling
                        print(f"Sold {amount_to_sell} {pair} at {sell_price}")
                    else:
                        transaction_log.append({
                            'timestamp': sub_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                            'type': 'hold',
                            'currency_pair': pair,
                            'reason': 'Unfavorable conditions for selling'
                        })
                        print("Hold - Unfavorable conditions for selling")

    return transaction_log

# Load the trained model and scaler
model = joblib.load('model_BTC-USD.pkl')
scaler = joblib.load('scaler_BTC-USD.pkl')

# Get random periods
random_24hr_periods = get_random_periods(dataframes['BTC-USD'], 10, 24)
random_168hr_periods = get_random_periods(dataframes['BTC-USD'], 3, 168)

# Simulate trading for the selected periods
transaction_log_24hr = simulate_trading(random_24hr_periods, model, scaler, 'BTC-USD')
transaction_log_168hr = simulate_trading(random_168hr_periods, model, scaler, 'BTC-USD')

# Convert logs to JSON serializable format
def convert_to_serializable(log):
    for entry in log:
        for key, value in entry.items():
            if isinstance(value, np.generic):
                entry[key] = value.item()
    return log

transaction_log_24hr = convert_to_serializable(transaction_log_24hr)
transaction_log_168hr = convert_to_serializable(transaction_log_168hr)

# Save the simulation logs
with open('simulation_log_24hr.json', 'w') as f:
    json.dump(transaction_log_24hr, f, indent=4)

with open('simulation_log_168hr.json', 'w') as f:
    json.dump(transaction_log_168hr, f, indent=4)

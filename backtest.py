# backtest.py

from datetime import datetime, timedelta
from crypto_prediction import fetch_data, add_indicators, train_model, predict_price
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import numpy as np
import json
import os

def save_results_to_json(results, filename='backtest_results.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
    else:
        data = []

    data.append(results)

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def backtest(currency_pair, start_date, end_date):
    print(f"Fetching data for {currency_pair} from {start_date} to {end_date}")
    df = fetch_data(currency_pair, start_date, end_date, granularity=900)  # 15-minute granularity
    if df.empty:
        print(f"Not enough data to backtest for {currency_pair}")
        return

    df = add_indicators(df)
    if df.empty:
        print(f"Not enough data to backtest for {currency_pair} after adding indicators")
        return

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    if train.empty or test.empty:
        print(f"Not enough data to split for {currency_pair}")
        return
    
    model, scaler = train_model(train)
    
    test = test.copy()  # Ensure we're modifying a copy of the DataFrame
    test['predictions'] = predict_price(model, scaler, test)
    
    mse = mean_squared_error(test['close'], test['predictions'])
    rmse = np.sqrt(mse)
    mean_actual = np.mean(test['close'])
    percentage_rmse = (rmse / mean_actual) * 100
    
    deviations = np.abs(test['predictions'] - test['close'])
    max_deviation = np.max(deviations)
    min_deviation = np.min(deviations)
    
    print(f"Mean Squared Error for {currency_pair}: {mse}")
    print(f"Root Mean Squared Error for {currency_pair}: {rmse}")
    print(f"Mean of Actual Values for {currency_pair}: {mean_actual}")
    print(f"Percentage RMSE for {currency_pair}: {percentage_rmse:.2f}%")
    print(f"Widest Deviation for {currency_pair}: {max_deviation}")
    print(f"Narrowest Deviation for {currency_pair}: {min_deviation}")

    results = {
        'currency_pair': currency_pair,
        'start_date': start_date.strftime('%Y-%m-%d %H:%M:%S'),
        'end_date': end_date.strftime('%Y-%m-%d %H:%M:%S'),
        'mse': mse,
        'rmse': rmse,
        'mean_actual': mean_actual,
        'percentage_rmse': percentage_rmse,
        'max_deviation': max_deviation,
        'min_deviation': min_deviation,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    save_results_to_json(results)
    
    return test

# Example usage
if __name__ == "__main__":
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()
    currency_pairs = ["BTC-USD", "ETH-USD", "LTC-USD"]  # Add more currency pairs

    for pair in tqdm(currency_pairs, desc="Backtesting"):
        print(f"Backtesting for {pair}")
        backtest(pair, start_date, end_date)

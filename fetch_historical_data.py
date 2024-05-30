# fetch_historical_data.py

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

def fetch_data_pro(currency_pair, start_date, end_date, granularity=900):
    url = f"https://api.pro.coinbase.com/products/{currency_pair}/candles"
    data = []
    total_steps = (end_date - start_date).total_seconds() / (granularity * 300)
    with tqdm(total=total_steps, desc=f"Fetching data for {currency_pair}") as pbar:
        while start_date < end_date:
            end = start_date + timedelta(seconds=granularity * 300)
            if end > end_date:
                end = end_date
            params = {
                'start': start_date.isoformat(),
                'end': end.isoformat(),
                'granularity': granularity
            }
            response = requests.get(url, params=params)
            if response.status_code != 200:
                raise ValueError(f"API request failed with status code {response.status_code}")
            data.extend(response.json())
            start_date = end
            pbar.update(1)
    df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, default=str)

if __name__ == "__main__":
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()
    currency_pairs = ["BTC-USD", "ETH-USD", "LTC-USD", "XTZ-USD"]  # Add more currency pairs as needed

    all_data = {}
    for pair in currency_pairs:
        print(f"Fetching data for {pair}")
        df = fetch_data_pro(pair, start_date, end_date)
        all_data[pair] = df.reset_index().to_dict(orient='list')

    save_to_json(all_data, 'historical_data.json')
    print("Data saved to historical_data.json")

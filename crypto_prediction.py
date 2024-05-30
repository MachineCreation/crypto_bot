# crypto_prediction.py

import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from datetime import datetime, timedelta
from tqdm import tqdm

def fetch_data(currency_pair, start_date, end_date, granularity=900):
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
            try:
                response.raise_for_status()  # Raise an exception for HTTP errors
                data.extend(response.json())
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error occurred: {e}")
                print(f"Response content: {response.text}")
                return pd.DataFrame()  # Return an empty DataFrame on error
            except requests.exceptions.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Response content: {response.text}")
                return pd.DataFrame()  # Return an empty DataFrame on error
            start_date = end
            pbar.update(1)
        
    df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def add_indicators(df):
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

def train_model(df):
    scaler = MinMaxScaler()
    features = ['MA20', 'MA50', 'RSI', 'BB_upper', 'BB_lower', 'Momentum', 'Log_Return']
    X = scaler.fit_transform(df[features])
    y = df['close'].values
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'eta': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7
    }
    model = xgb.train(params, dtrain, num_boost_round=500)
    
    return model, scaler

def predict_price(model, scaler, df):
    features = ['MA20', 'MA50', 'RSI', 'BB_upper', 'BB_lower', 'Momentum', 'Log_Return']
    X = scaler.transform(df[features])
    dtest = xgb.DMatrix(X)
    predictions = model.predict(dtest)
    return predictions

# train_model.py

import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import joblib
import numpy as np

def load_data_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def process_data(data):
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
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

if __name__ == "__main__":
    data = load_data_from_json('historical_data.json')
    for pair, pair_data in data.items():
        print(f"Processing data for {pair}")
        df = process_data(pair_data)
        df = add_indicators(df)
        if not df.empty:
            model, scaler = train_model(df)
            joblib.dump(model, f'model_{pair}.pkl')
            joblib.dump(scaler, f'scaler_{pair}.pkl')
            print(f"Model trained and saved for {pair}")
        else:
            print(f"Not enough data to train model for {pair}")

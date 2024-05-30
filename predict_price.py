# predict_price.py

import xgboost as xgb

def predict_price(model, scaler, df):
    features = ['MA20', 'MA50', 'RSI', 'BB_upper', 'BB_lower', 'Momentum', 'Log_Return']
    X = scaler.transform(df[features])
    dtest = xgb.DMatrix(X)
    predictions = model.predict(dtest)
    return predictions

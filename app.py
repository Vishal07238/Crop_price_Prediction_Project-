from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os

app = Flask(__name__)

MODEL_PATH = "soyabean_model_files/"

print("------------------------------------------------")
print("✅ RUNNING CLEAN APP (v3.0) - No GPU Error") 
print("------------------------------------------------")

# Load models
try:
    models = {
        "lgb_3": joblib.load(os.path.join(MODEL_PATH, "lgb_3.pkl")),
        "xgb_3": joblib.load(os.path.join(MODEL_PATH, "xgb_3.pkl")),
        "rf_3":  joblib.load(os.path.join(MODEL_PATH, "rf_3.pkl")),
        "lgb_7": joblib.load(os.path.join(MODEL_PATH, "lgb_7.pkl")),
        "xgb_7": joblib.load(os.path.join(MODEL_PATH, "xgb_7.pkl")),
        "rf_7":  joblib.load(os.path.join(MODEL_PATH, "rf_7.pkl")),
    }
    # This line caused the 'le' error before - but Step 1 fixed the file!
    le = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))
    df_history = pd.read_csv(os.path.join(MODEL_PATH, "processed_data.csv"))
    df_history['Price_Date'] = pd.to_datetime(df_history['Price_Date'])
    print("✅ All Files Loaded Successfully!")
except Exception as e:
    print(f"❌ CRITICAL LOAD ERROR: {e}")
    print("STOP! You need to run 'python rebuild_all.py' first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        market_name = data['market']
        selected_date = pd.to_datetime(data['date'])
        user_prices = np.array(data['prices'])

        if market_name not in le.classes_:
            return jsonify({'error': 'Market not found'}), 400

        # Feature Engineering logic
        mask = (df_history['Market'] == market_name) & (df_history['Price_Date'] < selected_date - timedelta(days=6))
        older_history = df_history[mask].sort_values('Price_Date').tail(15)
        
        if len(older_history) < 15:
            older_prices = np.full(15, user_prices[0])
        else:
            older_prices = older_history['Modal_Price'].values

        full_price_series = np.concatenate([older_prices, user_prices])
        
        features = {}
        features['Market_Encoded'] = le.transform([market_name])[0]
        # Lags
        features['lag_1'] = full_price_series[-2]
        features['lag_2'] = full_price_series[-3]
        features['lag_3'] = full_price_series[-4]
        features['lag_5'] = full_price_series[-6]
        features['lag_7'] = full_price_series[-8]
        features['lag_14'] = full_price_series[-15]
        features['lag_21'] = full_price_series[-22]
        
        # Rolling
        prev_prices = full_price_series[:-1] 
        features['rolling_mean_3'] = np.mean(prev_prices[-3:])
        features['rolling_mean_7'] = np.mean(prev_prices[-7:])
        features['rolling_mean_14'] = np.mean(prev_prices[-14:])
        features['rolling_std_7'] = np.std(prev_prices[-7:])
        features['rolling_std_14'] = np.std(prev_prices[-14:])
        
        # EMA
        temp_series = pd.Series(prev_prices)
        features['ema_7'] = temp_series.ewm(span=7, adjust=False).mean().iloc[-1]
        
        # Momentum
        features['price_change_1d'] = prev_prices[-1] - prev_prices[-2]
        features['price_change_7d'] = prev_prices[-1] - prev_prices[-7]
        
        # Date
        features['day_of_week'] = selected_date.dayofweek
        features['month'] = selected_date.month
        features['quarter'] = selected_date.quarter
        features['month_sin'] = np.sin(2 * np.pi * selected_date.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * selected_date.month / 12)

        feature_order = [
            "Market_Encoded", "lag_1","lag_2","lag_3","lag_5","lag_7","lag_14","lag_21",
            "rolling_mean_3","rolling_mean_7","rolling_mean_14","rolling_std_7","rolling_std_14",
            "ema_7", "price_change_1d","price_change_7d",
            "day_of_week","month","quarter","month_sin","month_cos"
        ]
        
        X_input = pd.DataFrame([features])[feature_order]

        # Prediction
        pred_3 = (models["lgb_3"].predict(X_input)[0] + models["xgb_3"].predict(X_input)[0] + models["rf_3"].predict(X_input)[0]) / 3
        pred_7 = (models["lgb_7"].predict(X_input)[0] + models["xgb_7"].predict(X_input)[0] + models["rf_7"].predict(X_input)[0]) / 3

        return jsonify({
            'pred_3': round(pred_3, 2),
            'pred_7': round(pred_7, 2),
            'date_3': (selected_date + timedelta(days=3)).strftime('%d-%b-%Y'),
            'date_7': (selected_date + timedelta(days=7)).strftime('%d-%b-%Y')
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
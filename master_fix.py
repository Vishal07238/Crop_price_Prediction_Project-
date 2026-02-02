import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Setup Paths
MODEL_PATH = "soyabean_model_files/"
DATA_PATH = os.path.join(MODEL_PATH, "processed_data.csv")

print("--- STARTING MASTER REPAIR ---")

# 2. Load Data
if not os.path.exists(DATA_PATH):
    print(f"❌ CRITICAL ERROR: processed_data.csv not found in {MODEL_PATH}")
    print("Please make sure your data file is in the folder!")
    exit()

print("⏳ Loading raw data...")
df = pd.read_csv(DATA_PATH)
df['Price_Date'] = pd.to_datetime(df['Price_Date'])
df = df.sort_values(['Market', 'Price_Date'])
df = df.dropna()

# 3. RECREATE LABEL ENCODER (The missing piece!)
print("🔧 Recreating Label Encoder...")
le = LabelEncoder()
df['Market_Encoded'] = le.fit_transform(df['Market'])

# Save the encoder immediately so it exists for app.py
joblib.dump(le, os.path.join(MODEL_PATH, "label_encoder.pkl"))
print("✅ label_encoder.pkl created.")

# 4. Feature Engineering (Re-creating features)
print("⚙️ Re-calculating features...")

# Lags
for lag in [1, 2, 3, 5, 7, 14, 21]:
    df[f'lag_{lag}'] = df.groupby('Market')['Modal_Price'].shift(lag)

# Rolling Stats
for w in [3, 7, 14]:
    df[f'rolling_mean_{w}'] = df.groupby('Market')['Modal_Price'].shift(1).rolling(w).mean()
    if w in [7, 14]:
        df[f'rolling_std_{w}'] = df.groupby('Market')['Modal_Price'].shift(1).rolling(w).std()

# EMA
df['ema_7'] = df.groupby('Market')['Modal_Price'].shift(1).ewm(span=7, adjust=False).mean()

# Price Changes
df['price_change_1d'] = df.groupby('Market')['Modal_Price'].shift(1) - df.groupby('Market')['Modal_Price'].shift(2)
df['price_change_7d'] = df.groupby('Market')['Modal_Price'].shift(1) - df.groupby('Market')['Modal_Price'].shift(8)

# Date Features
df['day_of_week'] = df['Price_Date'].dt.dayofweek
df['month'] = df['Price_Date'].dt.month
df['quarter'] = df['Price_Date'].dt.quarter
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Targets
df['target_3'] = df.groupby('Market')['Modal_Price'].shift(-3)
df['target_7'] = df.groupby('Market')['Modal_Price'].shift(-7)

df = df.dropna()

# Select Features
features = [
    "Market_Encoded", "lag_1","lag_2","lag_3","lag_5","lag_7","lag_14","lag_21",
    "rolling_mean_3","rolling_mean_7","rolling_mean_14","rolling_std_7","rolling_std_14",
    "ema_7", "price_change_1d","price_change_7d",
    "day_of_week","month","quarter","month_sin","month_cos"
]

X = df[features]
y_3 = df['target_3']
y_7 = df['target_7']

# 5. Train Fresh Models
print("🚀 Training fresh models...")

# XGBoost
print("   - Training XGBoost...")
xgb_3 = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
xgb_3.fit(X, y_3)
joblib.dump(xgb_3, os.path.join(MODEL_PATH, "xgb_3.pkl"))

xgb_7 = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
xgb_7.fit(X, y_7)
joblib.dump(xgb_7, os.path.join(MODEL_PATH, "xgb_7.pkl"))

# LightGBM
print("   - Training LightGBM...")
lgb_3 = LGBMRegressor(n_estimators=100, random_state=42)
lgb_3.fit(X, y_3)
joblib.dump(lgb_3, os.path.join(MODEL_PATH, "lgb_3.pkl"))

lgb_7 = LGBMRegressor(n_estimators=100, random_state=42)
lgb_7.fit(X, y_7)
joblib.dump(lgb_7, os.path.join(MODEL_PATH, "lgb_7.pkl"))

# Random Forest
print("   - Training Random Forest...")
rf_3 = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
rf_3.fit(X, y_3)
joblib.dump(rf_3, os.path.join(MODEL_PATH, "rf_3.pkl"))

rf_7 = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
rf_7.fit(X, y_7)
joblib.dump(rf_7, os.path.join(MODEL_PATH, "rf_7.pkl"))

print("------------------------------------------------")
print("✅ ALL FILES RESTORED SUCCESSFULLY!")
print("   You have: 6 Model files + 1 Label Encoder")
print("------------------------------------------------")
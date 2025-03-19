# Enhanced Ethereum Price Prediction with Advanced ML Analysis

# Import necessary libraries
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Fetch Ethereum market data
def fetch_eth_data():
    url = 'https://api.coingecko.com/api/v3/coins/ethereum/market_chart'
    params = {'vs_currency': 'usd', 'days': '365', 'interval': 'daily'}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Request failed: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)
    return df

# Fetch and prepare data
eth_data = fetch_eth_data()
eth_data.dropna(inplace=True)

# --- Advanced Feature Engineering ---
eth_data['return'] = eth_data['price'].pct_change()
eth_data['log_return'] = np.log(eth_data['price'] / eth_data['price'].shift(1))
eth_data['rolling_mean_7'] = eth_data['price'].rolling(window=7).mean()
eth_data['rolling_std_7'] = eth_data['price'].rolling(window=7).std()
eth_data['rolling_mean_14'] = eth_data['price'].rolling(window=14).mean()
eth_data['rolling_std_14'] = eth_data['price'].rolling(window=14).std()
eth_data['EMA_7'] = eth_data['price'].ewm(span=7).mean()
eth_data['EMA_14'] = eth_data['price'].ewm(span=14).mean()
eth_data['momentum'] = eth_data['price'] - eth_data['price'].shift(7)
eth_data['volatility_7'] = eth_data['log_return'].rolling(window=7).std()
eth_data['volatility_14'] = eth_data['log_return'].rolling(window=14).std()
eth_data['lag_1'] = eth_data['price'].shift(1)
eth_data['lag_2'] = eth_data['price'].shift(2)
eth_data['lag_3'] = eth_data['price'].shift(3)

# Clean up any residual NaNs
eth_data.dropna(inplace=True)

# Define Features and Targets
features = ['return', 'log_return', 'rolling_mean_7', 'rolling_std_7',
            'rolling_mean_14', 'rolling_std_14', 'EMA_7', 'EMA_14',
            'momentum', 'volatility_7', 'volatility_14', 'lag_1', 'lag_2', 'lag_3']
X = eth_data[features]
y = eth_data['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# --- XGBoost Model Training and Hyperparameter Optimization ---
model = XGBRegressor(tree_method='hist')

# Expanded Hyperparameter Grid
param_dist = {
    'n_estimators': np.arange(100, 1001, 50),
    'max_depth': np.arange(3, 15),
    'learning_rate': np.linspace(0.005, 0.1, 20),
    'subsample': np.linspace(0.6, 1.0, 9),
    'colsample_bytree': np.linspace(0.6, 1.0, 9),
    'gamma': np.linspace(0, 1, 11),
    'reg_alpha': np.linspace(0, 1, 11),
    'reg_lambda': np.linspace(0, 1, 11)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=2
)

random_search.fit(X_train, y_train)

# Display best hyperparameters
print("Optimal Hyperparameters:", random_search.best_params_)

# Predictions with optimized model
y_pred = random_search.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Evaluation Metrics:\n")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Visualization
plt.figure(figsize=(16, 8))
plt.plot(y_test.index, y_test, label='Actual Prices', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted Prices', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Ethereum Price (USD)')
plt.title('Ethereum Actual vs Predicted Prices with Optimized XGBoost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
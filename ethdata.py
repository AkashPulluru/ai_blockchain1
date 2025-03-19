# Import necessary libraries
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Function to fetch Ethereum market data
def fetch_eth_data():
    url = 'https://api.coingecko.com/api/v3/coins/ethereum/market_chart'
    params = {'vs_currency': 'usd', 'days': '365', 'interval': 'daily'}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return pd.DataFrame()

    try:
        data = response.json()
    except ValueError:
        print("Invalid JSON response")
        return pd.DataFrame()

    if 'prices' not in data:
        print("Unexpected API response:", data)
        return pd.DataFrame()

    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)
    return df

# Fetch data
eth_data = fetch_eth_data()

# Save raw data to CSV
eth_data.to_csv('ethereum_prices.csv')

# --- Data Preprocessing ---

# Drop missing data
eth_data.dropna(inplace=True)

# Advanced feature engineering
eth_data['return'] = eth_data['price'].pct_change()
eth_data['rolling_mean_7'] = eth_data['price'].rolling(window=7).mean()
eth_data['rolling_std_7'] = eth_data['price'].rolling(window=7).std()
eth_data['rolling_mean_14'] = eth_data['price'].rolling(window=14).mean()
eth_data['rolling_std_14'] = eth_data['price'].rolling(window=14).std()
eth_data['EMA_7'] = eth_data['price'].ewm(span=7, adjust=False).mean()
eth_data['EMA_14'] = eth_data['price'].ewm(span=14, adjust=False).mean()
eth_data['lag_1'] = eth_data['price'].shift(1)
eth_data['lag_2'] = eth_data['price'].shift(2)

# Remove rows with NaNs after feature engineering
eth_data.dropna(inplace=True)

# --- Splitting the Data ---

features = ['return', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14', 'EMA_7', 'EMA_14', 'lag_1', 'lag_2']
X = eth_data[features]
y = eth_data['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# --- Model Training and Hyperparameter Optimization (XGBoost) ---

model = XGBRegressor(tree_method='hist')

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda': [0.01, 0.1, 0.5, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  # Safe number of iterations
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=None,
    verbose=1
)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)

# Predict with best model
y_pred = random_search.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Evaluation Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Prices', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted Prices', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Ethereum Price (USD)')
plt.title('Ethereum Actual vs Predicted Prices (Enhanced XGBoost)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Enhanced ML analysis for cryptocurrency price prediction
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch market data
def fetch_market_data(coin_id, days=364):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["prices"], columns=["timestamp", f"{coin_id}_price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop("timestamp", axis=1, inplace=True)
    df.set_index("date", inplace=True)
    return df

# Fetch data
eth_data = fetch_market_data("ethereum")
btc_data = fetch_market_data("bitcoin")
combined_data = eth_data.join(btc_data, how='inner')

# Enhanced Feature Engineering
combined_data['eth_return'] = combined_data['ethereum_price'].pct_change()
combined_data['btc_return'] = combined_data['bitcoin_price'].pct_change()
combined_data['eth_log_return'] = np.log(combined_data['ethereum_price'] / combined_data['ethereum_price'].shift(1))
combined_data['btc_log_return'] = np.log(combined_data['bitcoin_price'] / combined_data['bitcoin_price'].shift(1))

# Technical indicators
combined_data['eth_rsi'] = 100 - (100 / (1 + combined_data['eth_return'].rolling(14).apply(lambda x: (x[x>0].sum()/-x[x<0].sum()), raw=True)))
combined_data['btc_rsi'] = 100 - (100 / (1 + combined_data['btc_return'].rolling(14).apply(lambda x: (x[x>0].sum()/-x[x<0].sum()), raw=True)))
combined_data['eth_macd'] = combined_data['ethereum_price'].ewm(span=12).mean() - combined_data['ethereum_price'].ewm(span=26).mean()
combined_data['btc_macd'] = combined_data['bitcoin_price'].ewm(span=12).mean() - combined_data['bitcoin_price'].ewm(span=26).mean()

# Volatility and moving averages
combined_data['eth_volatility_30'] = combined_data['eth_log_return'].rolling(30).std()
combined_data['btc_volatility_30'] = combined_data['btc_log_return'].rolling(30).std()
combined_data['eth_sma_30'] = combined_data['ethereum_price'].rolling(30).mean()
combined_data['btc_sma_30'] = combined_data['bitcoin_price'].rolling(30).mean()

# Lagged Features
for lag in [1, 3, 7, 14]:
    combined_data[f'eth_lag_{lag}'] = combined_data['ethereum_price'].shift(lag)
    combined_data[f'btc_lag_{lag}'] = combined_data['bitcoin_price'].shift(lag)

combined_data.dropna(inplace=True)

# Define features and target
features = [col for col in combined_data.columns if col not in ['ethereum_price']]
X = combined_data[features]
y = combined_data['ethereum_price']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time Series Cross-validation
split = int(len(X) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# Model and Hyperparameter tuning
ts_cv = TimeSeriesSplit(n_splits=5)
model = XGBRegressor(tree_method='hist', random_state=None)

#Model paramaters
param_dist = {
    "n_estimators": np.arange(100, 1001, 100),
    "max_depth": np.arange(3, 11),
    "learning_rate": np.linspace(0.01, 0.2, 20),
    "subsample": np.linspace(0.6, 1.0, 5),
    "colsample_bytree": np.linspace(0.6, 1.0, 5),
    "gamma": np.linspace(0, 1, 5),
    "reg_alpha": np.linspace(0, 1, 5),
    "reg_lambda": np.linspace(0, 1, 5),
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    scoring="neg_mean_squared_error",
    cv=ts_cv,
    n_jobs=-1,
    verbose=1,
    random_state=None
)

random_search.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = random_search.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Optimized Parameters:", random_search.best_params_)
print("\nModel Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")

# Feature Importance
plt.figure(figsize=(12, 8))
plot_importance(random_search.best_estimator_, max_num_features=15)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Predicted vs Actual Prices
plt.figure(figsize=(15, 8))
plt.plot(y_test.index, y_test, label='Actual Prices', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted Prices', linestyle='--', linewidth=2)
plt.title('Ethereum Prices: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
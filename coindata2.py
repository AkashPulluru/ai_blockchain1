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
    #url to access coin by market id - in this case, ethereum 
    url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
    
    params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Request failed: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("date", inplace=True)
    df.drop("timestamp", axis=1, inplace=True)
    return df

eth_data = fetch_eth_data()
eth_data.dropna(inplace=True)

# --- Mandelbrot-Inspired Feature Engineering (Considering Volatility Clustering and Heavy Tails) ---
eth_data["return"] = eth_data["price"].pct_change()
eth_data["log_return"] = np.log(eth_data["price"] / eth_data["price"].shift(1))

# Include fractal-like measures: volatility clustering
eth_data["rolling_volatility_7"] = eth_data["log_return"].rolling(window=7).std()
eth_data["rolling_volatility_30"] = eth_data["log_return"].rolling(window=30).std()

# Long-memory (long-range dependence) indicators
eth_data["return_squared"] = eth_data["return"]**2
eth_data["rolling_variance_30"] = eth_data["return_squared"].rolling(window=30).mean()

# Tail-risk proxies: extreme return indicators
eth_data["extreme_positive"] = (eth_data["return"] > eth_data["return"].quantile(0.95)).astype(int)
eth_data["extreme_negative"] = (eth_data["return"] < eth_data["return"].quantile(0.05)).astype(int)

# Moving averages and lags capturing memory in market
eth_data["EMA_12"] = eth_data["price"].ewm(span=12).mean()
eth_data["EMA_26"] = eth_data["price"].ewm(span=26).mean()

# Lagged variables to capture dependence
eth_data["lag_1"] = eth_data["price"].shift(1)
eth_data["lag_3"] = eth_data["price"].shift(3)
eth_data["lag_7"] = eth_data["price"].shift(7)

eth_data.dropna(inplace=True)

# Features selection to reflect Mandelbrot's insights
features = [
    "return",
    "log_return",
    "rolling_volatility_7",
    "rolling_volatility_30",
    "rolling_variance_30",
    "extreme_positive",
    "extreme_negative",
    "EMA_12",
    "EMA_26",
    "lag_1",
    "lag_3",
    "lag_7",
]

X = eth_data[features]
y = eth_data["price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# Model incorporating Mandelbrot's market intuition (long memory, volatility clustering)
model = XGBRegressor(tree_method="hist")

param_dist = {
    "n_estimators": np.arange(200, 1201, 50),
    "max_depth": np.arange(3, 12),
    "learning_rate": np.linspace(0.005, 0.05, 20),
    "subsample": np.linspace(0.5, 1.0, 11),
    "colsample_bytree": np.linspace(0.5, 1.0, 11),
    "gamma": np.linspace(0, 1, 11),
    "reg_alpha": np.linspace(0, 1, 11),
    "reg_lambda": np.linspace(0, 1, 11),
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=75,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Optimized Parameters:", random_search.best_params_)

y_pred = random_search.predict(X_test)

# Evaluation with an emphasis on tail risk (extremes)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")

# Plotting for visualizing volatility clustering
plt.figure(figsize=(16, 9))
plt.plot(y_test.index, y_test, label="Actual Prices", linewidth=2)
plt.plot(y_test.index, y_pred, label="Predicted Prices", linestyle="--", linewidth=2)
plt.title("Ethereum Prices: Actual vs Predicted with Mandelbrot-Inspired Features")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
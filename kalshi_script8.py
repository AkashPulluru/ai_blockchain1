import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb

# ---------------------------
# 1. Data Loading & Preparation
# ---------------------------

print("Loading dataset...")
data = pd.read_csv('KXMARMAD_trades.csv')
data['created_time'] = pd.to_datetime(data['created_time'], utc=True)

# ---------------------------
# 2. Deriving Market Outcomes
# ---------------------------

print("Deriving market outcomes...")
final_trade = data.sort_values('created_time').groupby('market_ticker').tail(1)
final_trade['yes_outcome'] = (final_trade['yes_price'] > final_trade['no_price']).astype(int)
data['yes_outcome'] = data['market_ticker'].map(final_trade.set_index('market_ticker')['yes_outcome'])

# ---------------------------
# 3. Feature Engineering & Aggregation
# ---------------------------

print("Engineering market-level features...")
market_features = data.groupby('market_ticker').agg({
    'yes_price': ['mean', 'std', 'min', 'max'],
    'no_price': ['mean', 'std', 'min', 'max'],
    'count': ['mean', 'sum'],
    'taker_side': lambda x: (x == 'yes').mean(),
    'yes_outcome': 'first'
})

market_features.columns = ['_'.join(col) for col in market_features.columns]
market_features.rename(columns={'yes_outcome_first': 'yes_outcome'}, inplace=True)

X = market_features.drop('yes_outcome', axis=1).fillna(0)
y = market_features['yes_outcome']

# ---------------------------
# 4. Train-Test Split
# ---------------------------

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------
# 5. Model Training
# ---------------------------

print("Training XGBoost model...")
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# ---------------------------
# 6. Model Evaluation
# ---------------------------

print("Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------
# 7. Backtesting Simulation (Simplified Example)
# ---------------------------

print("Running simplified backtest simulation...")
# Assume equal stake per market prediction
initial_capital = 10000
stake_per_bet = 100
num_bets = len(y_test)

# Simple strategy: bet on predicted outcome
results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'profit': stake_per_bet * ((y_test == y_pred).astype(int) - 0.5) * 2  # Win: +stake, Lose: -stake
})

total_profit = results['profit'].sum()
roi = total_profit / (stake_per_bet * num_bets)

print(f"Total Profit from Backtesting: ${total_profit:.2f}")
print(f"ROI: {roi:.2%}")

# ---------------------------
# 8. Preparing for Integration with Trading Bot (Kalshi API)
# ---------------------------

print("Model is trained and evaluated.")
print("Ready for integration with live trading via Kalshi API.")

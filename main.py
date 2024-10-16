import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

original_df = pd.read_csv('all_stocks_5yr.csv')
df = original_df[original_df['Name'] == 'AAL']
# data inspection
print(df.head())
print()
print(df.tail())
print()
print(df.info())
print()
print(df.describe())
print()
print(df.isnull().sum())
print()

# data cleaning
df = df.dropna()
df['date'] = pd.to_datetime(df['date'])
"""
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
"""
df.set_index('date', inplace=True)
df['price_change'] = df['close'] - df['open']
df['MA_5'] = df['close'].rolling(window=5).mean()
df['MA_10'] = df['close'].rolling(window=10).mean()
df['volatility'] = (df['high'] - df['low']) / df['low']
df['prev_close'] = df['close'].shift(1)
df = df.dropna()
print(df.head())
print()

# test if previous close price is correlated with current open price
# if so, remove from data
# correlation = df['prev_close'].corr(df['open'])
# print(correlation)
correlation, p_value = pearsonr(df['prev_close'], df['open'])
print(f'Correlation: {correlation}')
print(f'P-value: {p_value}')
print()
# p=0.0 significant, dont include prev_close

# data and target
X = df[['open', 'high', 'low', 'volume', 'price_change', 'MA_5', 'MA_10', 'volatility']]
y = df['close']

# model training
# training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # time-series split
print(X_test.head())
print()

# select model
# 1. XGBoost
params = {
    'alpha': 0.1,
    'lambda': 1,
    'learning_rate': 0.1,
    'max_depth': 9,
    'n_estimators': 300
}
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', alpha=0.1, learning_rate=0.1, max_depth=9, n_estimators=300
                             , reg_lambda=1)
model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)])
preds_xgb = model_xgb.predict(X_test)
mse = mean_squared_error(y_test, preds_xgb)
print(f'Mean Squared Error for XGB: {mse}')
print()

# Grid search for best model
param_grid = {
    'max_depth': [5, 7, 9],
    'learning_rate': [0.02, 0.06, 0.1],
    'n_estimators': [100, 200, 300],
    'alpha': [0, 0.1],  # L1 regularization
    'lambda': [0.1, 1]  # L2 regularization
}

"""
grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)  # Best Parameters: {'alpha': 0.1, 'lambda': 1, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 300}
print()
"""

# XGB plot Close Price over Time
fig1, ax1 = plt.subplots()
ax1.plot(df.index, df['close'], label='Close Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price')
plt.title('Close Price over Time')
plt.legend()
plt.show()

# XGB plot predicted vs actual
fig2, ax2 = plt.subplots()
ax2.plot(y_test.index, y_test, label='Actual Close Price')
ax2.plot(y_test.index, preds_xgb, label='Predict Close Price')
ax2.set_xlabel('Date')
ax2.set_ylabel('Close Price')
plt.title('XGB Actual vs Predict Close Price')
plt.legend()
plt.show()


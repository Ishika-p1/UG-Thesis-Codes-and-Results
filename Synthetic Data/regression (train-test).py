# =========================================================
# FULL OLS REGRESSION WITH PERMUTATION IMPORTANCE
# =========================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin
from math import sqrt

# -----------------------------
#       LOAD DATA
# -----------------------------
file_path = 'synthetic_dataset_pmm1000.csv'
data = pd.read_csv(file_path)

# Drop the 'Quarters' column if it exists
if 'Quarters' in data.columns:
    data = data.drop(columns=['Quarters'])

# Create a time index
data = data.reset_index(drop=True)
data['time'] = range(1, len(data) + 1)

# Define dependent and independent variables
Y = data['ExcessReturn']
X = data[['MarketReturn', 'SMB', 'HML', 'RMW', 'CMA', 'MOM','EPS', 'DeltaNetSales']]
X = sm.add_constant(X)

# -----------------------------
#       TIME-BASED SPLIT
# -----------------------------
max_time = data['time'].max()
split_time = int(max_time * 0.8)

train_idx = data['time'] <= split_time
test_idx = data['time'] > split_time

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# -----------------------------
#       FIT OLS MODEL
# -----------------------------
model = sm.OLS(Y_train, X_train).fit()

# -----------------------------
#       PREDICTIONS
# -----------------------------
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Save predictions and residuals
data.loc[train_idx, 'Predicted_Y_train'] = Y_train_pred
data.loc[train_idx, 'Residuals_train'] = Y_train - Y_train_pred
data.loc[test_idx,  'Predicted_Y_test'] = Y_test_pred
data.loc[test_idx,  'Residuals_test'] = Y_test - Y_test_pred

# -----------------------------
#       REGRESSION SUMMARY
# -----------------------------
print(model.summary())

# -----------------------------
#       ERROR METRICS
# -----------------------------
def MASE(y_train, y_true, y_pred):
    naive_errors = np.abs(np.diff(y_train))
    scale = np.mean(naive_errors)
    return np.mean(np.abs(y_true - y_pred)) / scale

mse_test = mean_squared_error(Y_test, Y_test_pred)
rmse_test = sqrt(mse_test)
mae_test = mean_absolute_error(Y_test, Y_test_pred)
medae_test = median_absolute_error(Y_test, Y_test_pred)
mase_test = MASE(Y_train.values, Y_test.values, Y_test_pred.values)
r2_test = r2_score(Y_test, Y_test_pred)

print("\nPerformance on Test Set:")
print(f"R2     : {r2_test:.6f}")
print(f"RMSE   : {rmse_test:.6f}")
print(f"MAE    : {mae_test:.6f}")
print(f"MedAE  : {medae_test:.6f}")
print(f"MASE   : {mase_test:.6f}")

# -----------------------------
# PERMUTATION IMPORTANCE
# -----------------------------
class StatsModelsWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model
    def fit(self, X, y=None):
        return self  # model is already fitted
    def predict(self, X):
        return self.model.predict(X)

wrapped_model = StatsModelsWrapper(model)

perm_result = permutation_importance(
    wrapped_model,
    X_test,
    Y_test,
    scoring='neg_root_mean_squared_error',  # RMSE
    n_repeats=20,
    random_state=42
)

perm_df = pd.DataFrame({
    'Variable': X_test.columns,
    'Importance_Mean': perm_result.importances_mean,
    'Importance_Std': perm_result.importances_std
}).sort_values(by='Importance_Mean', ascending=False)

print("\n================ PERMUTATION IMPORTANCE ==================")
print(perm_df.to_string(index=False))
print("===========================================================")

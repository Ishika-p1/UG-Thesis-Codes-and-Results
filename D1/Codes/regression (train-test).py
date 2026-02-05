import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from math import sqrt
import numpy as np

# Load the dataset
file_path = 'fama french dataset.csv'
data = pd.read_csv(file_path)

# Drop the 'Quarters' column if it exists
if 'Quarters' in data.columns:
    data = data.drop(columns=['Quarters'])

# Create a time index
data = data.reset_index(drop=True)
data['time'] = range(1, len(data) + 1)

# Define dependent and independent variables
Y = data['Excess return']
X = data[['market return', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]
X = sm.add_constant(X)

# Time-based split
max_time = data['time'].max()
split_time = int(max_time * 0.8)

train_idx = data['time'] <= split_time
test_idx = data['time'] > split_time

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# Fit model
model = sm.OLS(Y_train, X_train).fit()

# Predictions
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Save predictions
data.loc[train_idx, 'Predicted_Y_train'] = Y_train_pred
data.loc[train_idx, 'Residuals_train'] = Y_train - Y_train_pred
data.loc[test_idx,  'Predicted_Y_test'] = Y_test_pred
data.loc[test_idx,  'Residuals_test'] = Y_test - Y_test_pred

# Print regression summary
print(model.summary())

# -----------------------------
#       ERROR METRICS
# -----------------------------
# --- MASE ---
# Scale numerator using 1-step naive forecast errors from the training set
def MASE(y_train, y_true, y_pred):
    naive_errors = np.abs(np.diff(y_train))
    scale = np.mean(naive_errors)
    return np.mean(np.abs(y_true - y_pred)) / scale

# -----------------------------
#   TEST METRICS
# -----------------------------
mse_test = mean_squared_error(Y_test, Y_test_pred)
rmse_test = sqrt(mse_test)
mae_test = mean_absolute_error(Y_test, Y_test_pred)
medae_test = median_absolute_error(Y_test, Y_test_pred)

print("\nPerformance on Test Set:")
print(f"RMSE   : {rmse_test:.6f}")
print(f"MAE    : {mae_test:.6f}")
print(f"MedAE  : {medae_test:.6f}")
print(f"MASE   : {MASE(Y_train.values, Y_test.values, Y_test_pred.values):.6f}")
print(f"R2     : {r2_score(Y_test, Y_test_pred):.6f}")

# Optionally, if you want to output these to the DataFrame as well:
data.loc[test_idx, 'Predicted_Y_test'] = Y_test_pred
data.loc[test_idx, 'Residuals_test'] = Y_test - Y_test_pred

# -----------------------------
#   TRAINING SET METRICS
# -----------------------------
mse_train = mean_squared_error(Y_train, Y_train_pred)
rmse_train = sqrt(mse_train)
mae_train = mean_absolute_error(Y_train, Y_train_pred)
medae_train = median_absolute_error(Y_train, Y_train_pred)
mase_train = MASE(Y_train.values, Y_train.values, Y_train_pred.values)
r2_train = r2_score(Y_train, Y_train_pred)

print("\nPerformance on Training Set:")
print(f"R2     : {r2_train:.6f}")
print(f"RMSE   : {rmse_train:.6f}")
print(f"MAE    : {mae_train:.6f}")
print(f"MedAE  : {medae_train:.6f}")
print(f"MASE   : {mase_train:.6f}")


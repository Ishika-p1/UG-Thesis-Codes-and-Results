import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from math import sqrt
import numpy as np

# ===============================
# 1. LOAD DATA
# ===============================
file_path = 'complete dataset.csv'
data = pd.read_csv(file_path)

# Drop 'Quarters' if exists
if 'Quarters' in data.columns:
    data = data.drop(columns=['Quarters'])

# Create time index
data = data.reset_index(drop=True)
data['time'] = range(1, len(data) + 1)

# ===============================
# 2. DEFINE VARIABLES
# ===============================
Y = data['Excess return']
X = data[['market return', 'SMB', 'HML', 'RMW', 'CMA', 'MOM',
          'eps_diluted_aft_xtraord_item', 'HCE', 'delat_net sales', 'total_liability']]

# Add constant for intercept
X = sm.add_constant(X)

# ===============================
# 3. SAFE LOG FUNCTION
# ===============================
def safe_log(series):
    """
    Apply log transformation safely by shifting values if necessary.
    """
    min_val = series.min()
    shift = 1e-6 if min_val > 0 else abs(min_val) + 1e-6
    return np.log(series + shift)

# Apply log transformation
Y_log = safe_log(Y)
X_log = X.apply(safe_log)

# ===============================
# 4. TIME-BASED SPLIT
# ===============================
max_time = data['time'].max()
split_time = int(max_time * 0.8)

train_idx = data['time'] <= split_time
test_idx = data['time'] > split_time

X_train, X_test = X_log[train_idx], X_log[test_idx]
Y_train, Y_test = Y_log[train_idx], Y_log[test_idx]

# ===============================
# 5. FIT MODEL
# ===============================
model = sm.OLS(Y_train, X_train).fit()

# Predictions
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Save predictions and residuals
data.loc[train_idx, 'Predicted_Y_train'] = Y_train_pred
data.loc[train_idx, 'Residuals_train'] = Y_train - Y_train_pred
data.loc[test_idx,  'Predicted_Y_test'] = Y_test_pred
data.loc[test_idx,  'Residuals_test'] = Y_test - Y_test_pred

# ===============================
# 6. REGRESSION SUMMARY
# ===============================
print(model.summary())

# ===============================
# 7. ERROR METRICS
# ===============================
def MASE(y_train, y_true, y_pred):
    naive_errors = np.abs(np.diff(y_train))
    scale = np.mean(naive_errors)
    return np.mean(np.abs(y_true - y_pred)) / scale

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

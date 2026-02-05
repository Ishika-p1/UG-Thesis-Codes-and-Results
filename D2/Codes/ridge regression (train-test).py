import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from math import sqrt

# -----------------------------
#       LOAD DATA
# -----------------------------
file_path = "complete dataset.csv"
data = pd.read_csv(file_path)

# Drop the 'Quarters' column if it exists
if 'Quarters' in data.columns:
    data = data.drop(columns=['Quarters'])

# Create a time index
data = data.reset_index(drop=True)
data['time'] = range(1, len(data) + 1)

# Dependent & independent variables
Y = data['Excess return']
X = data[['market return', 'SMB', 'HML', 'RMW', 'CMA', 'MOM','eps_diluted_aft_xtraord_item', 'HCE', 'delat_net sales', 'total_liability']]

# -----------------------------
#      TRAIN/TEST SPLIT
# -----------------------------
max_time = data['time'].max()
split_time = int(max_time * 0.8)

train_idx = data['time'] <= split_time
test_idx  = data['time'] >  split_time

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# -----------------------------
#       RIDGE REGRESSION
# -----------------------------
# Cross-validated alpha selection
alphas = np.logspace(-4, 4, 100)

ridge_cv = RidgeCV(alphas=alphas, store_cv_values=False)
ridge_cv.fit(X_train, Y_train)

best_alpha = ridge_cv.alpha_
print(f"\nBest alpha selected by RidgeCV: {best_alpha}")

# Fit final Ridge model using best alpha
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train, Y_train)

# -----------------------------
#       PREDICTIONS
# -----------------------------
Y_train_pred = ridge_model.predict(X_train)
Y_test_pred = ridge_model.predict(X_test)

# Save predictions to DataFrame
data.loc[train_idx, 'Predicted_Y_train'] = Y_train_pred
data.loc[train_idx, 'Residuals_train'] = Y_train - Y_train_pred
data.loc[test_idx,  'Predicted_Y_test'] = Y_test_pred
data.loc[test_idx,  'Residuals_test'] = Y_test - Y_test_pred

# -----------------------------
#         COEFFICIENTS
# -----------------------------
print("\nRidge Regression Coefficients:")
coef_df = pd.DataFrame({
    'Variable': ['market return', 'SMB', 'HML', 'RMW', 'CMA', 'MOM','eps_diluted_aft_xtraord_item', 'HCE', 'delat_net sales', 'total_liability'],
    'Coefficient': ridge_model.coef_
})
print(coef_df)

print("\nIntercept:", ridge_model.intercept_)

# -----------------------------
#         ERROR METRICS
# -----------------------------
def MASE(y_train, y_true, y_pred):
    naive_errors = np.abs(np.diff(y_train))
    scale = np.mean(naive_errors)
    return np.mean(np.abs(y_true - y_pred)) / scale

# Test metrics
mse_test = mean_squared_error(Y_test, Y_test_pred)
rmse_test = sqrt(mse_test)
mae_test = mean_absolute_error(Y_test, Y_test_pred)
medae_test = median_absolute_error(Y_test, Y_test_pred)
mase_test = MASE(Y_train.values, Y_test.values, Y_test_pred)

# -----------------------------
#       PRINT RESULTS
# -----------------------------
print("\n------------------------------------")
print("    RIDGE MODEL PERFORMANCE (TEST)")
print("------------------------------------")
print(f"R2     : {r2_score(Y_test, Y_test_pred):.6f}")
print(f"RMSE   : {rmse_test:.6f}")
print(f"MAE    : {mae_test:.6f}")
print(f"MedAE  : {medae_test:.6f}")
print(f"MASE   : {mase_test:.6f}")

# -----------------------------
#     TRAINING METRICS
# -----------------------------
mse_train = mean_squared_error(Y_train, Y_train_pred)
rmse_train = sqrt(mse_train)
mae_train = mean_absolute_error(Y_train, Y_train_pred)
medae_train = median_absolute_error(Y_train, Y_train_pred)
mase_train = MASE(Y_train.values, Y_train.values, Y_train_pred)

print("\n------------------------------------")
print("  RIDGE MODEL PERFORMANCE (TRAIN)")
print("------------------------------------")
print(f"R2     : {r2_score(Y_train, Y_train_pred):.6f}")
print(f"RMSE   : {rmse_train:.6f}")
print(f"MAE    : {mae_train:.6f}")
print(f"MedAE  : {medae_train:.6f}")
print(f"MASE   : {mase_train:.6f}")


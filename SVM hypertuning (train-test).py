# ================================================
# SVM Model for Fama-French Factors (Time-Based Split)
# ================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# 1. Load dataset
df = pd.read_csv("fama french dataset.csv")

# 2. Create a time index column (1,2,3,...)
df = df.reset_index(drop=True)
df['time'] = range(1, len(df) + 1)

# 3. Define dependent + independent variables
X = df[['market return', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]
y = df['Excess return']

# 4. TIME-BASED SPLIT (80% train, 20% test)
max_time = df['time'].max()
split_time = int(max_time * 0.8)

train_idx = df['time'] <= split_time
test_idx  = df['time'] > split_time

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"\nTraining rows: {len(X_train)}, Testing rows: {len(X_test)}")

# 5. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Base SVR model
svr = SVR()

# 7. Hyperparameter tuning
param_grid = {
    'kernel': ['rbf', 'linear', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    svr, param_grid, cv=5, scoring='r2',
    verbose=1, n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# 8. Best model
print("\nBest Parameters:")
print(grid_search.best_params_)
best_svr = grid_search.best_estimator_

# 9. Predictions
y_pred_train = best_svr.predict(X_train_scaled)
y_pred_test = best_svr.predict(X_test_scaled)

# -------------------------------------------------------
#     ERROR METRICS FOR TEST SET: RMSE, MAE, MedAE, MASE
# -------------------------------------------------------

def MASE(y_train, y_test, y_pred_test):
    """Mean Absolute Scaled Error (Hyndman)."""
    naive_forecast_errors = np.abs(np.diff(y_train))  # |Y_t - Y_(t-1)|
    scale = naive_forecast_errors.mean()
    return np.mean(np.abs(y_test - y_pred_test)) / scale

# Compute metrics
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
medae_test = median_absolute_error(y_test, y_pred_test)
mase_test = MASE(y_train.values, y_test.values, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\nModel Performance:")
print(f"Test R²   : {r2_test:.4f}")
print(f"Test RMSE : {rmse_test:.4f}")
print(f"Test MAE  : {mae_test:.4f}")
print(f"Test MedAE: {medae_test:.4f}")
print(f"Test MASE : {mase_test:.4f}")

# ---------------- Feature Importance (Linear SVM) --------------
if best_svr.kernel == 'linear':
    # Absolute value of coefficients to get magnitude of importance
    coef_importance = pd.Series(best_svr.coef_.flatten(), index=X.columns)
    
    # Sort descending by absolute value
    coef_importance_sorted = coef_importance.abs().sort_values(ascending=False)
    
    print("\nFeature Importance (Descending Order):")
    print(coef_importance_sorted)
else:
    print("\nFeature importance unavailable for non-linear kernels.")

# ----------------------------- SHAP Analysis -----------------------------
if best_svr.kernel == 'linear':
    explainer = shap.LinearExplainer(best_svr, X_train_scaled, feature_perturbation="interventional")
else:
    background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], min(100, X_train_scaled.shape[0]), replace=False)]
    explainer = shap.KernelExplainer(best_svr.predict, background)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test_scaled)

# Convert SHAP values to DataFrame
shap_df = pd.DataFrame(shap_values, columns=X.columns)

# Print baseline / expected value
print("\nSHAP Baseline (Expected Value):")
print(explainer.expected_value)

# Compute mean absolute SHAP values for feature importance ranking
mean_shap = shap_df.abs().mean().sort_values(ascending=False)
print("\nMean Absolute SHAP Value (Feature Importance Ranking):")
print(mean_shap)

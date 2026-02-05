# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import StandardScaler
import shap

# Load the dataset
df = pd.read_csv("fama french dataset.csv")

# Define feature variables (independent variables) and target variable (dependent variable)
X = df[['market return', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]
y = df['Excess return']

# Optional: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)

# Define cross-validation strategy (5-fold)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validated predictions
y_pred_cv = cross_val_predict(regressor, X_scaled, y, cv=cv)

# ----------------------------------------------
# Error Metrics
# ----------------------------------------------

# MSE, RMSE, MAE
mse = mean_squared_error(y, y_pred_cv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred_cv)

# MedAE (Median Absolute Error)
medae = median_absolute_error(y, y_pred_cv)

# MASE (Mean Absolute Scaled Error)
# Naive forecast: y_{t-1}
naive_forecast = y.shift(1).dropna()

# Align y_true and predictions to naive forecast index
mase_y_true = y.iloc[1:]
mase_y_pred = y_pred_cv[1:]

mase = mae / mean_absolute_error(mase_y_true, naive_forecast)

# ----------------------------------------------
# Print performance metrics
# ----------------------------------------------
print("Model Performance Metrics (5-Fold Cross-Validation):")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Median Absolute Error (MedAE): {medae:.6f}")
print(f"Mean Absolute Scaled Error (MASE): {mase:.6f}")
print(f"R-squared (R²): {r2_score(y, y_pred_cv):.4f}\n")

# Retrain model on full data for feature importance
regressor.fit(X_scaled, y)

# Display feature importance
print("Feature Importances from Decision Tree:")
for feature, importance in zip(X.columns, regressor.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# -----------------------------
# SHAP Analysis (Print Format)
# -----------------------------

explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(X_scaled)

shap_df = pd.DataFrame(shap_values, columns=X.columns)

# Print baseline / expected value
print("\nSHAP Baseline (Expected Value):")
print(explainer.expected_value)

# Mean absolute SHAP values
mean_shap = shap_df.abs().mean().sort_values(ascending=False)
print("\nMean Absolute SHAP Value (Feature Importance Ranking):")
print(mean_shap)

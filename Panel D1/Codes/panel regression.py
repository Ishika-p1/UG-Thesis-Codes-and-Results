import pandas as pd
import numpy as np
from linearmodels import PanelOLS
from sklearn.metrics import mean_absolute_error, median_absolute_error

# ----------------------------------------------------
# 1️⃣ LOAD DATA
# ----------------------------------------------------
df = pd.read_csv("panel data.csv")

# Remove 'Quarters' column if present
if 'Quarters' in df.columns:
    df = df.drop(columns=['Quarters'])

# ----------------------------------------------------
# 2️⃣ CREATE ARTIFICIAL TIME INDEX
# ----------------------------------------------------
df['time'] = range(1, len(df) + 1)

# ----------------------------------------------------
# 3️⃣ RESHAPE WIDE → LONG PANEL FORMAT
# ----------------------------------------------------
factor_cols = ['market return', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
stock_cols = [c for c in df.columns if c not in factor_cols + ['time']]

long_df = df.melt(id_vars=['time'] + factor_cols,
                  value_vars=stock_cols,
                  var_name='Stock',
                  value_name='Return')

# Set MultiIndex (required by linearmodels)
long_df = long_df.set_index(['Stock', 'time'])

# ----------------------------------------------------
# 4️⃣ PANEL REGRESSION (FIXED EFFECTS)
# ----------------------------------------------------
exog = long_df[factor_cols].copy()
exog = exog.assign(constant=1)  # add intercept

mod = PanelOLS(
    long_df['Return'],
    exog,
    entity_effects=True   # stock fixed effects
)

res = mod.fit(cov_type='clustered', cluster_entity=True)

# ----------------------------------------------------
# 5️⃣ PREDICTIONS
# ----------------------------------------------------
# Use .values to avoid MultiIndex alignment issues
y_true_vals = long_df['Return'].values
y_pred_vals = res.predict(exog).values

# ----------------------------------------------------
# 6️⃣ PERFORMANCE METRICS
# ----------------------------------------------------

mse = np.mean((y_true_vals - y_pred_vals) ** 2)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_vals, y_pred_vals)
medae = median_absolute_error(y_true_vals, y_pred_vals)

# Calculate MASE
# For MASE, we need the in-sample naive forecast (lagged values)
# Since data is panel, we do this per entity (Stock)
# We'll compute the mean absolute difference of actuals for each stock
# and then compute the scaled MAE

# Prepare a DataFrame to compute in-sample naive errors
df_in_sample = long_df.copy()
df_in_sample['y_true'] = df_in_sample['Return']
df_in_sample['y_lag'] = df_in_sample.groupby('Stock')['Return'].shift(1)

# Drop rows where lag is NaN
df_lagged = df_in_sample.dropna(subset=['y_lag'])

# Calculate absolute differences of actuals (for scaling)
abs_actual_diff = np.abs(df_lagged['y_true'] - df_lagged['y_lag'])
mean_abs_actual_diff = abs_actual_diff.mean()

# Calculate the absolute errors of the predictions
abs_errors = np.abs(y_true_vals - y_pred_vals)
# Compute MASE
mase = abs_errors.mean() / mean_abs_actual_diff

# ----------------------------------------------------
# 7️⃣ OUTPUT
# ----------------------------------------------------
print("\n=== Full Panel Regression Summary ===")
print(res)

print("\n=== Performance Metrics ===")
print(f"RMSE  : {rmse:.6f}")
print(f"MAE   : {mae:.6f}")
print(f"MedAE : {medae:.6f}")
print(f"MASE  : {mase:.6f}")

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from math import sqrt
import numpy as np

# Load the dataset
file_path = "complete dataset.csv"
data = pd.read_csv(file_path)

# Define dependent and independent variables
Y = data['Excess return']
X = data[['market return', 'SMB', 'HML', 'RMW', 'CMA', 'MOM','eps_diluted_aft_xtraord_item', 'delat_net sales']]
#'eps_diluted_aft_xtraord_item', 'HCE', 'prov_contingencies', 'delat_net sales', "Shareholder's fund", 'Investment Ratio', 'Liquidity Ratio', 'total_income', 'ROA', 'total_liability', 'total_exp']

# Add constant for intercept
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(Y, X).fit()

# Predictions and residuals
data['Predicted Y'] = model.predict(X)
data['Residuals'] = Y - data['Predicted Y']

# ---- ERROR METRICS ----

# RMSE
rmse = sqrt(mean_squared_error(Y, data['Predicted Y']))

# MAE
mae = mean_absolute_error(Y, data['Predicted Y'])

# MedAE (Median Absolute Error)
medae = median_absolute_error(Y, data['Predicted Y'])

# MASE (Mean Absolute Scaled Error)
# Using naive 1-step forecast (y_t-1)
naive_forecast = Y.shift(1).dropna()
mase = mae / np.mean(np.abs(Y[1:] - naive_forecast))

print("\nModel Error Metrics:")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"MedAE: {medae:.6f}")
print(f"MASE : {mase:.6f}")

# Print regression summary
print(model.summary())

import pandas as pd
import numpy as np
from linearmodels import PanelOLS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

# ----------------------------------------------------
# 1️⃣ LOAD DATA
# ----------------------------------------------------
df = pd.read_csv("panel data.csv")

# Remove Quarter column if exists
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

long_df = df.melt(
    id_vars=['time'] + factor_cols,
    value_vars=stock_cols,
    var_name='Stock',
    value_name='Return'
)

# Set MultiIndex (entity, time)
long_df = long_df.set_index(['Stock', 'time'])

# ----------------------------------------------------
# 4️⃣ TIME-AWARE TRAIN-TEST SPLIT (80/20)
# ----------------------------------------------------
times = long_df.index.get_level_values('time').unique()
cutoff_idx = int(len(times) * 0.8)

train_times = times[:cutoff_idx]
test_times = times[cutoff_idx:]

train_df = long_df.loc[long_df.index.get_level_values('time').isin(train_times)]
test_df = long_df.loc[long_df.index.get_level_values('time').isin(test_times)]

# ----------------------------------------------------
# 5️⃣ PREPARE EXOG VARIABLES
# ----------------------------------------------------
def prepare_exog(df):
    exog = df[factor_cols]
    exog = exog.assign(constant=1)  # add intercept
    return exog

exog_train = prepare_exog(train_df)
exog_test = prepare_exog(test_df)

# ----------------------------------------------------
# 6️⃣ FIT PANELOLS MODEL ON TRAIN SET
# ----------------------------------------------------
model = PanelOLS(train_df['Return'], exog_train, entity_effects=True)
res = model.fit(cov_type='clustered', cluster_entity=True)

print("\n=== Training Set Regression Summary ===")
print(res)

# ----------------------------------------------------
# 7️⃣ PREDICT ON TEST SET
# ----------------------------------------------------
params = res.params
X_test = exog_test[params.index]  # align columns
y_true = test_df['Return'].values
y_pred = (X_test @ params).values

# ----------------------------------------------------
# 8️⃣ PERFORMANCE METRICS
# ----------------------------------------------------
# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAE
mae = mean_absolute_error(y_true, y_pred)

# MedAE
medae = median_absolute_error(y_true, y_pred)

# MASE (using first stock's train returns for naive scale)
first_stock_train = train_df.index.get_level_values('Stock')[0]
y_train = train_df.loc[first_stock_train]['Return'].values
def MASE(y_true, y_pred, y_train):
    naive_errors = np.abs(np.diff(y_train))
    scale = naive_errors.mean()
    return np.mean(np.abs(y_true - y_pred)) / scale

mase = MASE(y_true, y_pred, y_train)

# R-squared
r2 = r2_score(y_true, y_pred)

# ----------------------------------------------------
# 9️⃣ OUTPUT TEST PERFORMANCE
# ----------------------------------------------------
print("\n=== Test Set Performance ===")
print(f"R²    : {r2:.6f}")
print(f"RMSE  : {rmse:.6f}")
print(f"MAE   : {mae:.6f}")
print(f"MedAE : {medae:.6f}")
print(f"MASE  : {mase:.6f}")

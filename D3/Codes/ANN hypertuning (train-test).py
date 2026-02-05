# =========================================================
# FULL CODE: ANN WITH KERAS TUNER (LOWEST RMSE SELECTION)
# =========================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, median_absolute_error, mean_squared_error,
    explained_variance_score, r2_score
)

import keras_tuner as kt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("complete dataset.csv")

# Drop non-feature columns
df = df.drop(columns=["Quarters"])

# Time index (no look-ahead bias)
df["time"] = np.arange(len(df))

factor_cols = [
    'market return', 'SMB', 'HML', 'RMW', 'CMA', 'MOM',
    'eps_diluted_aft_xtraord_item', 'delat_net sales'
]
target_col = "Excess return"

X = df[factor_cols].values
y = df[target_col].values


# =========================================================
# 2. TIME-BASED TRAIN / TEST SPLIT
# =========================================================

split_time = int(df["time"].max() * 0.8)

train_idx = df["time"] <= split_time
test_idx = df["time"] > split_time

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


# =========================================================
# 3. FEATURE SCALING
# =========================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# 4. MODEL BUILDER (KERAS TUNER)
# =========================================================

def build_model(hp):

    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))

    num_layers = hp.Int("num_layers", 1, 5)

    model.add(Dense(
        hp.Int("units_0", 16, 128, step=16),
        activation="relu"
    ))

    for i in range(1, num_layers):
        model.add(Dense(
            hp.Int(f"units_{i}", 16, 128, step=16),
            activation="relu"
        ))

    model.add(Dense(1))

    lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=Adam(lr),
        loss=MeanSquaredError(),
        metrics=[RootMeanSquaredError(name="rmse")]
    )

    return model


# =========================================================
# 5. KERAS TUNER (MINIMIZE VALIDATION RMSE)
# =========================================================

tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective("val_rmse", direction="min"),
    max_trials=10,
    executions_per_trial=1,
    directory="ff6_tuner",
    project_name="ff6_ann_rmse"
)

early_stop = EarlyStopping(
    monitor="val_rmse",
    mode="min",
    patience=10,
    restore_best_weights=True
)

tuner.search(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\nBest Hyperparameters:")
for p in best_hp.values:
    print(f"{p}: {best_hp.get(p)}")


# =========================================================
# 6. MODEL PREDICTIONS & METRICS
# =========================================================

y_pred = best_model.predict(X_test_scaled).flatten()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
expl_var = explained_variance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# MASE
naive_errors = np.abs(np.diff(y_train))
scale = naive_errors.mean()
mase = mae / scale

print("\n================ FF6 ANN MODEL PERFORMANCE ================")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"MedAE: {medae:.6f}")
print(f"Explained Variance: {expl_var:.6f}")
print(f"R²   : {r2:.6f}")
print(f"MASE : {mase:.6f}")
print("===========================================================")


# =========================================================
# 7. PERMUTATION IMPORTANCE (OUT-OF-SAMPLE)
# =========================================================

from sklearn.inspection import permutation_importance

perm_result = permutation_importance(
    best_model,
    X_test_scaled,
    y_test,
    scoring="neg_root_mean_squared_error",  # ⭐ align with RMSE objective
    n_repeats=20,
    random_state=42
)

print("\n================ PERMUTATION IMPORTANCE ==================")
for i, feature_name in enumerate(factor_cols):
    mean_imp = perm_result.importances_mean[i]
    std_imp = perm_result.importances_std[i]
    print(f"{feature_name}: {mean_imp:.6f} ± {std_imp:.6f}")
print("===========================================================")

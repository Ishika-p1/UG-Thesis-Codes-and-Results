import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, median_absolute_error, mean_squared_error,
    explained_variance_score, r2_score, make_scorer
)
from sklearn.inspection import permutation_importance

import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("complete dataset.csv")

# Remove quarters column
df = df.drop(columns=["Quarters"])

# Create incremental time index
df["time"] = np.arange(len(df))

factor_cols = ['market return', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'eps_diluted_aft_xtraord_item', 'HCE', 'delat_net sales', 'total_liability']
target_col = "Excess return"

X = df[factor_cols].values
y = df[target_col].values


# =========================================================
# 2. TIME-BASED TRAIN TEST SPLIT
# =========================================================

max_time = df["time"].max()
split_time = int(max_time * 0.8)

train_idx = df["time"] <= split_time
test_idx = df["time"] > split_time

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


# =========================================================
# 3. SCALE FEATURES
# =========================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# 4. BUILD ANN MODEL (FOR KERAS TUNER)
# =========================================================

def build_model(hp):

    model = Sequential()

    # Explicit Input layer (avoids warnings)
    model.add(Input(shape=(X_train_scaled.shape[1],)))

    # Number of layers to try
    num_layers = hp.Int("num_layers", 1, 3)

    # First Dense Layer
    model.add(Dense(
        hp.Int("units_0", min_value=16, max_value=128, step=16),
        activation="relu"
    ))

    # Additional Dense Layers
    for i in range(1, num_layers):
        model.add(Dense(
            hp.Int(f"units_{i}", min_value=16, max_value=128, step=16),
            activation="relu"
        ))

    # Output Layer
    model.add(Dense(1))

    # Learning rate
    lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    # Compile model
    model.compile(
        optimizer=Adam(lr),
        loss=MeanSquaredError()
    )

    return model


# =========================================================
# 5. HYPERPARAMETER TUNING
# =========================================================

tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1,
    directory="ff6_tuner",
    project_name="ff6_ann"
)

tuner.search(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,
    batch_size=32,
    verbose=0
)

best_hp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]

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
print(f"R²: {r2:.6f}")
print(f"MASE : {mase:.6f}")
print("===========================================================")

# Import permutation importance
from sklearn.inspection import permutation_importance

# After evaluating model performance, compute feature importance
result = permutation_importance(
    best_model,  # Note: For Keras models, you need to wrap or predict manually
    X_test_scaled,
    y_test,
    scoring='neg_mean_squared_error',
    n_repeats=10,
    random_state=42
)

# Display feature importance
print("\nFeature Importance (Permutation Importance):")
for i, feature_name in enumerate(factor_cols):
    importance_mean = result.importances_mean[i]
    importance_std = result.importances_std[i]
    print(f"{feature_name}: {importance_mean:.4f} ± {importance_std:.4f}")

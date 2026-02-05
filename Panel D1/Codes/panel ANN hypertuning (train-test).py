import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, median_absolute_error, max_error, explained_variance_score, r2_score
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Load data
df = pd.read_csv("panel data.csv")
df = df.rename(columns={'Quarters': 'Quarter'})
df['time'] = range(1, len(df) + 1)

factor_cols = ['market return', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
stock_cols = [c for c in df.columns if c not in ['Quarter', 'time'] + factor_cols]

# Wide → long
long_df = df.melt(
    id_vars=['time'] + factor_cols,
    value_vars=stock_cols,
    var_name='Stock',
    value_name='Return'
)

# Encode stocks
le = LabelEncoder()
long_df['Stock_ID'] = le.fit_transform(long_df['Stock'])

X_factors = long_df[factor_cols].values
X_stock = long_df['Stock_ID'].values
y = long_df['Return'].values

# Scale factors
scaler = StandardScaler()
X_factors = scaler.fit_transform(X_factors)

# Train-test split
max_time = long_df['time'].max()
split_time = int(max_time * 0.8)
train_idx = long_df['time'] <= split_time
test_idx  = long_df['time'] > split_time

Xf_train, Xf_test = X_factors[train_idx], X_factors[test_idx]
Xs_train, Xs_test = X_stock[train_idx], X_stock[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

num_stocks = long_df['Stock_ID'].nunique()

# Define model builder for hyperparameter tuning
def build_model(hp):
    embedding_dim = hp.Int('embedding_dim', min_value=2, max_value=16, step=2)
    num_layers = hp.Int('num_layers', 1, 3)
    units_options = [hp.Int(f'units_{i}', min_value=8, max_value=64, step=8) for i in range(num_layers)]
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    
    stock_input = Input(shape=(1,))
    stock_emb = Embedding(num_stocks, embedding_dim)(stock_input)
    stock_emb = Flatten()(stock_emb)
    factor_input = Input(shape=(len(factor_cols),))
    x = Concatenate()([stock_emb, factor_input])
    
    for i in range(num_layers):
        x = Dense(units_options[i], activation='relu')(x)
    output = Dense(1)(x)
    
    model = Model([stock_input, factor_input], output)
    model.compile(optimizer=Adam(learning_rate), loss=MeanSquaredError())
    return model

# Hyperparameter tuning with Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='kt_tuner',
    project_name='stock_embedding_ann_tuning'
)

# Run the search
tuner.search([Xs_train, Xf_train], y_train,
             validation_data=([Xs_test, Xf_test], y_test),
             epochs=50,
             batch_size=32,
             verbose=0)

# Retrieve the best trial and hyperparameters
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
best_hp = best_trial.hyperparameters

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Print best hyperparameters
print("\nBest Hyperparameters Found:")
for param in best_hp.values:
    print(f"{param}: {best_hp.get(param)}")

# Predictions with best model
y_pred = best_model.predict([Xs_test, Xf_test]).flatten()

# Metrics
mse = np.mean((y_test - y_pred)**2)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
maxerr = max_error(y_test, y_pred)
expl_var = explained_variance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate MASE
# Use the first stock in training set for naive forecast scale
first_stock_in_train = le.classes_[0]
train_stock_data = long_df[(long_df['Stock'] == first_stock_in_train) & (long_df['time'] <= split_time)]
y_train_stock = train_stock_data['Return'].values

# Naive forecast errors (lagged differences)
naive_errors = np.abs(np.diff(y_train_stock))
scale = naive_errors.mean()

# Compute MASE
mase = np.mean(np.abs(y_test - y_pred)) / scale

# Output results
print("\n================ ANN MODEL PERFORMANCE (TUNED) =================")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"MedAE: {medae:.6f}")
print(f"Max Error: {maxerr:.6f}")
print(f"Explained Variance: {expl_var:.6f}")
print(f"R^2: {r2:.6f}")
print(f"MASE : {mase:.6f}")
print("=========================================================")

from sklearn.utils import shuffle

def permutation_importance(model, Xf, Xs, y_true, metric=mean_absolute_error, n_repeats=5):
    """
    Compute permutation importance for each factor feature.
    Note: Stock embeddings are harder to interpret directly as importance, 
    so we mainly focus on factor_cols here.
    """
    baseline = metric(y_true, model.predict([Xs, Xf]).flatten())
    importances = {}

    for i, col in enumerate(factor_cols):
        scores = []
        Xf_copy = Xf.copy()
        for _ in range(n_repeats):
            Xf_copy[:, i] = shuffle(Xf_copy[:, i])
            y_pred = model.predict([Xs, Xf_copy]).flatten()
            scores.append(metric(y_true, y_pred))
        # Importance = increase in error
        importances[col] = np.mean(scores) - baseline

    return importances

# Compute permutation importance
perm_importance = permutation_importance(best_model, Xf_test, Xs_test, y_test, metric=mean_absolute_error)

# Print results
print("\nPermutation Feature Importance (based on MAE increase):")
for feature, imp in perm_importance.items():
    print(f"{feature}: {imp:.6f}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import optuna

# --- Load datasets ---
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# --- Preprocessing ---
y = train_df['HotelValue']
train_ids = train_df['Id']
test_ids = test_df['Id']

train_df = train_df.drop(columns=['Id', 'HotelValue'])
test_df = test_df.drop(columns=['Id'])

combined_df = pd.concat([train_df, test_df], axis=0, sort=False)
categorical_features_indices = combined_df.select_dtypes(include='object').columns.tolist()

for col in categorical_features_indices:
    combined_df[col] = combined_df[col].fillna("MISSING")

numerical_cols = combined_df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    combined_df[col] = combined_df[col].fillna(combined_df[col].median())

X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

# --- Train/Validation split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 2000, 15000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),  # allow shallower and deeper trees
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # stochastic sampling
        'rsm': trial.suggest_float('rsm', 0.6, 1.0),  # random subspace for features
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 2),
        'random_strength': trial.suggest_float('random_strength', 0, 2),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': 0,
        'loss_function': 'RMSE',
        'random_seed': 42,
        'thread_count': -1,
        'cat_features': categorical_features_indices
    }

    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=500,
        verbose=False
    )

    preds = model.predict(X_val)
    rmse = np.sqrt( mean_squared_error(y_val, preds) )  # proper RMSE
    return rmse

# --- Run Optuna study ---
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500, show_progress_bar=True)

print("Best trial:")
trial = study.best_trial
print(trial.params)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

print(f"XGBoost version: {xgb.__version__}")

# Load Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Prepare data
y = train_df['HotelValue']

# Combine for preprocessing
combined_df = pd.concat([train_df.drop(columns=['Id', 'HotelValue']), test_df.drop(columns=['Id'])], axis=0, sort=False)

# Preprocessing
categorical_features = combined_df.select_dtypes(include='object').columns.tolist()
combined_df = pd.get_dummies(combined_df, columns=categorical_features, dummy_na=True)

numerical_features = combined_df.select_dtypes(include=np.number).columns.tolist()
for col in numerical_features:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)

X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]
X_test = X_test[X.columns]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting XGBoost debug run...")

try:
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, seed=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[xgb.callback.EarlyStopping(rounds=10)], verbose=False)
    print("\nSUCCESS: XGBoost ran with the 'callbacks' parameter.")
except TypeError as e:
    print(f"\nERROR with 'callbacks': {e}")
    print("\nAttempting fallback with 'early_stopping_rounds'...")
    try:
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, seed=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        print("\nSUCCESS: XGBoost ran with the 'early_stopping_rounds' parameter.")
    except TypeError as e2:
        print(f"\nERROR with 'early_stopping_rounds': {e2}")
        print("\nCould not find a valid method for early stopping.")


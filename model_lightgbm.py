import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import lightgbm
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from PreProcessing.preprocessing import save_model_results

# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# --- Preprocessing ---

# Separate target variable
y = train_df['HotelValue']
train_ids = train_df['Id']
test_ids = test_df['Id']
train_df = train_df.drop(columns=['Id', 'HotelValue'])
test_df = test_df.drop(columns=['Id'])

# Identify categorical features
categorical_features_indices = np.where(train_df.dtypes == 'object')[0]

# Handle Missing Values
# Numerical columns: fill with the median
for col in train_df.select_dtypes(include=np.number).columns:
    if train_df[col].isnull().any():
        train_df[col] = train_df[col].fillna(train_df[col].median())
    if test_df[col].isnull().any():
        test_df[col] = test_df[col].fillna(test_df[col].median())

# Categorical columns: fill with a specific value like 'None'
for col in train_df.select_dtypes(include='object').columns:
    if train_df[col].isnull().any():
        train_df[col] = train_df[col].fillna('None')
    if test_df[col].isnull().any():
        test_df[col] = test_df[col].fillna('None')


# Convert categorical columns to 'category' dtype for LightGBM
for col in train_df.select_dtypes(include='object').columns:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')


# --- Model Training and Evaluation ---

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(train_df, y, test_size=0.2, random_state=42)

# Initialize and train the LGBMRegressor model
model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.3,
    random_state=42,
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='l2', # l2 is MSE, l1 is MAE
          callbacks=[lightgbm.early_stopping(50, verbose=False)])

# The best iteration found by LightGBM
print(f"Best iteration: {model.best_iteration_}")

# Make predictions on the validation set and calculate error
val_predictions = model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)
val_r2 = model.score(X_val, y_val)

# Calculate training scores
train_r2 = model.score(X_train, y_train)

print(f"Training R-squared: {train_r2}")
print(f"Validation R-squared: {val_r2}")
print(f"Validation Mean Squared Error: {val_mse}")
val_rmse = np.sqrt(val_mse)
save_model_results(os.path.basename(__file__), 'LGBMRegressor', val_rmse)


# --- Prediction and Submission ---

# Predict on the actual test data
test_predictions = model.predict(test_df)

# Create the submission file
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
submission_df.to_csv('submission_lightgbm.csv', index=False)

print("Submission file 'submission_lightgbm.csv' created successfully.")

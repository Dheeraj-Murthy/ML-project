import pandas as pd
from sklearn.model_selection import train_test_split
# Updated import to use ElasticNetCV for combined L1/L2 regularization
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the same directory.")
    exit()

# --- Preprocessing and Feature Engineering ---

# Separate target variable
y = train_df['HotelValue']
train_ids = train_df['Id']
test_ids = test_df['Id']
train_df = train_df.drop(columns=['Id', 'HotelValue'])
test_df = test_df.drop(columns=['Id'])

# Combine for consistent processing
combined_df = pd.concat([train_df, test_df], axis=0, sort=False)

# Handle Missing Values
# Numerical columns: fill with the median
for col in combined_df.select_dtypes(include=np.number).columns:
    if combined_df[col].isnull().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

# Categorical columns: fill with the mode
for col in combined_df.select_dtypes(include='object').columns:
    if combined_df[col].isnull().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

# One-Hot Encode Categorical Features
combined_df = pd.get_dummies(combined_df, drop_first=True)

# Separate back into training and testing sets
X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

# Align columns after one-hot encoding
train_cols = X.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X[c] = 0

X_test = X_test[train_cols]

# --- Feature Scaling (CRITICAL for Regularized Linear Models) ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# --- Model Training and Evaluation (ElasticNetCV) ---

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define ranges for cross-validation:
# l1_ratio: 0.0 is pure L2 (Ridge), 1.0 is pure L1 (LASSO), 0.5 is equal mix
l1_ratios = [0.0, 0.1, 0.5, 0.9, 1.0]
# alphas: A good range of regularization strength values
alphas = [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]

# Initialize and train the ElasticNetCV model
# cv=5 sets up 5-fold cross-validation
# Added max_iter=10000 to resolve ConvergenceWarning
model = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=5, random_state=42, n_jobs=-1, max_iter=10000)
model.fit(X_train, y_train)

print("--- ElasticNetCV (L1 + L2) Results ---")
print(f"Best alpha found: {model.alpha_}")
print(f"Best L1 Ratio found (0.0=L2, 1.0=L1): {model.l1_ratio_}")

# Make predictions on the validation set and calculate error
val_predictions = model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)

# Calculate scores
train_r2 = model.score(X_train, y_train)
val_r2 = model.score(X_val, y_val)

print(f"Training R-squared: {train_r2}")
print(f"Validation R-squared: {val_r2}")
print(f"Validation Mean Squared Error: {val_mse}")


# --- Prediction and Submission ---

# Predict on the actual test data
test_predictions = model.predict(X_test_scaled)

# Create the submission file
submission_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_predictions})
submission_df.to_csv('submission_elastic_net.csv', index=False)

print("Submission file 'submission_elastic_net.csv' created successfully.")

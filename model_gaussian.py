import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- Load Data ---
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

y = train_df["HotelValue"]
train_ids = train_df["Id"]
test_ids = test_df["Id"]

X = train_df.drop(columns=["Id", "HotelValue"])
X_test = test_df.drop(columns=["Id"])

# --- Handle Missing Values ---
for df in [X, X_test]:
    for col in df.select_dtypes(include=np.number):
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].fillna(df[col].mode()[0])

# --- One-hot encode categoricals ---
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

# --- Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# --- Split for validation ---
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Kernel Definition ---
# RBF for smooth trends + WhiteKernel for noise
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1)

# --- Model ---
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-10, normalize_y=True)
gpr.fit(X_train, y_train)

print("Optimized Kernel:", gpr.kernel_)

# --- Predict and Evaluate ---
y_pred, y_std = gpr.predict(X_val, return_std=True)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")
print(f"Mean uncertainty (σ): {np.mean(y_std):.4f}")

# --- Plot ---
plt.figure(figsize=(10,6))
plt.errorbar(range(len(y_pred)), y_pred, yerr=2*y_std, fmt='o', alpha=0.4, label='Predicted ±2σ')
plt.scatter(range(len(y_val)), y_val, color='red', alpha=0.5, label='True Values')
plt.legend()
plt.title("Gaussian Process Regression Predictions with Uncertainty")
plt.show()

# --- Predict Test ---
test_pred, test_std = gpr.predict(X_test_scaled, return_std=True)

# --- Submission ---
submission = pd.DataFrame({
    "Id": test_ids,
    "HotelValue": test_pred,
    "Uncertainty": test_std
})
submission.to_csv("submission_gaussian_process.csv", index=False)

print("✅ Submission saved as 'submission_gaussian_process.csv'")

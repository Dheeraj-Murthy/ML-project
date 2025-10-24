import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import joblib

warnings.filterwarnings('ignore')

def remove_outliers(X_train, y_train):
    """
    Removes key outliers from the training data by identifying properties with
    extremely large total area but disproportionately low sale price.
    This step is crucial for linear and tree-based models.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (HotelValue).

    Returns:
        pd.DataFrame, pd.Series: Cleaned features and target.
    """
    df = X_train.copy()
    df['HotelValue'] = y_train

    # Feature necessary for outlier detection
    df['TotalSF'] = (df['GroundFloorArea'] + df['UpperFloorArea'] + df['BasementTotalSF'])

    # Heuristic for detecting high-leverage points:
    # 1. Properties with TotalSF > 5000 sq ft AND HotelValue < $200,000 (usually errors)
    # 2. Properties with LandArea > 100,000 sq ft (extreme acreage)
    outlier_condition = (df['TotalSF'] > 5000) & (df['HotelValue'] < 200000)
    outlier_condition |= (df['LandArea'] > 100000)

    # Filter out the rows where the condition is True
    X_cleaned = df[~outlier_condition].drop(columns=['HotelValue', 'TotalSF'])
    y_cleaned = y_train[~outlier_condition]

    print(f"Removed {len(X_train) - len(X_cleaned)} outliers based on TotalSF/Price and LandArea.")
    return X_cleaned, y_cleaned

def major_feature_engineering(df):
    """
    Performs major feature engineering steps for the Hotel Value dataset.
    """
    df = df.copy()

    # --- 1. Area Features ---
    # Total Usable Area (Above Ground + Basement)
    df['TotalSF'] = (df['GroundFloorArea'] + df['UpperFloorArea'] + df['BasementTotalSF'])

    # --- 2. Temporal Features ---
    df['Age'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRemodel'] = df['YearSold'] - df['RenovationYear']
    df['YearsSinceRemodel'] = np.where(df['YearsSinceRemodel'] < 0, 0, df['YearsSinceRemodel'])
    df.loc[df['RenovationYear'] == 0, 'YearsSinceRemodel'] = df['Age']

    # --- 3. Quality and Condition Scores ---
    df['OverallScore'] = (df['OverallQuality'] + df['OverallCondition']) / 2.0

    # --- 4. Count Features ---
    df['TotalBaths'] = (df['FullBaths'] + 0.5 * df['HalfBaths'] +
                        df['BasementFullBaths'] + 0.5 * df['BasementHalfBaths'])

    # --- 5. Interaction Features (Example) ---
    df['Qual_x_GroundSF'] = df['OverallQuality'] * df['GroundFloorArea']

    # --- 6. Feature Reduction/Drop ---
    drop_cols = ['GroundFloorArea', 'UpperFloorArea', 'BasementTotalSF',
                 'ConstructionYear', 'RenovationYear', 'OverallQuality',
                 'OverallCondition', 'FullBaths', 'HalfBaths',
                 'BasementFullBaths', 'BasementHalfBaths']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    return df

def create_preprocessor_pipeline(X_train):
    """
    Creates and fits the ColumnTransformer preprocessing pipeline.
    """
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Numerical: Impute with median, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Impute NaNs with 'None', then One-Hot Encode (to capture absence of feature)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    return preprocessor

def run_preprocessing(train_file="train.csv", test_file="test.csv", target_log_transform=True):
    """
    Main function to load, engineer, and preprocess the data.
    """
    # Load the datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    test_ids = test_df['Id']

    y_full_train = train_df['HotelValue']
    X_full_train = train_df.drop(['Id', 'HotelValue'], axis=1)
    X_test = test_df.drop('Id', axis=1)

    # --- 1. Outlier Removal (Applied to Training Data ONLY) ---
    X_full_train_clean, y_full_train_clean = remove_outliers(X_full_train, y_full_train)

    # --- 2. Feature Engineering ---
    X_full_train_fe = major_feature_engineering(X_full_train_clean)
    X_test_fe = major_feature_engineering(X_test)

    # Align columns
    X_test_fe = X_test_fe[X_full_train_fe.columns]

    # --- 3. Target Transformation ---
    y_train_transformed = np.log1p(y_full_train_clean) if target_log_transform else y_full_train_clean

    # --- 4. Create and Fit Preprocessor ---
    preprocessor = create_preprocessor_pipeline(X_full_train_fe)
    preprocessor.fit(X_full_train_fe)

    # Save the fitted preprocessor
    joblib.dump(preprocessor, 'fitted_preprocessor.joblib')

    # --- 5. Transform Data ---
    X_train_processed = preprocessor.transform(X_full_train_fe)
    X_test_processed = preprocessor.transform(X_test_fe)

    print("\n--- Data Preprocessing Summary ---")
    print(f"Processed Training Data Shape: {X_train_processed.shape}")
    print(f"Fitted preprocessor saved to 'fitted_preprocessor.joblib'.")

    return X_train_processed, X_test_processed, y_train_transformed, test_ids, preprocessor

if __name__ == '__main__':
    # Execute the preprocessing script
    X_train_proc, X_test_proc, y_train_trans, test_ids, fitted_preprocessor = preprocess_data(target_log_transform=True)
    print("\nSuccessfully loaded and transformed data. Ready for model training.")

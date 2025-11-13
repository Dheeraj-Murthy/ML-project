import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import os
import csv
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- 1. Define Global Ordinal Mappings ---
QUAL_MAPPING = {
    'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, 'NA': 0, np.nan: 0
}

BSMT_HEIGHT_MAPPING = {
    'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, np.nan: 0
}

# --- 2. Custom Ordinal Encoder Class ---


class OrdinalEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, mappings):
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.mappings.items():
            X_copy[col] = X_copy[col].fillna('None').astype(
                str).map(mapping).fillna(0).astype(int)
        return X_copy


def run_preprocessing(
        train_file="train.csv",
        test_file="test.csv",
        target_log_transform=True):
    """
    Loads data, performs outlier removal and feature engineering, and applies
    (or fits/saves, if missing) the ColumnTransformer using the functions
    defined in the imported preprocessing module.

    Returns processed training/test data, log-transformed target, test IDs, and the preprocessor.
    """
    # Define paths relative to the current script's location
    DATA_PATH = os.path.join(
        os.path.dirname(__file__),
        '..',
        'dataset') + os.sep
    PREPROCESSOR_PATH = os.path.join(
        os.path.dirname(__file__),
        '..',
        'PreProcessing',
        'fitted_preprocessor.joblib')

    # --- 1. Load Data ---
    try:
        train_df = pd.read_csv(DATA_PATH + train_file)
        test_df = pd.read_csv(DATA_PATH + test_file)
    except FileNotFoundError as e:
        print(
            f"ERROR: Could not find data files. Ensure they are in the '{DATA_PATH}' folder.")
        raise e

    test_ids = test_df['Id']
    y_full = train_df['HotelValue']
    X_full = train_df.drop(columns=['Id', 'HotelValue'])
    X_test = test_df.drop(columns=['Id'])

    # --- 2. Outlier Removal (Applied only to training data) ---
    print("\nStarting preprocessing steps...")
    X_full_clean, y_full_clean = remove_outliers(X_full, y_full)

    # --- 3. Feature Engineering (Applied to clean train and raw test data) ---
    X_full_fe = major_feature_engineering(X_full_clean)
    X_test_fe = major_feature_engineering(X_test)

    # --- 4. Target Transformation ---
    y_train_log = np.log1p(
        y_full_clean) if target_log_transform else y_full_clean

    # --- 5. Load/Fit Preprocessor ---
    preprocessor = None
    try:
        # Load the preprocessor (assumes it was fitted by a separate script
        # run)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"Successfully loaded preprocessor from {PREPROCESSOR_PATH}.")
    except FileNotFoundError:
        print(
            f"WARNING: Preprocessor not found at {PREPROCESSOR_PATH}. Fitting and saving it now...")
        # If not found, fit and save it using the function from
        # preprocessing.py
        preprocessor = create_and_fit_preprocessor(X_full_fe)

    # --- 6. Transform Data ---
    # Transform to NumPy arrays as tree-based models work optimally with them
    X_train_proc = preprocessor.transform(X_full_fe)
    X_test_proc = preprocessor.transform(X_test_fe)

    print(
        f"Data ready. Training data shape: {
            X_train_proc.shape}, Test data shape: {
            X_test_proc.shape}")

    return X_train_proc, X_test_proc, y_train_log, y_full_clean, test_ids, preprocessor


# ----------------------------------------------------------------------
# --- UPDATED FUNCTION: Drop Columns with High Missingness or High Zero Count ---
def drop_low_variance_cols(df, threshold=0.95):
    """
    Drops columns where the percentage of:
    1. Missing values (NaN) in any column.
    2. Zero values (0) in numerical columns.
    ...exceeds the specified threshold.
    """
    df_copy = df.copy()
    cols_to_drop = []

    for col in df_copy.columns:
        series = df_copy[col]
        drop_reason = None

        # 1. Check for high missingness (NaN)
        missing_percent = series.isnull().sum() / len(series)
        if missing_percent >= threshold:
            cols_to_drop.append(col)
            drop_reason = 'Missing'
            continue

        # 2. Check for high zero count in numerical columns (excluding PropertyClass and Year columns)
        # PropertyClass is used as an ordinal category, and Year columns are
        # treated temporally.
        if pd.api.types.is_numeric_dtype(series) and col not in [
            'PropertyClass',
            'ConstructionYear',
            'RenovationYear',
            'ParkingConstructionYear',
            'YearSold',
                'MonthSold']:
            zero_count = (series == 0).sum()
            zero_percent = zero_count / len(series)

            if zero_percent >= threshold:
                cols_to_drop.append(col)
                drop_reason = 'Zero-Value'

    # Remove duplicates and execute drop
    cols_to_drop = list(set(cols_to_drop))

    if cols_to_drop:
        print(
            f"Dropping {
                len(cols_to_drop)} columns with >{
                threshold *
                100}% missing/zero values: {cols_to_drop}")
        df_copy = df_copy.drop(columns=cols_to_drop, errors='ignore')

    return df_copy
# ----------------------------------------------------------------------


def major_feature_engineering(df):
    """Performs major feature engineering steps on the feature set."""
    df = df.copy()

    # --- 1. Area and Count Aggregations ---
    # NOTE: These columns are created regardless of whether their components
    # were dropped.
    df['TotalOutdoorArea'] = (
        df['TerraceArea'] +
        df['OpenVerandaArea'] +
        df['EnclosedVerandaArea'] +
        df['ScreenPorchArea']).fillna(0)
    df['TotalSF'] = (
        df['GroundFloorArea'] +
        df['UpperFloorArea'] +
        df['ParkingArea'] +
        df['TotalOutdoorArea']).fillna(0)

    df['TotalBaths'] = (
        df['FullBaths'] +
        0.5 *
        df['HalfBaths'] +
        df['BasementFullBaths'] +
        0.5 *
        df['BasementHalfBaths']).fillna(0)

    # Assumes these columns were NOT dropped
    df['OverallScore'] = (df['OverallQuality'] + df['OverallCondition']) / 2.0

    # --- 2. Temporal Features ---
    df['Age'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRemodel'] = df['YearSold'] - df['RenovationYear']
    df['YearsSinceRemodel'] = np.where(
        df['YearsSinceRemodel'] < 0, 0, df['YearsSinceRemodel'])
    df.loc[df['RenovationYear'] == df['ConstructionYear'],
           'YearsSinceRemodel'] = df['Age']

    # --- 3. Interaction Feature (Example) ---
    df['Qual_x_GroundSF'] = df['OverallQuality'] * \
        df['GroundFloorArea']  # Assumes these columns were NOT dropped

    # --- 4. Feature Reduction/Drop ---
    drop_cols = [
        'GroundFloorArea',
        'UpperFloorArea',
        'ConstructionYear',
        'RenovationYear',
        'FullBaths',
        'HalfBaths',
        'BasementFullBaths',
        'BasementHalfBaths',
        'BasementFacilitySF1',
        'BasementFacilitySF2',
        'TerraceArea',
        'OpenVerandaArea',
        'EnclosedVerandaArea',
        'ScreenPorchArea']

    if 'Parking Area' in df.columns:
        drop_cols.append('Parking Area')

    df = df.drop(
        columns=[
            col for col in drop_cols if col in df.columns],
        errors='ignore')
    return df


def remove_outliers(X, y):
    """Removes key outliers from the training data using IQR on TotalSF."""
    data = pd.concat([X, y.rename('HotelValue')], axis=1)

    data['TotalSF_tmp'] = (
        data['GroundFloorArea'].fillna(0) +
        data['UpperFloorArea'].fillna(0) +
        data['BasementTotalSF'].fillna(0))

    # 1. Simple, high-leverage outlier removal (Domain Knowledge)
    data = data.loc[~((data['LandArea'] > 50000) &
                      (data['HotelValue'] < 200000))]

    # 2. IQR-based removal for combined area (TotalSF)
    q1 = data['TotalSF_tmp'].quantile(0.25)
    q3 = data['TotalSF_tmp'].quantile(0.75)
    iqr = q3 - q1
    data = data.loc[data['TotalSF_tmp'] <= (q3 + 3 * iqr)]

    data = data.drop(columns=['TotalSF_tmp'])

    return data.drop(columns=['HotelValue']), data['HotelValue']


def create_and_fit_preprocessor(X_train_fe):
    """
    Creates, fits, and saves the fitted ColumnTransformer for consistent
    imputation, scaling, and encoding, including the new Ordinal Encoding.
    """

    # --- 1. Define Master Feature Groups (All possible columns) ---
    master_qual_cond_cols = [
        'OverallQuality',
        'OverallCondition',
        'ExteriorQuality',
        'ExteriorCondition',
        'HeatingQuality',
        'KitchenQuality',
        'LoungeQuality',
        'ParkingQuality',
        'ParkingCondition',
        'PoolQuality',
        'BasementCondition',
        'BasementExposure',
        'LandElevation',
        'LandSlope',
        'PropertyClass']

    master_bsmt_height_col = ['BasementHeight']

    # --- 2. Filter Feature Groups based on presence in X_train_fe ---
    present_cols = set(X_train_fe.columns)

    # Filter ordinal columns
    qual_cond_cols = [
        col for col in master_qual_cond_cols if col in present_cols]
    bsmt_height_col = [
        col for col in master_bsmt_height_col if col in present_cols]

    # Identify numerical and nominal columns currently present
    numerical_cols = X_train_fe.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    nominal_cols = X_train_fe.select_dtypes(
        include=['object']).columns.tolist()

    # Ensure no overlap between numerical and ordinal/nominal lists
    ordinal_nominal_cols = qual_cond_cols + bsmt_height_col + nominal_cols
    numerical_cols = [
        col for col in numerical_cols if col not in ordinal_nominal_cols]

    # --- 3. Transformers Definition ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    ordinal_mappings = {col: QUAL_MAPPING for col in qual_cond_cols}
    ordinal_mappings.update(
        {col: BSMT_HEIGHT_MAPPING for col in bsmt_height_col})

    ordinal_transformer = Pipeline(steps=[
        ('ordinal_encode', OrdinalEncoderCustom(mappings=ordinal_mappings)),
        ('scaler', StandardScaler())
    ])

    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # --- 4. ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', ordinal_transformer, qual_cond_cols + bsmt_height_col),
            ('nominal', nominal_transformer, nominal_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='passthrough'
    )

    preprocessor.fit(X_train_fe)
    joblib.dump(preprocessor, 'fitted_preprocessor.joblib')

    return preprocessor


def save_model_results(file_name, model_name, rmse_score):
    """
    Saves model results (file name, model name, RMSE score) to a CSV file.
    """
    results_file = os.path.join(os.path.dirname(__file__), '..', 'results.csv')
    file_exists = os.path.isfile(results_file)

    with open(results_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(
                ['File Name', 'Model Name', 'RMSE Validation Score'])
        writer.writerow([file_name, model_name, rmse_score])


if __name__ == '__main__':
    # --- Execution Block to Generate Joblib File ---
    try:
        DATA_PATH = '../dataset/'
        train_df = pd.read_csv(DATA_PATH + 'train.csv')

        # 1. Separate and Log-Transform Target
        y_full = np.log1p(train_df['HotelValue'])
        X_full = train_df.drop(columns=['Id', 'HotelValue'])

        # --- NEW STEP: Drop Low Variance Columns (Missing or Zero) ---
        X_full = drop_low_variance_cols(X_full, threshold=0.95)

        # 2. Outlier Removal
        X_full_clean, y_full_clean = remove_outliers(X_full, y_full)

        # 3. Feature Engineering
        X_full_fe = major_feature_engineering(X_full_clean.copy())

        # 4. Fit and Save Preprocessor
        create_and_fit_preprocessor(X_full_fe)

        print(
            "\n'preprocessing.py' executed successfully with low-variance column removal.")
        print(
            "The necessary file 'fitted_preprocessor.joblib' is ready for model training.")

    except FileNotFoundError:
        print(f"ERROR: Could not find data files. Ensure 'train.csv' is in the current working directory or adjust the path.")

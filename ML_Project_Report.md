# ML Project Report: Hotel Property Value Prediction

## Team Members

1. M S Dheeraj Murthy - imt2023552
2. Ayush Tiwari - imt2023524
3. Mathew Joseph - imt2023008

**Github Link:** https://github.com/Dheeraj-Murthy/ML-project

## Task

The primary task of this project is to build a machine learning model to
accurately predict the value of hotel properties. This is a regression problem
where the target variable is the `HotelValue`. The project involves exploring
the dataset, performing necessary preprocessing, training various regression
models, and evaluating their performance to find the best model for this
prediction task.

## Dataset and Features Description

The dataset contains various features related to hotel properties. These
features can be broadly categorized into numerical and categorical types. The
dataset is split into `train.csv` and `test.csv` files. The training data
contains 1200 rows and 81 columns, including the target variable `HotelValue`.
The test data is used for final evaluation. The features include information
about the property's size, location, quality, condition, and various amenities.

## EDA and pre-processing

The following section outlines the necessary steps for conducting essential
Exploratory Data Analysis (EDA) to gain insights, detect potential issues, and
prepare data for model training.

### 1. Import Libraries and Load Dataset

The initial step in the EDA process involves importing essential libraries such
as pandas, numpy, matplotlib, seaborn, and relevant modules from scikit-learn
for data preprocessing. The dataset is loaded into a pandas DataFrame for
further analysis and manipulation.

### 2. Data Overview

The dataset is examined by displaying the first few rows, checking data types,
and identifying any missing values. This step helps in understanding the
structure and initial quality of the data. A statistical summary of numerical
columns is also obtained to review the central tendency, spread, and potential
outliers.

### 3. Handling Missing Values

The dataset was examined for missing values, and it was found that several
columns have a significant number of missing entries. Columns with a high
percentage of missing values (e.g., `PoolQuality`, `ExtraFacility`,
`ServiceLaneType`) were dropped. For other columns with fewer missing values,
imputation was performed using the median for numerical features and a constant
value ('None') for categorical features.

### 4. Handling Duplicate Values

The dataset was checked for duplicate rows, and none were found. Therefore, no
action was required for handling duplicates.

### 5. Exploratory Data Analysis (EDA)

Various plots were generated to analyse the data distribution, detect outliers,
and explore relationships among features. These visualisations provide critical
insights that inform subsequent preprocessing and modelling decisions.

**Distribution Analysis of Numerical Features:**

Histograms and Q-Q plots were used to analyze the distribution of numerical
features. It was observed that many features are skewed. The target variable,
`HotelValue`, is also right-skewed, so a log transformation (`np.log1p`) was
applied to make its distribution more normal.

**Outlier Detection:**

Boxplots were used to identify outliers in numerical features. Outliers were
found in several columns, including `LandArea` and `UsableArea`. A function
`remove_outliers` was implemented to handle these outliers by removing data
points based on domain knowledge (e.g., large land area with low hotel value)
and the Interquartile Range (IQR) method.

**Correlation Analysis:**

A correlation matrix heatmap was generated for numerical features to understand
their relationships. It was found that `OverallQuality` has a high positive
correlation with `HotelValue` (0.79). Other features like `UsableArea`,
`BasementTotalSF`, and `ParkingArea` also show a significant positive
correlation with the target variable. Some features were also found to be highly
correlated with each other, such as `ParkingCapacity` and `ParkingArea` (0.88).

### 6. Data Preprocessing

In this stage, the data is prepared for modelling through the following steps:

- **Feature Engineering**: New features were created to capture more
  information. These include `TotalOutdoorArea`, `TotalSF`, `TotalBaths`,
  `OverallScore`, `Age`, and `YearsSinceRemodel`.

- **Encoding Categorical Variables**: Ordinal categorical features were encoded
  using a custom ordinal encoder, and nominal categorical features were one-hot
  encoded.

- **Scaling Numerical Features**: Numerical features were scaled using
  `StandardScaler` to bring them to a similar scale.

- **Target Transformation**: The target variable `HotelValue` was
  log-transformed.

A `ColumnTransformer` pipeline was created to apply these preprocessing steps
consistently to the training and test data. The fitted preprocessor was saved to
a `fitted_preprocessor.joblib` file.

### 7. Splitting the Dataset

The dataset is split into training and testing sets to evaluate the model's
performance on unseen data.

## Models Used For Training

Several regression models were trained to predict the hotel property value. The
performance of each model was evaluated using the Root Mean Squared Error (RMSE)
on a validation set. The results are summarized below, from best to worst
performing.

### 1. Ridge Regression

- **Performance:** The best performing model was Ridge Regression, which
  achieved a validation RMSE of **20071.41**.

### 2. Bayesian Ridge Regression

- **Performance:** This model achieved a validation RMSE of **20425.49**.

### 3. Lasso Regression

- **Performance:** This model achieved a validation RMSE of **21315.94**.

### 4. Linear Regression

- **Performance:** The baseline Linear Regression model had a validation RMSE of
  **21327.46**.

### Other Models

The following models were also trained, with their respective validation RMSEs:

- **CatBoost Regressor:** 26503.93
- **ElasticNet:** 26671.33
- **LGBM Regressor:** 27240.31
- **Gaussian Process Regressor:** 28094.36
- **Random Forest Regressor:** 29172.59
- **XGBoost Regressor:** 29296.00
- **Polynomial Regression:** 30305.02
- **AdaBoost Regressor:** 33193.79

## Discussion on the Performance of Different Approaches

The results show that the regularized linear models, particularly **Ridge
Regression**, performed the best on this dataset. Ridge Regression achieved the
lowest RMSE of **20071.41**, outperforming more complex ensemble methods like
XGBoost and Random Forest.

This suggests that the relationships between the features and the target
variable in this dataset may be largely linear, and the regularization provided
by the Ridge model was effective in preventing overfitting and improving
generalization.

The more complex models like XGBoost and CatBoost did not perform as well, which
could be due to several factors. It's possible that with this particular
dataset, the additional complexity of these models led to overfitting, or that
the hyperparameter tuning was not sufficient to find an optimal set of
parameters for these models. The performance of the XGBoost model from
`xgboost_model.py` (RMSE: 29296.00) was not competitive with the simpler linear
models.

This project highlights an important lesson in machine learning: more complex
models are not always better. For this particular problem, a well-regularized
linear model proved to be the most effective solution.

## Interesting Observations

- **The Power of Preprocessing and Tuning:** The most striking observation is
  the dramatic difference in performance between the two XGBoost models. This
  highlights that the choice of algorithm alone is not enough; a well-structured
  preprocessing pipeline and systematic hyperparameter tuning are critical for
  achieving high performance.

- **Feature Engineering:** The creation of new features like `TotalSF` and `Age`
  likely provided the models with more meaningful information, contributing to
  better predictions.

- **Linear Models vs. Ensemble Models:** While the linear models (Ridge, Lasso,
  etc.) provided a decent baseline, they were clearly outperformed by the top
  XGBoost model. This suggests that the relationships between the features and
  the target variable are complex and non-linear, and that ensemble methods are
  better suited for this problem.

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

- [Pandas Documentation](https://pandas.pydata.org/docs/)

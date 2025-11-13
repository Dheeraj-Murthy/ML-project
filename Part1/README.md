# Hotel Property Value Prediction

### Machine Learning Project Report

**AIT 511**
**Auschwitz Kitchen**

**Team Members:**

* M S Dheeraj Murthy - IMT2023552
* Ayush Tiwari - IMT2023524
* Mathew Joseph - IMT2023008

**GitHub Link:** [https://github.com/Dheeraj-Murthy/ML-project](https://github.com/Dheeraj-Murthy/ML-project)

---

## Table of Contents

1. [Task](#task)
2. [Dataset and Feature Description](#dataset-and-feature-description)
3. [EDA and Pre-processing](#eda-and-pre-processing)
4. [Data Preprocessing](#data-preprocessing)
5. [Splitting the Dataset](#splitting-the-dataset)
6. [Models Used For Training](#models-used-for-training)
7. [Model Performance Summary](#model-performance-summary)
8. [Discussion on Performance](#discussion-on-the-performance-of-different-approaches)
9. [Interesting Observations](#interesting-observations)
10. [References](#references)

---

## Task

The primary task of this project is to build a machine learning model to accurately predict the value of hotel properties.
This is a **regression** problem where the target variable is `HotelValue`.
The project involves exploring the dataset, performing preprocessing, training various regression models, and evaluating their performance to find the best model for this prediction task.

---

## Dataset and Feature Description

The dataset includes **1200 rows** and **81 features** in the training set.

* **Numerical:** Areas (Parking, Basement, Living), Counts (Bathrooms, Rooms)
* **Categorical:** Location, Heating type, Quality/Condition ratings, etc.
* **Target Variable:** **HotelValue**

The dataset is split into `train.csv` and `test.csv`.
The training data has 1200 rows and 81 columns, including the target variable.

---

## EDA and Pre-processing

### Import Libraries and Load Dataset

Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, and modules from `scikit-learn` for data preprocessing.
The dataset was loaded into a pandas DataFrame for analysis and manipulation.

### Data Overview

The dataset was inspected for:

* Data types
* Missing values
* Statistical summaries
  This helped understand the data structure, quality, and distribution.

### Handling Missing Values

Columns with high missing values (e.g., `PoolQuality`, `ExtraFacility`, `ServiceLaneType`) were dropped.
For remaining columns:

* **Numerical:** Imputed with median
* **Categorical:** Imputed with `'None'`

![Missing Data Matrix](missing_data.png)

### Handling Duplicate Values

No duplicate rows were found in the dataset.

### Exploratory Data Analysis (EDA)

#### Distribution Analysis of Numerical Features

Histograms and Q-Q plots showed that many numerical features were skewed.
The target variable `HotelValue` was **right-skewed**, so a **log transformation (`np.log1p`)** was applied.

![Q-Q Plot](qq_plot.png)
![District Proportion](pie.png)

#### Outlier Detection

Boxplots revealed outliers in columns like `LandArea` and `UsableArea`.
A function `remove_outliers` was implemented using the **IQR method** and domain knowledge.

![Boxplot of UsableData](outlier.png)

#### Correlation Analysis

A correlation matrix was created to study relationships among features.

Key correlations:

* `OverallQuality` ↔ `HotelValue`: **0.79**
* `UsableArea`, `BasementTotalSF`, and `ParkingArea`: strong positive correlations
* `ParkingCapacity` and `ParkingArea`: **0.88**

![Correlation Heatmap](correlation.png)

For a detailed dataset report, refer to the notebook:
👉 [Dataset Report Notebook](https://github.com/Dheeraj-Murthy/ML-project/blob/main/generate_report.ipynb)

---

## Data Preprocessing

### Feature Engineering

New features were created to enhance model learning:

| **New Feature**     | **Description**                                              |
| ------------------- | ------------------------------------------------------------ |
| `TotalOutdoorArea`  | Sum of terrace, veranda, and porch areas                     |
| `TotalSF`           | Sum of ground floor, upper floor, parking, and outdoor areas |
| `TotalBaths`        | Sum of full and half baths (including basement)              |
| `OverallScore`      | Average of OverallQuality and OverallCondition               |
| `Age`               | `YearSold - ConstructionYear`                                |
| `YearsSinceRemodel` | `YearSold - RenovationYear`                                  |

### Encoding Categorical Variables

* **Ordinal features:** Custom ordinal encoder
* **Nominal features:** One-hot encoding

### Scaling Numerical Features

Scaled using **StandardScaler** to normalize the range.

### Target Transformation

Applied **log transformation** to `HotelValue`.

A `ColumnTransformer` pipeline was created and saved as
`fitted_preprocessor.joblib`.

---

## Splitting the Dataset

The dataset was split into **training** and **testing** sets to evaluate model generalization.

---

## Models Used For Training

### Linear Models

These assume a linear relationship between features and the target.

#### Linear Regression

Serves as the baseline model.

#### Ridge, Lasso, and ElasticNet

Regularized versions to prevent overfitting:

* Ridge: L2 penalty
* Lasso: L1 penalty
* ElasticNet: Combination of both

#### Bayesian Ridge Regression

A probabilistic version of Ridge that estimates regularization parameters automatically.

---

### Non-Linear Models

#### Polynomial Regression

Captures non-linear relationships using polynomial expansion.

#### Gaussian Process Regressor

A Bayesian non-parametric model providing uncertainty estimates.

---

### Ensemble Models

#### Random Forest Regressor

An ensemble of decision trees.
**Best parameters:**

* `n_estimators=1000`
* `max_depth=15`
* `min_samples_leaf=5`

#### AdaBoost Regressor

Boosts weak learners (decision trees).
**Best parameters:**

* `n_estimators=100`
* `learning_rate=0.1`

#### XGBoost, CatBoost, and LightGBM Regressors

Powerful gradient boosting models.

**CatBoost Regressor:**

```
iterations = 10000
learning_rate = 0.022767
border_count = 597
depth = 5
loss_function = RMSE
l2_leaf_reg = 3.0
subsample = 1
rsm = 1
bagging_temperature = 1
random_strength = 1
```

**XGBoost Regressor:**

```
n_estimators = 100000
learning_rate = 0.022767
max_depth = 5
subsample = 1.0
colsample_bytree = 1.0
reg_lambda = 3.0
objective = reg:squarederror
```

---

## Model Performance Summary

| **Rank** | **Model**                  | **Validation RMSE** |
| -------- | -------------------------- | ------------------- |
| 1        | Ridge Regression           | 20071.41            |
| 2        | Bayesian Ridge Regression  | 20425.49            |
| 3        | Lasso Regression           | 21315.94            |
| 4        | Linear Regression          | 21327.46            |
| 5        | CatBoost Regressor         | 26503.93            |
| 6        | ElasticNet                 | 26671.33            |
| 7        | LGBM Regressor             | 27240.31            |
| 8        | Gaussian Process Regressor | 28094.36            |
| 9        | Random Forest Regressor    | 29172.59            |
| 10       | XGBoost Regressor          | 29296.00            |
| 11       | Polynomial Regression      | 30305.02            |
| 12       | AdaBoost Regressor         | 33193.79            |

---

## Discussion on the Performance of Different Approaches

Ridge Regression achieved the **lowest RMSE of 20071.41** (test RMSE: **18237.81** on Kaggle).
Despite testing complex models (XGBoost, CatBoost, Random Forest), **regularized linear models** performed best — suggesting mostly **linear relationships** in the dataset.

Complex models may have overfit or required more hyperparameter tuning.
This demonstrates that **simpler models can outperform more complex ones** when properly regularized.

---

## Interesting Observations

* **Linear Models Outperforming Ensembles:**
  Ridge Regression outperformed XGBoost, CatBoost, and Random Forest — showing simplicity and regularization can be powerful.

* **Importance of Regularization:**
  Ridge and Lasso outperformed unregularized Linear Regression, confirming the role of regularization in avoiding overfitting.

* **Impact of Feature Engineering:**
  Features like `TotalSF` and `Age` added meaningful context that improved model performance.

---

## References

* [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [Pandas Documentation](https://pandas.pydata.org/docs/)

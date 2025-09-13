# California Housing Price Prediction

##  Overview
This project predicts California housing prices using **Linear Regression** and **Random Forest Regressor**.  
It covers the complete pipeline: data loading, exploration, preprocessing, feature engineering, model training, and evaluation.

Dataset: [California Housing Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)  

---

##  1. Import Required Libraries
- `pandas`, `numpy` → data manipulation
- `matplotlib`, `seaborn` → data visualization
- `scikit-learn` → machine learning models and preprocessing

---

##  2. Load Dataset
- Load `housing.csv` into a pandas DataFrame.
- Assign features and target:
  - **Features (X):** all columns except `median_house_value`
  - **Target (y):** `median_house_value`
- Train-test split: **80% training, 20% testing**

---

##  3. Exploratory Data Analysis (EDA)
- Preview dataset using `head()` and `info()`
- Plot histograms of features
- Generate heatmap of correlations between numeric features

---

## 4. Data Transformation
- Apply **log(x+1)** transformation on:
  - `total_rooms`
  - `total_bedrooms`
  - `population`
  - `households`
- Helps to normalize skewed distributions

---

##  5. Encoding Categorical Feature
- One-hot encode `ocean_proximity` using `pd.get_dummies()`
- Drop the original `ocean_proximity` column

---

##  6. Feature Engineering
- Create new ratio-based features:
  - `bedrooms_ratio = total_bedrooms / total_rooms`
  - `households_rooms = total_rooms / households`
- Improves predictive power

---

##  7. Data Visualization
- Correlation heatmap after encoding & feature engineering
- Scatterplot of `latitude` vs `longitude`, color-coded by `median_house_value`

---

##  8. Model Training

### 8.1 Linear Regression
- Standardize features using `StandardScaler`
- Train a **Linear Regression** model
- Evaluate using `.score()` on test data

### 8.2 Random Forest Regressor
- Train a **Random Forest** model
- Compare R² score with Linear Regression

---

##  9. Hyperparameter Tuning (Grid Search)
- Performed using **GridSearchCV**
- Parameters tuned:
  - `n_estimators: [100, 200, 300]`
  - `min_samples_split: [2, 4]`
  - `max_depth: [None, 4, 8]`
- Best estimator is selected and tested on evaluation set

---

##  10. Model Evaluation
- Metrics:
  - R² Score on test set
  - Comparison between Linear Regression, Random Forest, and Grid Search optimized model

---

##  Requirements
Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

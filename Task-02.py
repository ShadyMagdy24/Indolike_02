import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats
import streamlit as st

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("House Price Prediction - Streamlit App")

# Section 1: Load the Data
st.header("1. Load and Explore Data")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

st.subheader("Train Data - Overview")
st.write(train_df.head())

st.subheader("Test Data - Overview")
st.write(test_df.head())

st.subheader("Missing Values in Train Dataset")
st.write(train_df.isnull().sum()[train_df.isnull().sum() > 0])

st.subheader("Missing Values in Test Dataset")
st.write(test_df.isnull().sum()[test_df.isnull().sum() > 0])

# Section 2: Handle Missing Values
st.header("2. Handle Missing Values")

# Impute missing values
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(train_df['LotFrontage'].median())
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(test_df['LotFrontage'].median())

numerical_columns_train = train_df.select_dtypes(include=['float64', 'int64']).columns
numerical_columns_test = test_df.select_dtypes(include=['float64', 'int64']).columns

train_df[numerical_columns_train] = train_df[numerical_columns_train].fillna(train_df[numerical_columns_train].mean())
test_df[numerical_columns_test] = test_df[numerical_columns_test].fillna(test_df[numerical_columns_test].mean())

columns_with_high_missing = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
for col in columns_with_high_missing:
    train_df[col] = train_df[col].fillna("NA")
    test_df[col] = test_df[col].fillna("NA")

st.success("Missing values handled successfully!")

# Section 3: Data Visualization
st.header("3. Data Visualization")

# Numerical feature distribution
st.subheader("Numerical Feature Distribution")
numerical_columns = train_df.select_dtypes(include=['float64', 'int64']).columns
column_to_plot = st.selectbox("Select a numerical column to visualize", numerical_columns)

if column_to_plot:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(train_df[column_to_plot], kde=True, ax=ax)
    st.pyplot(fig)

# Heatmap of correlations
st.subheader("Correlation Heatmap")

# Select only numerical columns for correlation
numerical_columns_only = train_df.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix for numerical columns
correlation_matrix = numerical_columns_only.corr()

# Plot the heatmap
fig, ax = plt.subplots(figsize=(20, 16))  # Increased figure size for better clarity
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={'shrink': .8}, vmax=1.0, vmin=-1.0, linewidths=0.5, ax=ax)

ax.set_title('Correlation Heatmap of Numerical Features', fontsize=16)
ax.tick_params(axis='x', labelsize=10, rotation=45)
ax.tick_params(axis='y', labelsize=10)

st.pyplot(fig)


# Section 4: Feature Selection
st.header("4. Feature Selection")
correlation_threshold = st.slider("Set Correlation Threshold", 0.0, 1.0, 0.5)
correlation_with_target = correlation_matrix['SalePrice']
impactful_features = correlation_with_target[correlation_with_target.abs() > correlation_threshold].index.tolist()

st.write(f"Selected Features: {impactful_features}")

# Section 5: Dataset Preparation
st.header("5. Dataset Preparation")

selected_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'FullBath']
X = train_df[selected_features]
y = train_df['SalePrice']
X_test = test_df[selected_features]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

st.success("Data prepared successfully!")

# Section 6: Model Training and Evaluation
st.header("6. Model Training and Evaluation")

# Linear Regression
st.subheader("Linear Regression")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_val_pred = model.predict(X_val_scaled)

rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

st.write(f"Validation RMSE: {rmse}")
st.write(f"Validation MAE: {mae}")
st.write(f"Validation R²: {r2}")

# Polynomial Regression
st.subheader("Polynomial Regression")
degree = st.slider("Select Degree for Polynomial Features", 2, 5, value=4)

# Transform features to polynomial
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_val_poly = poly.transform(X_val_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_val_pred_poly = poly_model.predict(X_val_poly)

rmse_poly = np.sqrt(mean_squared_error(y_val, y_val_pred_poly))
mae_poly = mean_absolute_error(y_val, y_val_pred_poly)
r2_poly = r2_score(y_val, y_val_pred_poly)

st.write(f"Validation RMSE: {rmse_poly}")
st.write(f"Validation MAE: {mae_poly}")
st.write(f"Validation R²: {r2_poly}")

# Ridge Regression
st.subheader("Ridge Regression")
alpha_ridge = st.slider("Select Alpha for Ridge Regression", 0.1, 10.0, value=1.0)
ridge_model = Ridge(alpha=alpha_ridge)
ridge_model.fit(X_train_poly, y_train)
y_val_pred_ridge = ridge_model.predict(X_val_poly)

rmse_ridge = np.sqrt(mean_squared_error(y_val, y_val_pred_ridge))
mae_ridge = mean_absolute_error(y_val, y_val_pred_ridge)
r2_ridge = r2_score(y_val, y_val_pred_ridge)

st.write(f"Validation RMSE: {rmse_ridge}")
st.write(f"Validation MAE: {mae_ridge}")
st.write(f"Validation R²: {r2_ridge}")

# Lasso Regression
st.subheader("Lasso Regression")
alpha_lasso = st.slider("Select Alpha for Lasso Regression", 0.01, 1.0, value=0.1)
lasso_model = Lasso(alpha=alpha_lasso)
lasso_model.fit(X_train_poly, y_train)
y_val_pred_lasso = lasso_model.predict(X_val_poly)

rmse_lasso = np.sqrt(mean_squared_error(y_val, y_val_pred_lasso))
mae_lasso = mean_absolute_error(y_val, y_val_pred_lasso)
r2_lasso = r2_score(y_val, y_val_pred_lasso)

st.write(f"Validation RMSE: {rmse_lasso}")
st.write(f"Validation MAE: {mae_lasso}")
st.write(f"Validation R²: {r2_lasso}")

# Section 7: Predict and Save Submission File
st.header("7. Predict and Save Submission File")

# Predict on test data
y_test_pred = model.predict(X_test_scaled)
sample_submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_test_pred})

st.write(sample_submission.head())

# Downloadable CSV
st.download_button(
    label="Download Submission File",
    data=sample_submission.to_csv(index=False),
    file_name="submission.csv",
    mime="text/csv"
)



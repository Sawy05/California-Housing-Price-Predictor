# main_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

# Load dataset
df = pd.read_csv("housing.csv")

# ---------------------------
# Step 2: Exploratory Data Analysis
# ---------------------------
print(df.info())
print(df.describe())

# Histogram
df.hist(bins=30, figsize=(16, 12), edgecolor='black')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.drop(columns=["ocean_proximity"]).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Scatter plots
features = ['median_income', 'housing_median_age', 'total_rooms', 'latitude', 'longitude']
plt.figure(figsize=(16, 12))
for i, feature in enumerate(features):
    plt.subplot(3, 2, i + 1)
    sns.scatterplot(x=df[feature], y=df['median_house_value'], alpha=0.4)
    plt.title(f"{feature} vs median_house_value")
plt.tight_layout()
plt.show()

# ---------------------------
# Step 3: Data Cleaning
# ---------------------------
print("Missing values:\n", df.isnull().sum())
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
df['ocean_proximity'] = df['ocean_proximity'].astype('category')

# ---------------------------
# Step 4: Model Development
# ---------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("median_house_value", axis=1)
y = df_encoded["median_house_value"]

train_encoded_cols = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("\nModel Evaluation:")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Save model and scaler
joblib.dump(model, "linear_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(train_encoded_cols, "train_encoded_cols.pkl")
# ---------------------------
# Step 5: Regression Diagnostics
# ---------------------------
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs. Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

qqplot(residuals, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

X_vif = sm.add_constant(X_train_scaled)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif, i+1) for i in range(X.shape[1])]
print("\nVariance Inflation Factors:")
print(vif_data)

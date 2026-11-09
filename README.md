# 🏡 California Housing Real Estate Valuation Engine & Streamlit App

End-to-end **Predictive Real Estate Valuation Pipeline** and interactive **Streamlit Web Application** built on the California Housing dataset. Includes Exploratory Data Analysis (EDA), Multicollinearity Diagnostics via Variance Inflation Factor (VIF), Residual Q-Q analysis, and serialized Scikit-Learn inference pipelines.

---

## ✨ Key Engineering Features

- **Interactive Streamlit Valuation UI (`app.py`)**:
  - Real-time property valuation given Longitude, Latitude, Housing Median Age, Total Rooms/Bedrooms, Population, Households, Median Income, and Ocean Proximity category.
- **Statistical & Regression Diagnostics (`train_valuation_pipeline.py`)**:
  - Median imputation for missing bedroom counts and One-Hot Encoding for spatial proximity categories (`<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`).
  - `StandardScaler` normalization and `LinearRegression` training with **MSE**, **MAE**, and **$R^2$** evaluation.
  - **Variance Inflation Factor (VIF)** computation via `statsmodels` to quantify feature multicollinearity and Q-Q residual normality plots.

---

## 📂 Repository Contents

- `app.py` — Interactive Streamlit web application for real-time house price inference.
- `train_valuation_pipeline.py` — Full training, EDA plotting, VIF diagnostics, and artifact serialization script.
- `California_Housing_Regression_Analysis.ipynb` — Complete Jupyter Notebook walkthrough with visualizations.
- `linear_model.pkl`, `scaler.pkl`, `train_encoded_cols.pkl` — Pre-trained model artifacts ready for instant deployment.
- `housing.csv` — California Housing benchmark dataset.

---

## 🚀 Running the Web App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

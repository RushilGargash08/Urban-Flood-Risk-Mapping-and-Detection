"""
Ridge Regression Model – Urban Flood Risk Prediction
=====================================================
Predicts `flood_risk_score` (0-1) from geospatial, meteorological,
topographic, and SAR-inspired features.

Pipeline:
  1. Load & inspect the dataset
  2. One-hot encode categorical feature (soil_type)
  3. Train / test split (80/20)
  4. Standard-scale features
  5. Tune alpha with RidgeCV (built-in LOO / GCV)
  6. Evaluate on hold-out test set
  7. Print metrics & feature importances (coefficients)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# ── 1. Load data ─────────────────────────────────────────────────────────────
DATA_PATH = "urban_flood_risk_dataset.csv"
df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("URBAN FLOOD RISK – RIDGE REGRESSION MODEL")
print("=" * 65)
print(f"\nDataset shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print(f"\nTarget stats  :\n{df['flood_risk_score'].describe()}\n")

# ── 2. Feature engineering ───────────────────────────────────────────────────
# Drop the derived categorical label (it's just a bin of the target)
df = df.drop(columns=["flood_risk_category"])

# One-hot encode soil_type
df = pd.get_dummies(df, columns=["soil_type"], drop_first=True)

# Define feature matrix X and target y
TARGET = "flood_risk_score"
FEATURES = [c for c in df.columns if c != TARGET]
X = df[FEATURES].values
y = df[TARGET].values

print(f"Features ({len(FEATURES)}):")
for f in FEATURES:
    print(f"  • {f}")

# ── 3. Train / test split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test  size : {X_test.shape[0]}")

# ── 4. Feature scaling ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ── 5. Tune regularisation strength (alpha) with RidgeCV ────────────────────
alphas = np.logspace(-3, 3, 100)  # 0.001 → 1000
ridge_cv = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error", cv=5)
ridge_cv.fit(X_train_sc, y_train)

best_alpha = ridge_cv.alpha_
print(f"\nBest alpha (via 5-fold CV) : {best_alpha:.6f}")

# ── 6. Retrain final Ridge with best alpha ──────────────────────────────────
model = Ridge(alpha=best_alpha)
model.fit(X_train_sc, y_train)

# ── 7. Evaluate ─────────────────────────────────────────────────────────────
y_pred_train = model.predict(X_train_sc)
y_pred_test = model.predict(X_test_sc)

def report(label, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n── {label} ──")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  R²   : {r2:.6f}")
    return rmse, mae, r2

report("Train set", y_train, y_pred_train)
test_rmse, test_mae, test_r2 = report("Test set", y_test, y_pred_test)

# Cross-validated R² on full training data for robustness check
cv_r2 = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="r2")
print(f"\n── 5-Fold Cross-Validated R² ──")
print(f"  Mean : {cv_r2.mean():.6f}")
print(f"  Std  : {cv_r2.std():.6f}")

# ── 8. Feature importances (standardised coefficients) ──────────────────────
print("\n── Feature Coefficients (standardised) ──")
coef_df = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": model.coef_,
    "|coefficient|": np.abs(model.coef_),
}).sort_values("|coefficient|", ascending=False)

print(coef_df.to_string(index=False))

print(f"\nIntercept : {model.intercept_:.6f}")
print("=" * 65)
print("Done ✓")

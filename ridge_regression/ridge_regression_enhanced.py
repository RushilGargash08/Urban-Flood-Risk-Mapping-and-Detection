"""
Enhanced Ridge Regression – Urban Flood Risk Prediction
========================================================
Techniques applied to boost performance over the baseline:

  1. Domain-driven feature engineering
     • rain × built_up interaction (mimics the data-generation rule)
     • rain² and moisture² (capture diminishing-returns curves)
     • elevation_inverted (lower → riskier, makes relationship monotonic)
     • rain_moisture interaction
     • proximity score (inverse of distance_to_water_body_km)

  2. Polynomial features (degree 2, interaction-only option too)

  3. Broader alpha search with RidgeCV

  4. Full comparison: Baseline → Engineered → Poly(2) → Poly(2)+Engineered
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    """Train, predict, print metrics, return dict."""
    model.fit(X_tr, y_tr)
    p_tr = model.predict(X_tr)
    p_te = model.predict(X_te)
    rmse_tr = np.sqrt(mean_squared_error(y_tr, p_tr))
    rmse_te = np.sqrt(mean_squared_error(y_te, p_te))
    mae_te  = mean_absolute_error(y_te, p_te)
    r2_tr   = r2_score(y_tr, p_tr)
    r2_te   = r2_score(y_te, p_te)
    cv      = cross_val_score(model, X_tr, y_tr, cv=3, scoring="r2")
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(f"  Features      : {X_tr.shape[1]}")
    print(f"  Train RMSE    : {rmse_tr:.6f}   R² : {r2_tr:.6f}")
    print(f"  Test  RMSE    : {rmse_te:.6f}   R² : {r2_te:.6f}")
    print(f"  Test  MAE     : {mae_te:.6f}")
    print(f"  CV R² (3-fold): {cv.mean():.6f} ± {cv.std():.6f}")
    return {
        "name": name, "n_feat": X_tr.shape[1],
        "train_r2": r2_tr, "test_r2": r2_te,
        "test_rmse": rmse_te, "test_mae": mae_te,
        "cv_r2_mean": cv.mean(), "cv_r2_std": cv.std(),
    }


def add_engineered(df):
    """Return df with hand-crafted features appended."""
    df = df.copy()
    # Key interaction the generator uses
    df["rain_x_buildup"] = df["rainfall_mm_per_day"] * df["built_up_percentage"]
    # Quadratic terms for top predictors
    df["rain_sq"] = df["rainfall_mm_per_day"] ** 2
    df["moisture_sq"] = df["moisture_index"] ** 2
    # Inverted elevation (lower = riskier)
    df["elevation_inv"] = 500.0 - df["elevation_m"]
    # Cross interactions
    df["rain_x_moisture"] = df["rainfall_mm_per_day"] * df["moisture_index"]
    df["buildup_x_moisture"] = df["built_up_percentage"] * df["moisture_index"]
    # Proximity to water (closer = riskier)
    df["water_proximity"] = 1.0 / (df["distance_to_water_body_km"] + 0.1)
    # High-risk binary flag (mirrors generator's if-check)
    df["high_rain_buildup"] = (
        (df["rainfall_mm_per_day"] > 200) & (df["built_up_percentage"] > 70)
    ).astype(int)
    return df


# ── 1. Load & prepare ───────────────────────────────────────────────────────
DATA_PATH = "urban_flood_risk_dataset.csv"
raw = pd.read_csv(DATA_PATH)

print("=" * 60)
print("  ENHANCED RIDGE REGRESSION – PERFORMANCE COMPARISON")
print("=" * 60)
print(f"  Dataset: {raw.shape[0]} rows × {raw.shape[1]} cols")

raw = raw.drop(columns=["flood_risk_category"])
raw = pd.get_dummies(raw, columns=["soil_type"], drop_first=True)

TARGET = "flood_risk_score"
ALL_FEAT = [c for c in raw.columns if c != TARGET]

X = raw[ALL_FEAT].values
y = raw[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ── 2. Prepare variants ─────────────────────────────────────────────────────
alphas = np.logspace(-3, 3, 50)

results = []

# ── A. Baseline (same as original script) ────────────────────────────────────
scaler_a = StandardScaler()
Xa_tr = scaler_a.fit_transform(X_train)
Xa_te = scaler_a.transform(X_test)
model_a = RidgeCV(alphas=alphas)
model_a.fit(Xa_tr, y_train)
ridge_a = Ridge(alpha=model_a.alpha_)
results.append(evaluate("A) Baseline (linear features only)", ridge_a, Xa_tr, y_train, Xa_te, y_test))

# ── B. Hand-engineered features ──────────────────────────────────────────────
raw_eng = add_engineered(raw)
FEAT_ENG = [c for c in raw_eng.columns if c != TARGET]
Xe = raw_eng[FEAT_ENG].values
Xe_train, Xe_test = Xe[raw.index.isin(
    raw.iloc[: len(X_train)].index
)], Xe[~raw.index.isin(raw.iloc[: len(X_train)].index)]
# Simpler: just re-split with same seed
Xe_train, Xe_test, _, _ = train_test_split(
    Xe, y, test_size=0.20, random_state=42
)
scaler_b = StandardScaler()
Xb_tr = scaler_b.fit_transform(Xe_train)
Xb_te = scaler_b.transform(Xe_test)
model_b = RidgeCV(alphas=alphas)
model_b.fit(Xb_tr, y_train)
ridge_b = Ridge(alpha=model_b.alpha_)
results.append(evaluate("B) + Hand-engineered features", ridge_b, Xb_tr, y_train, Xb_te, y_test))

# ── C. Polynomial features (degree 2) on original ───────────────────────────
poly_c = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
Xc_tr = poly_c.fit_transform(X_train)
Xc_te = poly_c.transform(X_test)
scaler_c = StandardScaler()
Xc_tr = scaler_c.fit_transform(Xc_tr)
Xc_te = scaler_c.transform(Xc_te)
model_c = RidgeCV(alphas=alphas)
model_c.fit(Xc_tr, y_train)
ridge_c = Ridge(alpha=model_c.alpha_)
results.append(evaluate("C) Polynomial features (deg=2)", ridge_c, Xc_tr, y_train, Xc_te, y_test))

# ── D. Polynomial features on engineered set ────────────────────────────────
poly_d = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
Xd_tr = poly_d.fit_transform(Xe_train)
Xd_te = poly_d.transform(Xe_test)
scaler_d = StandardScaler()
Xd_tr = scaler_d.fit_transform(Xd_tr)
Xd_te = scaler_d.transform(Xd_te)
model_d = RidgeCV(alphas=alphas)
model_d.fit(Xd_tr, y_train)
ridge_d = Ridge(alpha=model_d.alpha_)
results.append(evaluate("D) Engineered + Poly interactions", ridge_d, Xd_tr, y_train, Xd_te, y_test))

# ── 3. Summary table ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
summary = pd.DataFrame(results)
summary["Δ R² vs baseline"] = summary["test_r2"] - results[0]["test_r2"]
print(summary[["name", "n_feat", "test_rmse", "test_mae", "test_r2", "cv_r2_mean", "Δ R² vs baseline"]].to_string(index=False))

best = summary.loc[summary["test_r2"].idxmax()]
print(f"\n🏆 Best model : {best['name']}")
print(f"   Test R²    : {best['test_r2']:.6f}")
print(f"   Test RMSE  : {best['test_rmse']:.6f}")
print(f"   Test MAE   : {best['test_mae']:.6f}")
print(f"   Improvement: +{best['Δ R² vs baseline']:.4f} R² over baseline")

# ── 4. Top features for best model (B) ──────────────────────────────────────
print(f"\n{'─' * 60}")
print("  Top 15 Feature Coefficients (best model)")
print(f"{'─' * 60}")
# Retrain B to access coefficients
ridge_b.fit(Xb_tr, y_train)
coef_df = pd.DataFrame({
    "feature": FEAT_ENG,
    "coefficient": ridge_b.coef_,
    "|coef|": np.abs(ridge_b.coef_),
}).sort_values("|coef|", ascending=False).head(15)
print(coef_df.to_string(index=False))

print("\n" + "=" * 60)
print("  Done ✓")
print("=" * 60)

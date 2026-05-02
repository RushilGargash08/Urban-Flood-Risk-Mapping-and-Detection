"""
Urban Flood Risk MLP Model
==========================
Targets:
  - Regression  : flood_risk_score  (0–1)
  - Classification: flood_risk_category (Low / Medium / High)
Metrics: R², RMSE, MAE, Confusion Matrix
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_auc_score
)
from sklearn.pipeline import Pipeline

# Load
df = pd.read_csv("/mnt/user-data/uploads/urban_flood_risk_dataset.csv")
print(f"Dataset shape: {df.shape}")

# Feature Engineering

# Region flag (Punjab = 1, Delhi-NCR = 0)
df["region"] = (df["lat"] > 30.0).astype(int)

# Non-linear interaction (key driver from README formula)
df["rain_x_builtup"] = df["rainfall_mm_per_day"] * df["built_up_percentage"] / 100

# Flood potential index: rain / (elevation + 1) — high rain + low elevation → high risk
df["flood_potential"] = df["rainfall_mm_per_day"] / (df["elevation_m"] + 1)

# Drainage relief: higher drainage on lower slopes → better relief
df["drainage_relief"] = df["drainage_density"] / (df["slope_percent"] + 0.01)

# SAR wetness proxy: backscatter + moisture combined
df["sar_wetness"] = df["sar_backscatter_coefficient"] * df["moisture_index"]

# Proximity risk: inverse distance to water body
df["proximity_risk"] = 1 / (df["distance_to_water_body_km"] + 0.1)

# Log-transform skewed features
df["log_elevation"]   = np.log1p(df["elevation_m"])
df["log_rainfall"]    = np.log1p(df["rainfall_mm_per_day"])
df["log_flood_pot"]   = np.log1p(df["flood_potential"])
df["log_proximity"]   = np.log1p(df["proximity_risk"])

# Encode soil_type (one-hot)
df = pd.get_dummies(df, columns=["soil_type"], drop_first=False)

print("\nEngineered features added. New shape:", df.shape)

# Prepare X, y
drop_cols = ["flood_risk_score", "flood_risk_category"]
X = df.drop(columns=drop_cols).astype(float)

y_reg   = df["flood_risk_score"].values
y_cat   = df["flood_risk_category"].values

# Encode class labels with a consistent ordering
class_order = ["Low", "Medium", "High"]
le = LabelEncoder()
le.fit(class_order)
y_clf = le.transform(y_cat)

print(f"\nFeatures used ({len(X.columns)}): {list(X.columns)}")
print(f"Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Train / Test Split  (stratified on class)
X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
    X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)

print(f"\nTrain: {len(X_tr)} | Test: {len(X_te)}")

# Mlp Regressor
reg_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        batch_size=64
    ))
])

reg_pipe.fit(X_tr, yr_tr)
yr_pred = reg_pipe.predict(X_te)

r2   = r2_score(yr_te, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_te, yr_pred))
mae  = mean_absolute_error(yr_te, yr_pred)

print("\n── REGRESSION METRICS ──────────────────")
print(f"  R²   : {r2:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")

# CV R²
cv_r2 = cross_val_score(reg_pipe, X, y_reg, cv=5, scoring="r2")
print(f"  5-Fold CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# Mlp Classifier
clf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        batch_size=64
    ))
])

clf_pipe.fit(X_tr, yc_tr)
yc_pred = clf_pipe.predict(X_te)
yc_prob = clf_pipe.predict_proba(X_te)

clf_acc = (yc_pred == yc_te).mean()
print("\n── CLASSIFICATION METRICS ──────────────")
print(f"  Accuracy : {clf_acc:.4f}")
print(classification_report(yc_te, yc_pred, target_names=le.classes_))

# ROC-AUC (OvR)
roc_auc = roc_auc_score(
    label_binarize(yc_te, classes=[0,1,2]), yc_prob, multi_class="ovr", average="macro"
)
print(f"  ROC-AUC (macro OvR): {roc_auc:.4f}")

# Plots
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#0f1117")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

DARK_BG  = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT   = "#4f8ef7"
GREEN    = "#2ecc71"
RED      = "#e74c3c"
ORANGE   = "#f39c12"
TEXT     = "#e8eaf0"
GRID     = "#2a2d3a"

PALETTE  = {"Low": GREEN, "Medium": ORANGE, "High": RED}
CAT_COLS = [PALETTE[c] for c in le.classes_]   # Low, Medium, High

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6)

# Predicted vs Actual (regression)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(yr_te, yr_pred, alpha=0.4, s=18, color=ACCENT, edgecolors="none")
lims = [min(yr_te.min(), yr_pred.min())-0.02, max(yr_te.max(), yr_pred.max())+0.02]
ax1.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
ax1.set_xlim(lims); ax1.set_ylim(lims)
ax1.set_xlabel("Actual"); ax1.set_ylabel("Predicted")
ax1.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT)
style_ax(ax1, "Predicted vs Actual (Regression)")

# Residual distribution
ax2 = fig.add_subplot(gs[0, 1])
residuals = yr_pred - yr_te
ax2.hist(residuals, bins=40, color=ACCENT, edgecolor=PANEL_BG, alpha=0.85)
ax2.axvline(0, color=RED, lw=1.5, linestyle="--")
ax2.set_xlabel("Residual"); ax2.set_ylabel("Count")
style_ax(ax2, "Residual Distribution")

# Metrics summary card
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(PANEL_BG)
ax3.axis("off")
metrics = [
    ("R²",            f"{r2:.4f}",    GREEN  if r2 > 0.85 else ORANGE),
    ("RMSE",          f"{rmse:.4f}",  GREEN  if rmse < 0.05 else ORANGE),
    ("MAE",           f"{mae:.4f}",   GREEN  if mae < 0.04 else ORANGE),
    ("CV R² (mean)",  f"{cv_r2.mean():.4f}", GREEN if cv_r2.mean() > 0.8 else ORANGE),
    ("Accuracy",      f"{clf_acc:.4f}", GREEN if clf_acc > 0.85 else ORANGE),
    ("ROC-AUC",       f"{roc_auc:.4f}", GREEN if roc_auc > 0.90 else ORANGE),
]
y_pos = 0.92
ax3.text(0.5, 1.02, "Model Performance Summary", transform=ax3.transAxes,
         ha="center", color=TEXT, fontsize=12, fontweight="bold")
for label, val, color in metrics:
    ax3.text(0.12, y_pos, label, transform=ax3.transAxes, color=TEXT, fontsize=10)
    ax3.text(0.72, y_pos, val,   transform=ax3.transAxes, color=color,  fontsize=10, fontweight="bold")
    y_pos -= 0.13
for sp in ax3.spines.values():
    sp.set_color(GRID)
ax3.set_title("Key Metrics", color=TEXT, fontsize=11, fontweight="bold", pad=10)

# Confusion Matrix
ax4 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(yc_te, yc_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(
    cm_pct, annot=True, fmt=".1f", cmap="Blues",
    xticklabels=le.classes_, yticklabels=le.classes_,
    ax=ax4, cbar_kws={"shrink": 0.8},
    linewidths=0.5, linecolor=PANEL_BG
)
ax4.set_xlabel("Predicted", color=TEXT); ax4.set_ylabel("Actual", color=TEXT)
ax4.tick_params(colors=TEXT)
ax4.set_title("Confusion Matrix (%)", color=TEXT, fontsize=11, fontweight="bold", pad=10)
ax4.set_facecolor(PANEL_BG)

# Per-class precision / recall / F1
ax5 = fig.add_subplot(gs[1, 1])
report = classification_report(yc_te, yc_pred, target_names=le.classes_, output_dict=True)
classes_show = le.classes_
metrics_show = ["precision", "recall", "f1-score"]
x = np.arange(len(classes_show))
w = 0.25
bar_colors = [GREEN, ACCENT, ORANGE]
for i, (metric, col) in enumerate(zip(metrics_show, bar_colors)):
    vals = [report[c][metric] for c in classes_show]
    ax5.bar(x + i*w, vals, w, label=metric.capitalize(), color=col, alpha=0.85)
ax5.set_xticks(x + w)
ax5.set_xticklabels(classes_show, color=TEXT)
ax5.set_ylim(0, 1.15)
ax5.set_ylabel("Score")
ax5.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT)
style_ax(ax5, "Per-Class P / R / F1")

# Feature importance (permutation-style via weight magnitude)
ax6 = fig.add_subplot(gs[1, 2])
# Use absolute sum of first hidden layer weights as proxy importance
scaler_fit  = reg_pipe.named_steps["scaler"]
mlp_weights = reg_pipe.named_steps["mlp"].coefs_[0]   # shape (n_features, hidden1)
feat_imp = np.abs(mlp_weights).mean(axis=1)
fi_df = pd.Series(feat_imp, index=X.columns).sort_values(ascending=True).tail(15)
colors_fi = [ACCENT]*len(fi_df)
ax6.barh(fi_df.index, fi_df.values, color=colors_fi, alpha=0.85)
ax6.set_xlabel("Mean |Weight|")
style_ax(ax6, "Top-15 Feature Importances (MLP)")

# Flood risk score distribution by region
ax7 = fig.add_subplot(gs[2, 0])
df["region_label"] = df["region"].map({1: "Punjab", 0: "Delhi-NCR"})
for region, col in [("Punjab", GREEN), ("Delhi-NCR", ACCENT)]:
    vals = df.loc[df["region_label"]==region, "flood_risk_score"]
    ax7.hist(vals, bins=30, alpha=0.65, color=col, label=region, edgecolor=PANEL_BG)
ax7.set_xlabel("Flood Risk Score"); ax7.set_ylabel("Count")
ax7.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT)
style_ax(ax7, "Risk Score by Region")

# Class distribution
ax8 = fig.add_subplot(gs[2, 1])
cat_counts = pd.Series(le.inverse_transform(yc_te)).value_counts().reindex(["Low","Medium","High"])
ax8.bar(cat_counts.index, cat_counts.values,
        color=[PALETTE[c] for c in cat_counts.index], alpha=0.85, edgecolor=PANEL_BG)
ax8.set_ylabel("Count")
style_ax(ax8, "Test Set Class Distribution")

# Regression error by risk band
ax9 = fig.add_subplot(gs[2, 2])
res_df = pd.DataFrame({"actual": yr_te, "predicted": yr_pred,
                        "label": le.inverse_transform(yc_te)})
for cat, col in PALETTE.items():
    sub = res_df[res_df["label"]==cat]
    ax9.scatter(sub["actual"], sub["predicted"], alpha=0.45, s=16,
                color=col, label=cat, edgecolors="none")
ax9.plot([0,1],[0,1],"r--",lw=1.3)
ax9.set_xlabel("Actual"); ax9.set_ylabel("Predicted")
ax9.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT)
style_ax(ax9, "Pred vs Actual by Risk Category")

# Master title
fig.suptitle(
    "Urban Flood Risk  –  MLP Neural Network Results",
    color=TEXT, fontsize=16, fontweight="bold", y=0.98
)

plt.savefig("/mnt/user-data/outputs/mlp_flood_risk_results.png",
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
print("\nPlot saved → mlp_flood_risk_results.png")

# Print final summary
print("        FINAL MODEL SUMMARY               ")
print(f"  Architecture : 128 → 64 → 32 (ReLU)    ")
print(f"  Optimizer    : Adam (lr=0.001)          ")
print(f"  Reg. R²      : {r2:.4f}                   ")
print(f"  Reg. RMSE    : {rmse:.4f}                  ")
print(f"  Reg. MAE     : {mae:.4f}                  ")
print(f"  CV R² (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}     ")
print(f"  Clf Accuracy : {clf_acc:.4f}                  ")
print(f"  ROC-AUC      : {roc_auc:.4f}                  ")
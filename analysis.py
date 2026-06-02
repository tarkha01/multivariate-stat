# ============================================================
# Multivariate Statistics - Final Project
# YSU Master's Program, Academic Year 2025/2026
# Dataset: WHO Life Expectancy (2000-2015)
# Source: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
#
# Run from terminal:
#   cd C:/Users/rozii/Desktop/ASDS/multivariate_stat
#   pip install -r requirements.txt
#   python analysis.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# ---------- plot style ----------
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 100
OUT = "."   # folder where PNG files are saved

print("=" * 50)
print("WHO Life Expectancy – Multivariate Analysis")
print("=" * 50)


# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
df_raw = pd.read_csv("Life Expectancy Data.csv")
df_raw.columns = (df_raw.columns
                  .str.strip()
                  .str.replace(" ", "_")
                  .str.replace(r"[^A-Za-z0-9_]", "", regex=True))

KEY_VARS = [
    "Life_expectancy",               # Dependent variable
    "Adult_Mortality",               # Deaths per 1000 adults
    "Alcohol",                       # Per-capita alcohol (litres)
    "BMI",                           # Average national BMI
    "HIVAIDS",                       # HIV/AIDS deaths per 1000 live births
    "GDP",                           # GDP per capita (USD)
    "Schooling",                     # Average years of schooling
    "Income_composition_of_resources"  # HDI income index
]

df = df_raw[["Country", "Status", "Year"] + KEY_VARS].dropna()
df_num = df[KEY_VARS].copy()

print(f"\nDataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Countries : {df['Country'].nunique()}")
print(f"Years     : {df['Year'].min()} – {df['Year'].max()}")
print(f"\nStatus breakdown:\n{df['Status'].value_counts()}")

print("\nSummary statistics:")
print(df_num.describe().round(2).to_string())


# ============================================================
# 2. SCATTER PLOT MATRIX  (3 variants as in the assignment)
# ============================================================
PLOT_VARS = ["Life_expectancy", "Adult_Mortality", "BMI", "GDP", "Schooling"]
colors = {"Developing": "black", "Developed": "red"}

# --- Version 1: basic colored by status ---
fig, axes = plt.subplots(len(PLOT_VARS), len(PLOT_VARS),
                          figsize=(10, 10), sharex="col", sharey="row")
for i, vi in enumerate(PLOT_VARS):
    for j, vj in enumerate(PLOT_VARS):
        ax = axes[i, j]
        if i == j:
            for status, grp in df.groupby("Status"):
                ax.hist(grp[vi], bins=20, alpha=0.6,
                        color=colors[status], density=True)
        else:
            for status, grp in df.groupby("Status"):
                ax.scatter(grp[vj], grp[vi], s=4, alpha=0.4,
                           color=colors[status])
        if i == len(PLOT_VARS) - 1:
            ax.set_xlabel(vj, fontsize=7)
        if j == 0:
            ax.set_ylabel(vi, fontsize=7)
        ax.tick_params(labelsize=6)

patches = [mpatches.Patch(color=c, label=s) for s, c in colors.items()]
fig.legend(handles=patches, loc="upper right", fontsize=8)
fig.suptitle("Scatter Plot Matrix  (Black = Developing, Red = Developed)",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_01_scattermatrix_basic.png"),
            bbox_inches="tight")
plt.close()

# --- Version 2: seaborn pairplot with KDE on diagonal ---
g2 = sns.pairplot(df[PLOT_VARS + ["Status"]],
                  hue="Status",
                  palette={"Developing": "steelblue", "Developed": "tomato"},
                  diag_kind="kde",
                  plot_kws=dict(alpha=0.3, s=10),
                  diag_kws=dict(fill=True))
g2.figure.suptitle("Scatter Plot Matrix with KDE Curves", y=1.01)
g2.savefig(os.path.join(OUT, "plot_02_scattermatrix_kde.png"),
           bbox_inches="tight")
plt.close()

# --- Version 3: seaborn pairplot with regression lines ---
g3 = sns.pairplot(df[PLOT_VARS],
                  kind="reg",
                  diag_kind="kde",
                  plot_kws=dict(scatter_kws=dict(alpha=0.2, s=5),
                                line_kws=dict(color="navy", lw=1),
                                ci=95),
                  diag_kws=dict(color="steelblue", fill=True))
g3.figure.suptitle("Scatter Plot Matrix with Regression Lines (95% CI)", y=1.01)
g3.savefig(os.path.join(OUT, "plot_03_scattermatrix_regression.png"),
           bbox_inches="tight")
plt.close()
print("\nScatter plot matrices saved.")


# ============================================================
# 3. NORMALITY ASSESSMENT
# ============================================================

# --- 3a. Shapiro-Wilk per variable ---
print("\n--- Shapiro-Wilk Test (H0: normal distribution) ---")
sw_results = {}
rng = np.random.default_rng(42)
for col in KEY_VARS:
    sample = rng.choice(df_num[col].values,
                        size=min(5000, len(df_num)), replace=False)
    stat, p = stats.shapiro(sample)
    sw_results[col] = {"W": round(stat, 4), "p-value": round(p, 6),
                       "Normal?": "YES" if p > 0.05 else "NO"}
sw_df = pd.DataFrame(sw_results).T
print(sw_df.to_string())

# --- 3b. Q-Q plots ---
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, col in zip(axes.flat, KEY_VARS):
    stats.probplot(df_num[col], dist="norm", plot=ax)
    ax.set_title(f"Q-Q: {col}", fontsize=9)
    ax.get_lines()[0].set(markersize=2, alpha=0.4, color="steelblue")
    ax.get_lines()[1].set(color="red", lw=1.5)
plt.suptitle("Q-Q Plots for All Variables", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_04_qqplots.png"), bbox_inches="tight")
plt.close()

# --- 3c. Histograms with KDE overlay ---
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, col in zip(axes.flat, KEY_VARS):
    ax.hist(df_num[col], bins=35, density=True,
            color="steelblue", alpha=0.6, edgecolor="white")
    kde = stats.gaussian_kde(df_num[col])
    xs  = np.linspace(df_num[col].min(), df_num[col].max(), 200)
    ax.plot(xs, kde(xs), color="red", lw=1.5)
    ax.set_title(col, fontsize=8)
    ax.tick_params(labelsize=7)
plt.suptitle("Distributions (Red = KDE Curve)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_05_histograms.png"), bbox_inches="tight")
plt.close()

# --- 3d. Mardia's multivariate normality test (manual implementation) ---
def mardia_test(X):
    """Returns Mardia skewness and kurtosis test statistics."""
    n, p = X.shape
    X_c = X - X.mean(axis=0)
    S   = np.cov(X_c.T)
    S_inv = np.linalg.pinv(S)
    # Skewness
    M = X_c @ S_inv @ X_c.T
    skewness = (M ** 3).sum() / (n ** 2)
    chi2_stat = n * skewness / 6
    df_skew  = p * (p + 1) * (p + 2) / 6
    p_skew   = 1 - stats.chi2.cdf(chi2_stat, df_skew)
    # Kurtosis
    kurtosis = np.trace(M @ M) / n
    z_kurt   = (kurtosis - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    p_kurt   = 2 * (1 - stats.norm.cdf(abs(z_kurt)))
    return {"Skewness stat": round(skewness, 4),
            "Chi2 (skew)":   round(chi2_stat, 4),
            "p (skew)":      round(p_skew, 6),
            "Kurtosis stat": round(kurtosis, 4),
            "Z (kurtosis)":  round(z_kurt, 4),
            "p (kurtosis)":  round(p_kurt, 6)}

print("\n--- Mardia Multivariate Normality Test ---")
mardia = mardia_test(df_num.values)
for k, v in mardia.items():
    print(f"  {k}: {v}")

# Log-transform skewed variables
df_log = df_num.copy()
df_log["GDP"]            = np.log1p(df_log["GDP"])
df_log["HIVAIDS"]        = np.log1p(df_log["HIVAIDS"])
df_log["Adult_Mortality"]= np.log1p(df_log["Adult_Mortality"])

print("\nMardia test after log-transforming GDP, HIV.AIDS, Adult_Mortality:")
mardia_log = mardia_test(df_log.values)
for k, v in mardia_log.items():
    print(f"  {k}: {v}")


# ============================================================
# 4. LINEAR REGRESSION
# ============================================================
formula = "Life_expectancy ~ Adult_Mortality + Alcohol + BMI + HIVAIDS + GDP + Schooling + Income_composition_of_resources"

model_full = smf.ols(formula, data=df_num).fit()
print("\n=== Full OLS Regression ===")
print(model_full.summary())

# --- 4a. Regression diagnostics ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

fitted  = model_full.fittedvalues
resid   = model_full.resid
std_res = resid / resid.std()

# Residuals vs Fitted
axes[0, 0].scatter(fitted, resid, s=5, alpha=0.3, color="steelblue")
axes[0, 0].axhline(0, color="red", lw=1)
axes[0, 0].set(xlabel="Fitted values", ylabel="Residuals",
               title="Residuals vs Fitted")

# Q-Q of residuals
stats.probplot(resid, plot=axes[0, 1])
axes[0, 1].set_title("Q-Q Plot of Residuals")
axes[0, 1].get_lines()[0].set(markersize=3, alpha=0.4)
axes[0, 1].get_lines()[1].set(color="red")

# Scale-location
axes[1, 0].scatter(fitted, np.sqrt(np.abs(std_res)),
                   s=5, alpha=0.3, color="steelblue")
axes[1, 0].set(xlabel="Fitted values", ylabel="|Std. Residuals|^0.5",
               title="Scale-Location")

# Residuals histogram
axes[1, 1].hist(resid, bins=40, color="steelblue",
                alpha=0.7, edgecolor="white", density=True)
xs = np.linspace(resid.min(), resid.max(), 200)
axes[1, 1].plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()),
                color="red", lw=1.5)
axes[1, 1].set(xlabel="Residuals", title="Residual Distribution")

plt.suptitle("Linear Regression Diagnostics", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_06_regression_diagnostics.png"),
            bbox_inches="tight")
plt.close()

# --- 4b. Variance Inflation Factors ---
X_vif = sm.add_constant(df_num.drop(columns="Life_expectancy"))
vif_data = pd.DataFrame({
    "Variable": X_vif.columns[1:],
    "VIF": [variance_inflation_factor(X_vif.values, i+1)
            for i in range(X_vif.shape[1] - 1)]
}).set_index("Variable").round(2)
print("\n--- VIF (>5 indicates multicollinearity) ---")
print(vif_data.to_string())


# ============================================================
# 5. VARIABLE SELECTION
# ============================================================

# --- 5a. Forward stepwise by AIC ---
def stepwise_aic(df_data, dependent, candidates):
    """Forward-backward stepwise regression minimising AIC."""
    selected, remaining = [], list(candidates)
    current_aic = smf.ols(f"{dependent} ~ 1", data=df_data).fit().aic
    improved = True
    while improved and remaining:
        improved = False
        aic_candidates = {}
        for var in remaining:
            formula_try = f"{dependent} ~ {' + '.join(selected + [var])}"
            aic_try = smf.ols(formula_try, data=df_data).fit().aic
            aic_candidates[var] = aic_try
        best_var = min(aic_candidates, key=aic_candidates.get)
        if aic_candidates[best_var] < current_aic:
            selected.append(best_var)
            remaining.remove(best_var)
            current_aic = aic_candidates[best_var]
            improved = True
    # Backward pass: drop variables that no longer help
    for var in selected[:]:
        formula_try = f"{dependent} ~ {' + '.join(v for v in selected if v != var)}"
        if not formula_try.endswith("~"):
            aic_try = smf.ols(formula_try, data=df_data).fit().aic
            if aic_try <= current_aic:
                selected.remove(var)
                current_aic = aic_try
    return selected

predictors = [c for c in KEY_VARS if c != "Life_expectancy"]
step_vars   = stepwise_aic(df_num, "Life_expectancy", predictors)
model_step  = smf.ols(
    f"Life_expectancy ~ {' + '.join(step_vars)}", data=df_num
).fit()
print(f"\n--- Stepwise AIC selected: {step_vars} ---")
print(model_step.summary().tables[0])

# --- 5b. LASSO with cross-validation ---
X_mat = df_num[predictors].values
y_vec = df_num["Life_expectancy"].values
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_mat)

lasso_cv = LassoCV(cv=10, random_state=42, max_iter=5000).fit(X_sc, y_vec)
lasso_coefs = pd.Series(lasso_cv.coef_, index=predictors)
lasso_selected = lasso_coefs[lasso_coefs != 0].index.tolist()
print(f"\n--- LASSO selected (optimal lambda = {lasso_cv.alpha_:.4f}) ---")
print(lasso_coefs.round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1),
            color="steelblue", lw=1.5)
ax.axvline(lasso_cv.alpha_, color="red", lw=1.5, linestyle="--",
           label=f"Optimal λ = {lasso_cv.alpha_:.4f}")
ax.set(xlabel="λ (regularisation strength)", ylabel="Mean CV MSE",
       title="LASSO Cross-Validation Path")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_08_lasso_cv.png"), bbox_inches="tight")
plt.close()

# --- 5c. Model comparison table ---
model_lasso_fit = smf.ols(
    f"Life_expectancy ~ {' + '.join(lasso_selected)}", data=df_num
).fit()

comparison = pd.DataFrame({
    "Model":       ["Full (7 vars)", "Stepwise AIC", "LASSO"],
    "R²":          [model_full.rsquared, model_step.rsquared,
                    model_lasso_fit.rsquared],
    "Adj. R²":     [model_full.rsquared_adj, model_step.rsquared_adj,
                    model_lasso_fit.rsquared_adj],
    "AIC":         [model_full.aic, model_step.aic, model_lasso_fit.aic],
    "# Variables": [7, len(step_vars), len(lasso_selected)],
}).round(4)
print("\n--- Model Comparison ---")
print(comparison.to_string(index=False))

# Best subsets: AdjR² for k = 1 … (p-1)
adjr2_list = []
for n in range(1, len(predictors)):   # must be < n_features
    sfs_n = SequentialFeatureSelector(LinearRegression(),
                                       n_features_to_select=n,
                                       direction="forward", cv=5)
    sfs_n.fit(X_sc, y_vec)
    sel_names = [predictors[i] for i in range(len(predictors))
                 if sfs_n.get_support()[i]]
    m = smf.ols(f"Life_expectancy ~ {' + '.join(sel_names)}", data=df_num).fit()
    adjr2_list.append((n, round(m.rsquared_adj, 4), round(m.aic, 1),
                        ", ".join(sel_names)))

best_df = pd.DataFrame(adjr2_list,
                        columns=["# Vars", "Adj. R²", "AIC", "Variables selected"])
print("\n--- Forward Subset Selection ---")
print(best_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot([r[0] for r in adjr2_list], [r[1] for r in adjr2_list],
             "o-", color="steelblue")
axes[0].set(xlabel="# Variables", ylabel="Adj. R²", title="Adj. R² vs # Variables")
axes[1].plot([r[0] for r in adjr2_list], [r[2] for r in adjr2_list],
             "o-", color="tomato")
axes[1].set(xlabel="# Variables", ylabel="AIC", title="AIC vs # Variables")
plt.suptitle("Forward Subset Selection", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_07_variable_selection.png"),
            bbox_inches="tight")
plt.close()


# ============================================================
# 6. PRINCIPAL COMPONENT ANALYSIS
# ============================================================
scaler_pca = StandardScaler()
X_pca_sc   = scaler_pca.fit_transform(df_num.values)

pca = PCA(n_components=len(KEY_VARS))
pca.fit(X_pca_sc)
scores      = pca.transform(X_pca_sc)
loadings    = pca.components_.T   # shape (n_vars, n_components)
explained   = pca.explained_variance_ratio_ * 100

print("\n=== PCA Eigenvalues & Explained Variance ===")
eig_df = pd.DataFrame({
    "PC":               [f"PC{i+1}" for i in range(len(KEY_VARS))],
    "Eigenvalue":       pca.explained_variance_.round(3),
    "Variance (%)":     explained.round(2),
    "Cumulative (%)":   np.cumsum(explained).round(2),
})
print(eig_df.to_string(index=False))

print("\nLoadings (PC1 – PC4):")
load_df = pd.DataFrame(loadings[:, :4], index=KEY_VARS,
                        columns=[f"PC{i+1}" for i in range(4)]).round(3)
print(load_df.to_string())

# --- 6a. Scree plot ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(range(1, len(KEY_VARS)+1), explained,
            color="steelblue", edgecolor="white")
axes[0].plot(range(1, len(KEY_VARS)+1), explained, "o-",
             color="navy", lw=1.5)
for i, v in enumerate(explained):
    axes[0].text(i+1, v+0.5, f"{v:.1f}%", ha="center", fontsize=8)
axes[0].axhline(y=100/len(KEY_VARS), color="red", linestyle="--",
                lw=1, label="Equal share")
axes[0].set(xlabel="Principal Component", ylabel="Variance Explained (%)",
            title="Scree Plot")
axes[0].legend(fontsize=8)

axes[1].plot(range(1, len(KEY_VARS)+1), np.cumsum(explained),
             "o-", color="seagreen", lw=1.5)
axes[1].axhline(80, color="red", linestyle="--", lw=1, label="80% threshold")
axes[1].set(xlabel="Number of PCs", ylabel="Cumulative Variance (%)",
            title="Cumulative Variance")
axes[1].legend(fontsize=8)

plt.suptitle("PCA – Scree Plot", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_09_screeplot.png"), bbox_inches="tight")
plt.close()


# ============================================================
# 7. CORRELATION CIRCLE
# ============================================================
pc1_var = explained[0]
pc2_var = explained[1]

# Correlation between original variables and PCs
corr_circle = loadings[:, :2] * np.sqrt(pca.explained_variance_[:2])

fig, ax = plt.subplots(figsize=(7, 7))
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), color="gray", lw=1)
ax.axhline(0, color="gray", lw=0.8, linestyle="--")
ax.axvline(0, color="gray", lw=0.8, linestyle="--")

# Color arrows by cos² (quality of representation)
cos2 = corr_circle[:, 0]**2 + corr_circle[:, 1]**2
cmap = plt.cm.RdYlBu_r
norm = plt.Normalize(cos2.min(), cos2.max())

for i, var in enumerate(KEY_VARS):
    color = cmap(norm(cos2[i]))
    ax.annotate("", xy=(corr_circle[i, 0], corr_circle[i, 1]),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2))
    offset_x = 0.05 if corr_circle[i, 0] >= 0 else -0.05
    offset_y = 0.05 if corr_circle[i, 1] >= 0 else -0.05
    ax.text(corr_circle[i, 0] + offset_x,
            corr_circle[i, 1] + offset_y,
            var.replace("_", "\n"), fontsize=8, ha="center",
            color="darkblue", fontweight="bold")

sm_patch = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm_patch.set_array([])
plt.colorbar(sm_patch, ax=ax, label="cos² (quality of representation)",
             fraction=0.046, pad=0.04)

ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2),
       xlabel=f"PC1 ({pc1_var:.1f}% variance)",
       ylabel=f"PC2 ({pc2_var:.1f}% variance)",
       title="Correlation Circle – Variables on PC1–PC2",
       aspect="equal")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_10_correlation_circle.png"),
            bbox_inches="tight")
plt.close()
print("Correlation circle saved.")


# ============================================================
# 8. INDIVIDUAL PROJECTIONS ONTO PC1–PC2
# ============================================================
status_colors = {"Developing": "steelblue", "Developed": "tomato"}
status_vals   = df["Status"].values

fig, ax = plt.subplots(figsize=(9, 7))
for status, color in status_colors.items():
    mask = status_vals == status
    ax.scatter(scores[mask, 0], scores[mask, 1],
               s=18, alpha=0.55, color=color, label=status)
    # 95% confidence ellipse
    pts = scores[mask, :2]
    mean = pts.mean(axis=0)
    cov  = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    chi2_val = stats.chi2.ppf(0.95, df=2)
    angle    = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    w, h     = 2 * np.sqrt(chi2_val * eigvals)
    from matplotlib.patches import Ellipse
    ell = Ellipse(xy=mean, width=w, height=h, angle=angle,
                  edgecolor=color, facecolor=color, alpha=0.12, lw=1.5)
    ax.add_patch(ell)

ax.set(xlabel=f"PC1 ({pc1_var:.1f}% variance)",
       ylabel=f"PC2 ({pc2_var:.1f}% variance)",
       title="Countries Projected onto PC1–PC2 (95% Confidence Ellipses)")
ax.axhline(0, color="gray", lw=0.8, linestyle="--")
ax.axvline(0, color="gray", lw=0.8, linestyle="--")
ax.legend(title="Development Status")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_11_pca_individuals.png"),
            bbox_inches="tight")
plt.close()

# Biplot (individuals + variable arrows)
fig, ax = plt.subplots(figsize=(10, 8))
for status, color in status_colors.items():
    mask = status_vals == status
    ax.scatter(scores[mask, 0], scores[mask, 1],
               s=10, alpha=0.35, color=color, label=status)

scale = scores[:, :2].std(axis=0).mean() / corr_circle.std(axis=0).mean() * 0.7
for i, var in enumerate(KEY_VARS):
    ax.annotate("", xy=(corr_circle[i, 0]*scale, corr_circle[i, 1]*scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(corr_circle[i, 0]*scale*1.1, corr_circle[i, 1]*scale*1.1,
            var.replace("_", "\n"), fontsize=7, color="black", ha="center")

ax.set(xlabel=f"PC1 ({pc1_var:.1f}%)",
       ylabel=f"PC2 ({pc2_var:.1f}%)",
       title="PCA Biplot – Countries and Variables")
ax.axhline(0, color="gray", lw=0.7, linestyle="--")
ax.axvline(0, color="gray", lw=0.7, linestyle="--")
ax.legend(title="Status")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_12_biplot.png"), bbox_inches="tight")
plt.close()
print("Individual projection plots saved.")


# ============================================================
# 9. CORRELATION MATRIX
# ============================================================
cor_mat = df_num.corr()
print("\n=== Correlation Matrix ===")
print(cor_mat.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 8))
mask = np.triu(np.ones_like(cor_mat, dtype=bool))   # upper triangle only
sns.heatmap(cor_mat, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, square=True,
            ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Pearson Correlation Matrix", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_13_corrplot.png"), bbox_inches="tight")
plt.close()


# ============================================================
# 10. ADDITIONAL: HIERARCHICAL CLUSTERING
# ============================================================
X_sc_cl = StandardScaler().fit_transform(df_num.values)
Z = linkage(X_sc_cl, method="ward")

fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-3, 2])
ax.set(title="Hierarchical Clustering Dendrogram (Ward's Method)",
       xlabel="Observations", ylabel="Distance")
ax.axhline(Z[-3, 2], color="red", lw=1.5, linestyle="--",
           label="Cut for k=3")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_14_dendrogram.png"), bbox_inches="tight")
plt.close()

# Assign 3 clusters
df["cluster"] = fcluster(Z, t=3, criterion="maxclust").astype(str)
cluster_colors = {"1": "steelblue", "2": "tomato", "3": "seagreen"}

fig, ax = plt.subplots(figsize=(9, 7))
for cl, color in cluster_colors.items():
    mask = df["cluster"] == cl
    ax.scatter(scores[mask.values, 0], scores[mask.values, 1],
               s=18, alpha=0.6, color=color, label=f"Cluster {cl}")
ax.set(xlabel=f"PC1 ({pc1_var:.1f}%)",
       ylabel=f"PC2 ({pc2_var:.1f}%)",
       title="Hierarchical Clusters on PC1–PC2")
ax.axhline(0, color="gray", lw=0.7, linestyle="--")
ax.axvline(0, color="gray", lw=0.7, linestyle="--")
ax.legend(title="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_15_clusters_pca.png"), bbox_inches="tight")
plt.close()

# Cluster mean profiles
print("\n--- Cluster Profiles (mean per cluster) ---")
profile = df_num.copy()
profile["cluster"] = df["cluster"]
print(profile.groupby("cluster").mean().round(2).to_string())


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Observations : {len(df_num)}")
print(f"Variables    : {len(KEY_VARS)}")
print(f"Full model R²: {model_full.rsquared:.4f}")
print(f"Adj. R²      : {model_full.rsquared_adj:.4f}")
print(f"PC1 variance : {pc1_var:.1f}%")
print(f"PC2 variance : {pc2_var:.1f}%")
print(f"PC1+PC2 total: {pc1_var + pc2_var:.1f}%")
print(f"Stepwise vars: {', '.join(step_vars)}")
print(f"LASSO vars   : {', '.join(lasso_selected)}")
print(f"\n15 PNG plots saved to: {os.path.abspath(OUT)}")
print("=" * 50)

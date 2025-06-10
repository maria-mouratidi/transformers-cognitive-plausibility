import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load results
attn_method, task, model_name = "raw", "task2", "llama"
results_path = f"outputs/{attn_method}/{task}/{model_name}/ols/ols_results_summary.csv"
results_df = pd.read_csv(results_path)

# Remove intercept for plotting (optional, as it can dominate the scale)
results_df = results_df[results_df["predictor"] != "intercept"]

# Set up plot aesthetics
sns.set(style="whitegrid", font_scale=1.1)

# Set save directory to match results location
save_dir = os.path.dirname(results_path)

# Example 1: Coefficient heatmap for all models and targets
pivot_coef = results_df.pivot_table(
    index=["predictor"],
    columns=["model", "target"],
    values="coef"
)

plt.figure(figsize=(14, 8))
sns.heatmap(pivot_coef, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("OLS Coefficients by Predictor, Model, and Target")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ols_coefficients_heatmap.png"))
plt.show()

# Example 2: Barplot of coefficients for each model/target (grouped)
plt.figure(figsize=(16, 7))
sns.barplot(
    data=results_df,
    x="predictor",
    y="coef",
    hue="model",
    errorbar=None,
    dodge=True
)
plt.title("OLS Coefficients by Predictor and Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ols_coefficients_barplot.png"))
plt.show()

# Example 3: Highlight significant predictors (p < 0.05)
results_df["significant"] = results_df["pval"] < 0.05
plt.figure(figsize=(16, 7))
sns.scatterplot(
    data=results_df,
    x="predictor",
    y="coef",
    hue="model",
    style="significant",
    size="rsquared",
    sizes=(40, 200),
    alpha=0.8
)
plt.axhline(0, color="gray", linestyle="--")
plt.title("OLS Coefficients (Significant Predictors Highlighted)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ols_coefficients_significance.png"))
plt.show()

# Performance analysis
results_path = f"outputs/{attn_method}/{task}/{model_name}/ols/model_performance.csv"
performance_df = pd.read_csv(results_path)

# Group for error bars (mean ± std)
summary = performance_df.groupby(["model", "type"]).agg(
    r2_mean=("r2", "mean"),
    r2_std=("r2", "std"),
    rmse_mean=("rmse", "mean"),
    rmse_std=("rmse", "std")
).reset_index()

print("Summary for plotting:")
print(summary)

# Plot R² comparison (including Baseline)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=summary,
    x="type",
    y="r2_mean",
    hue="model",
    palette="Set2",
    capsize=0.15,
    errorbar="sd"
)
plt.ylabel("Test $R^2$")
plt.title("Model Comparison: Test $R^2$ by Model and Target Type")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(results_path), "model_comparison_r2.png"))
plt.show()

# Plot RMSE comparison (including Baseline)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=summary,
    x="type",
    y="rmse_mean",
    hue="model",
    palette="Set2",
    capsize=0.15,
    errorbar="sd"
)
plt.ylabel("Test RMSE")
plt.title("Model Comparison: Test RMSE by Model and Target Type")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(results_path), "model_comparison_rmse.png"))
plt.show()
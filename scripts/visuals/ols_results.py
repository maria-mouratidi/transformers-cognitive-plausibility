import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

import pandas as pd
import glob
import os

# --- CONFIG ---
RESULTS_ROOT = "outputs"
MODEL_PERF_FILENAME = "model_performance.csv"
ATTN_METHODS = ["raw", "mean", "other"]  # adjust as needed
TASKS = ["task1", "task2"]               # adjust as needed
MODEL_TYPES = ["TextOnly", "TextAttn", "Baseline"]  # adjust as needed
TARGET_TYPES = ["Gaze", "PCA"]

# --- LOAD ALL RESULTS ---
all_perf = []
for attn_method in ATTN_METHODS:
    for task in TASKS:
        # Find all model_performance.csv files for this attn_method/task
        search_path = os.path.join(RESULTS_ROOT, attn_method, task, "*", "ols", MODEL_PERF_FILENAME)
        for perf_path in glob.glob(search_path):
            df = pd.read_csv(perf_path)
            df["results_path"] = perf_path
            all_perf.append(df)
perf_df = pd.concat(all_perf, ignore_index=True)

# --- PREPROCESS ---
# Extract model type and target type from OLS_model
def parse_model_type(ols_model):
    if "TextAttn" in ols_model:
        return "Attention"
    elif "TextOnly" in ols_model:
        return "TextOnly"
    elif "Baseline" in ols_model:
        return "Baseline"
    else:
        return "Other"

def parse_target_type(ols_model):
    if "Gaze" in ols_model:
        return "Gaze"
    elif "PCA" in ols_model:
        return "PCA"
    else:
        return "Other"

perf_df["ModelType"] = perf_df["OLS_model"].apply(parse_model_type)
perf_df["TargetType"] = perf_df["OLS_model"].apply(parse_target_type)

# --- PLOT ---
plt.figure(figsize=(12, 8))
# Create a y-axis label combining attention method and target type
perf_df["y_label"] = perf_df["attention_method"] + " | " + perf_df["TargetType"]

# Only keep relevant models
plot_df = perf_df[perf_df["ModelType"].isin(["Attention", "TextOnly", "Baseline"])]

# Sort for nice plotting
plot_df = plot_df.sort_values(["attention_method", "TargetType", "task", "ModelType"])

# Barplot: y=attention method + target type, x=R2, hue=model type, col=task
g = sns.catplot(
    data=plot_df,
    kind="bar",
    y="y_label",
    x="rsquared",
    hue="ModelType",
    col="task",
    ci="sd",
    palette="Set2",
    height=6,
    aspect=1.2,
    dodge=True
)

g.set_axis_labels("Test $R^2$", "Attention Method | Target Type")
g.set_titles("Task: {col_name}")
g._legend.set_title("Model Type")

# --- WILCOXON ANNOTATIONS ---
# If Wilcoxon p-values are present, annotate significant comparisons
for ax, task in zip(g.axes.flat, plot_df["task"].unique()):
    task_df = plot_df[plot_df["task"] == task]
    for ytick, (y_label, group) in enumerate(task_df.groupby("y_label")):
        # Compare Attention vs TextOnly
        attn_r2 = group[group["ModelType"] == "Attention"]["rsquared"]
        text_r2 = group[group["ModelType"] == "TextOnly"]["rsquared"]
        if not attn_r2.empty and not text_r2.empty:
            from scipy.stats import wilcoxon
            try:
                stat, p = wilcoxon(attn_r2, text_r2)
                if p < 0.05:
                    ax.annotate(
                        "*",
                        xy=(max(attn_r2.mean(), text_r2.mean()), ytick),
                        xytext=(10, 0),
                        textcoords="offset points",
                        va="center",
                        color="red",
                        fontsize=18
                    )
            except Exception:
                pass

plt.tight_layout()
plt.savefig("outputs/unified_ols_barplot.png")
plt.show()
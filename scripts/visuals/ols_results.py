import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

all_perf = []
for perf_path in glob.glob("outputs/**/**/**/ols/model_performance.csv", recursive=True):
    df = pd.read_csv(perf_path)
    df["source_path"] = perf_path
    all_perf.append(df)

if all_perf:
    perf_df = pd.concat(all_perf, ignore_index=True)
    perf_df.to_csv("outputs/unified_model_performance.csv", index=False)
    print("Unified performance CSV created at outputs/unified_model_performance.csv")
else:
    print("No model_performance.csv files found.")

# --- LOAD UNIFIED RESULTS ---
perf_df = pd.read_csv("outputs/unified_model_performance.csv")

# --- PREPROCESS ---
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

# Only keep relevant models
plot_df = perf_df[perf_df["ModelType"].isin(["Attention", "TextOnly", "Baseline"])]

# Create a new x-axis label combining task and target type
plot_df["x_label"] = plot_df["task"] + " - " + plot_df["TargetType"]

# --- PLOT ---
plt.figure(figsize=(14, 7))
sns.set(style="whitegrid")

# Facet by attention_method (one row per attention method)
g = sns.catplot(
    data=plot_df,
    kind="bar",
    x="x_label",
    y="rsquared",
    hue="ModelType",
    row="attention_method",
    ci="sd",
    palette="Set2",
    height=4,
    aspect=2,
    dodge=True
)

g.set_axis_labels("Task & Target", "Test $R^2$")
g.set_titles("Attention Method: {row_name}")
g._legend.set_title("Model Type")

# --- WILCOXON ANNOTATIONS ---
from scipy.stats import wilcoxon

for ax, (attn_method, attn_df) in zip(g.axes.flat, plot_df.groupby("attention_method")):
    for xtick, (x_label, group) in enumerate(attn_df.groupby("x_label")):
        attn_r2 = group[group["ModelType"] == "Attention"]["rsquared"]
        text_r2 = group[group["ModelType"] == "TextOnly"]["rsquared"]
        if not attn_r2.empty and not text_r2.empty:
            try:
                stat, p = wilcoxon(attn_r2, text_r2)
                if p < 0.05:
                    ax.annotate(
                        "*",
                        xy=(xtick, max(attn_r2.mean(), text_r2.mean())),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        color="red",
                        fontsize=18
                    )
            except Exception:
                pass

plt.tight_layout()
plt.savefig("outputs/unified_ols_barplot.png")
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from scripts.analysis.correlation import FEATURES


llm_models = ["llama", "bert"]
tasks = ["task2", "task3"]
attn_methods = ["raw", "flow", "saliency"]
# all_perf = []
# for model_name in llm_models:
#     for task in tasks:
#         for attn_method in attn_methods:
#             perf_path = f"outputs/{attn_method}/{task}/{model_name}/ols/model_performance.csv"
#             # if attn_method == "flow" and model_name == "bert":
#             #     continue
#             df = pd.read_csv(perf_path)
#             df["source_path"] = perf_path
#             all_perf.append(df)

# if all_perf:
#     perf_df = pd.concat(all_perf, ignore_index=True)
#     perf_df.to_csv("outputs/unified_model_performance.csv", index=False)
#     print("Unified performance CSV created at outputs/unified_model_performance.csv")

# --- LOAD UNIFIED RESULTS ---
perf_df = pd.read_csv("outputs/ols_unified_performance.csv")

# --- AGGREGATE TO MODEL LEVEL ---
# Only keep one row per model/run (aggregate by mean, since rsquared/rmse are repeated per feature)
group_cols = [
    "task", "llm_model", "attn_method", "predictors", "dependent"
]
agg_cols = {
    "rsquared": "mean",
    "rsquared_adj": "mean",
    "rmse": "mean"
}
perf_df = perf_df.groupby(group_cols, as_index=False).agg(agg_cols)

# --- PREPROCESS ---
from scripts.analysis.correlation import FEATURES

# ModelType: Attention or TextOnly
def get_model_type(row):
    if row["predictors"] == "text_only":
        return "TextOnly"
    elif row["attn_method"] in ["raw", "flow", "saliency"]:
        return "Attention"
    else:
        return "Other"

perf_df["ModelType"] = perf_df.apply(get_model_type, axis=1)

# TargetType: Gaze, PCA, Other
perf_df["TargetType"] = perf_df["dependent"].apply(
    lambda x: "Gaze" if x in FEATURES else ("PCA" if x == "pca" else "Other")
)

# Add attention_method column for plotting
perf_df["attention_method"] = perf_df["attn_method"]

# Only keep relevant models
plot_df = perf_df[perf_df["ModelType"].isin(["Attention", "TextOnly"])]

def plot_metric(plot_df, metric, filename):
    plot_df = plot_df.copy()
    plot_df["bar_type"] = plot_df.apply(
        lambda row: "text only" if row["ModelType"] == "TextOnly"
        else row["attention_method"], axis=1
    )
    plot_df = plot_df[
        ((plot_df["ModelType"] == "TextOnly") & (plot_df["bar_type"] == "text only")) |
        ((plot_df["ModelType"] == "Attention") & (plot_df["bar_type"].isin(["raw", "flow", "saliency"])))
    ]
    plot_df["x_label"] = plot_df["TargetType"] + " - " + plot_df["task"].astype(str)
    target_order = ["Gaze", "PCA"]
    task_order = sorted(plot_df["task"].unique())
    x_order = [f"{target} - {task}" for target in target_order for task in task_order]
    hue_order = ["text only", "raw", "flow", "saliency"]

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=plot_df, x="x_label", y=metric, hue="bar_type",
        order=x_order, hue_order=hue_order, errorbar="sd"
    )
    ax.set_xlabel("Target & Task")
    ax.set_ylabel(f"{metric.capitalize()}")
    ax.set_title(f"Ordinary Least Squares Models")
    ax.set_ylim(top=0.5)
    ax.legend(title="Predictors")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- PLOT FOR ALL MODELS EXCEPT BERT ---
not_bert_df = plot_df[plot_df["llm_model"] != "bert"]
plot_metric(not_bert_df, "rsquared", "outputs/ols_barplot_r2_llama.png")

# --- PLOT FOR BERT ONLY ---
bert_df = plot_df[plot_df["llm_model"] == "bert"]
if not bert_df.empty:
    plot_metric(bert_df, "rsquared", "outputs/ols_barplot_r2_bert.png")
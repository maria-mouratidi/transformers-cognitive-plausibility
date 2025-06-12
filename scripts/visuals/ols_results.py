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

perf_df["ModelType"] = perf_df["predictors"].apply(lambda x: "Attention" if x in ["raw", "flow", "saliency"] else "TextOnly")
perf_df["TargetType"] = perf_df["dependent"].apply(lambda x: "Gaze" if x in FEATURES else ("PCA" if x == "pca" else "Other"))

# Only keep relevant models
plot_df = perf_df[perf_df["ModelType"].isin(["Attention", "TextOnly"])]

def plot_metric(plot_df, metric, filename):

    # Create a bar_type column for grouping
    plot_df = plot_df.copy()
    plot_df["bar_type"] = plot_df.apply(
        lambda row: "text only" if row["ModelType"] == "TextOnly"
        else row["attention_method"], axis=1
    )

    # bars for text only, or attention with raw/flow/saliency
    plot_df = plot_df[
        ((plot_df["ModelType"] == "TextOnly") & (plot_df["bar_type"] == "text_only")) |
        ((plot_df["ModelType"] == "Attention") & (plot_df["bar_type"].isin(["raw", "flow", "saliency"])))
    ]

    # Set x_label
    plot_df["x_label"] = plot_df["TargetType"] + " - " + plot_df["task"].astype(str)

    # Set order for x and hue
    target_order = ["Gaze", "PCA"]
    task_order = sorted(plot_df["task"].unique())
    x_order = [f"{target} - {task}" for target in target_order for task in task_order]
    hue_order = ["text only", "raw", "flow", "saliency"]

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=plot_df, x="x_label", y=metric, hue="bar_type", order=x_order, hue_order=hue_order, errorbar="sd")

    ax.set_xlabel("Target & Task")
    ax.set_ylabel(f"{metric.capitalize()}")
    ax.set_title(f"Ordinary Least Squares Models")
    ax.set_ylim(top=0.5)  # Set max y-tick to 0.5
    ax.legend(title="Predictors")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- PLOT FOR ALL MODELS EXCEPT BERT ---
not_bert_df = plot_df[plot_df["llm_model"] != "bert"]
plot_metric(not_bert_df, "rsquared", "outputs/ols_barplot_r2_llama.png")
#plot_metric(not_bert_df, "rmse", "outputs/ols_barplot_rmse_llama.png")

# --- PLOT FOR BERT ONLY ---
bert_df = plot_df[plot_df["llm_model"] == "bert"]
if not bert_df.empty:
    plot_metric(bert_df, "rsquared", "outputs/ols_barplot_r2_bert.png")
    #plot_metric(bert_df, "rmse", "outputs/ols_barplot_rmse_bert.png")
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
        else f"text+{row['attention_method']}", axis=1
    )
    plot_df = plot_df[
        ((plot_df["ModelType"] == "TextOnly") & (plot_df["bar_type"] == "text only")) |
        ((plot_df["ModelType"] == "Attention") & (plot_df["bar_type"].isin(["text+raw", "text+flow", "text+saliency"])))
    ]
    plot_df["x_label"] = plot_df["TargetType"] + " - " + plot_df["task"].astype(str)
    target_order = ["Gaze", "PCA"]
    task_order = sorted(plot_df["task"].unique())
    x_order = [f"{target} - {task}" for target in target_order for task in task_order]
    hue_order = ["text only", "text+raw", "text+flow", "text+saliency"]

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=plot_df, x="x_label", y=metric, hue="bar_type",
        order=x_order, hue_order=hue_order, errorbar="sd"
    )
    ax.set_xlabel("Target & Task")
    ax.set_ylabel(f"{metric.capitalize()}")
    ax.set_title(f"Ordinary Least Squares Models")
    ax.set_ylim(top=0.5)
    ax.legend(title="Models")
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

def plot_attention_feature_importances(perf_path, filename_prefix, llm_model):
    df = pd.read_csv(perf_path)
    # Only keep attention models and remove constant
    df = df[df["attn_method"].isin(["raw", "flow", "saliency"])]
    df = df[df["feature_name"] != "const"]
    # Remove attention_layer_1 and attention_layer_31
    df = df[~df["feature_name"].isin(["attention_layer_1", "attention_layer_31"])]
    # Only keep rows for the specified llm_model
    df = df[df["llm_model"] == llm_model]
    # Add TargetType and x_label for grouping
    df["TargetType"] = df["dependent"].apply(
        lambda x: "Gaze" if x in FEATURES else ("PCA" if x == "pca" else "Other")
    )
    df["x_label"] = df["TargetType"] + " - " + df["task"].astype(str)
    # Set model label as in predictors
    df["Model"] = df["attn_method"].apply(lambda x: f"text+{x}")
    # Rename only attention_layer_0 to the LLM model name
    def rename_feature(feat):
        if feat == "attention_layer_0":
            return llm_model
        return feat
    df["feature_name"] = df["feature_name"].apply(rename_feature)
    # Group by x_label, Model, feature_name
    summary = (
        df.groupby(["TargetType", "x_label", "feature_name", "Model"])
        .agg(mean_abs_t=("t", lambda x: x.abs().mean()))
        .reset_index()
    )
    # Ensure the model name is first in feature order
    feature_order = [llm_model] + sorted([f for f in summary["feature_name"].unique() if f != llm_model])
    model_order = [f"text+raw", f"text+flow", f"text+saliency"]

    for target_type in ["Gaze", "PCA"]:
        sub = summary[summary["TargetType"] == target_type]
        x_order = sorted(sub["x_label"].unique())
        g = sns.FacetGrid(
            sub,
            col="x_label",
            col_wrap=2,
            sharey=True,
            height=4,
            aspect=1.2
        )
        g.map_dataframe(
            sns.barplot,
            x="feature_name",
            y="mean_abs_t",
            hue="Model",
            hue_order=model_order,
            order=feature_order,
            dodge=True,
            palette="Set2"
        )
        g.set_axis_labels("Model Feature", "Mean |t| value (feature importance)")
        g.set_titles(col_template="{col_name}")
        for ax in g.axes.flatten():
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        g.add_legend(title="Model")
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_{llm_model}_{target_type.lower()}.png")
        plt.close()

plot_attention_feature_importances(
    "outputs/ols_unified_performance.csv",
    "outputs/ols_attention_feature_importances",
    "llama"
)
plot_attention_feature_importances(
    "outputs/ols_unified_performance.csv",
    "outputs/ols_attention_feature_importances",
    "bert"
)
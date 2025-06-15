import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.analysis.correlation import FEATURES

def get_model_type(row):
    if row["predictors"] == "text_only":
        return "TextOnly"
    elif row["attn_method"] in ["raw", "flow", "saliency"]:
        return "Attention"
    return "Other"

def preprocess_perf_df(perf_path):
    df = pd.read_csv(perf_path)
    group_cols = ["task", "llm_model", "attn_method", "predictors", "dependent"]
    agg_cols = {"rsquared": "mean", "rsquared_adj": "mean", "rmse": "mean"}
    df = df.groupby(group_cols, as_index=False).agg(agg_cols)
    df["ModelType"] = df.apply(get_model_type, axis=1)
    df["TargetType"] = df["dependent"].apply(lambda x: "Gaze" if x in FEATURES else ("PCA" if x == "pca" else "Other"))
    df["attention_method"] = df["attn_method"]
    return df[df["ModelType"].isin(["Attention", "TextOnly"])]

def plot_metric(plot_df, metric, filename):
    plot_df = plot_df.copy()
    plot_df["bar_type"] = plot_df.apply(
        lambda row: "text only" if row["ModelType"] == "TextOnly" else f"text+{row['attention_method']}", axis=1
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
    ax.set_title("Ordinary Least Squares Models")
    ax.set_ylim(top=0.5)
    ax.legend(title="Models")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_attention_feature_importances(perf_path, filename_prefix, llm_model):
    df = pd.read_csv(perf_path)
    df = df[
        (df["attn_method"].isin(["raw", "flow", "saliency"])) &
        (df["feature_name"] != "const") &
        (~df["feature_name"].isin(["attention_layer_1", "attention_layer_31"])) &
        (df["llm_model"] == llm_model)
    ]
    df["TargetType"] = df["dependent"].apply(lambda x: "Gaze" if x in FEATURES else ("PCA" if x == "pca" else "Other"))
    df["Model"] = df["attn_method"].apply(lambda x: f"text+{x}")
    df["feature_name"] = df["feature_name"].replace({"attention_layer_0": llm_model})

    # Only keep Gaze and PCA targets
    df = df[df["TargetType"].isin(["Gaze", "PCA"])]
    df["abs_t"] = df["t"].abs()

    feature_order = [llm_model] + sorted([f for f in df["feature_name"].unique() if f != llm_model])
    model_order = [f"text+raw", f"text+flow", f"text+saliency"]

    g = sns.FacetGrid(
        df, col="task", sharey=True, height=5, aspect=1.2
    )
    g.map_dataframe(
        sns.barplot,
        x="feature_name",
        y="abs_t",
        hue="Model",
        hue_order=model_order,
        order=feature_order,
        dodge=True,
        palette="Set2",
        errorbar="sd"
    )
    g.set_axis_labels("Model Feature", "Mean |t| value (feature importance, avg. Gaze & PCA)")
    g.set_titles(col_template="Task: {col_name}")
    for ax in g.axes.flatten():
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    g.add_legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{llm_model}.png")
    plt.close()

# --- MAIN EXECUTION ---
perf_path = "outputs/ols_unified_performance.csv"
plot_df = preprocess_perf_df(perf_path)

# Plot metrics for llama and bert
plot_metric(plot_df[plot_df["llm_model"] != "bert"], "rsquared", "outputs/ols_barplot_r2_llama.png")
bert_df = plot_df[plot_df["llm_model"] == "bert"]
if not bert_df.empty:
    plot_metric(bert_df, "rsquared", "outputs/ols_barplot_r2_bert.png")

# Plot averaged feature importances for both models
for model in ["llama", "bert"]:
    plot_attention_feature_importances(perf_path, "outputs/ols_attention_feature_importances", model)
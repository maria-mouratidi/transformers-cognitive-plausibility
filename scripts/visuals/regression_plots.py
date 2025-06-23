import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scripts.analysis.correlation import FEATURES

def set_academic_rcparams():
    plt.rcParams.update({
        'font.size': 38,
        'axes.labelsize': 36,
        'axes.titlesize': 42,
        'xtick.labelsize': 40,
        'ytick.labelsize': 44,
        'legend.fontsize': 34,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif']
    })

MODEL_TITLE = {"bert": "BERT", "llama": "Llama"}

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

CUSTOM_PALETTE = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00']

def plot_metric(plot_df, metric, filename):
    set_academic_rcparams()
    plot_df = plot_df.copy()
    plot_df["bar_type"] = plot_df.apply(
        lambda row: "text only" if row["ModelType"] == "TextOnly" else f"text+{row['attention_method']}", axis=1
    )
    plot_df = plot_df[
        ((plot_df["ModelType"] == "TextOnly") & (plot_df["bar_type"] == "text only")) |
        ((plot_df["ModelType"] == "Attention") & (plot_df["bar_type"].isin(["text+raw", "text+flow", "text+saliency"])))
    ]
    plot_df["x_label"] = plot_df["TargetType"] + "\n" + plot_df["task"].astype(str)
    target_order = ["Gaze", "PCA"]
    task_order = sorted(plot_df["task"].unique())
    x_order = [f"{target}\n{task}" for target in target_order for task in task_order]
    hue_order = ["text only", "text+raw", "text+flow", "text+saliency"]

    llm_models = ["bert", "llama"]
    fig, axes = plt.subplots(1, 2, figsize=(32, 12), sharey=True)
    for idx, llm_model in enumerate(llm_models):
        ax = axes[idx]
        subdf = plot_df[plot_df["llm_model"] == llm_model]
        sns.barplot(
            data=subdf, x="x_label", y=metric, hue="bar_type",
            order=x_order, hue_order=hue_order, errorbar="sd", palette=CUSTOM_PALETTE, ax=ax
        )
        ax.set_xlabel("")
        ax.set_ylabel(r"$R^2$ adjusted" if metric == "rsquared_adj" else f"{metric.capitalize()}")
        ax.set_ylim(top=0.6)
        ax.set_title(MODEL_TITLE[llm_model], fontweight='bold')
        if idx == 1:
            ax.legend(title="Model", loc='upper right')
        else:
            ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

def plot_attention_feature_importances(perf_path, filename):
    set_academic_rcparams()
    df = pd.read_csv(perf_path)
    df = df[
        (df["attn_method"].isin(["raw", "flow", "saliency"])) &
        (df["feature_name"] != "const") &
        (~df["feature_name"].isin(["attention_layer_31"]))
    ]
    df["TargetType"] = df["dependent"].apply(lambda x: "Gaze" if x in FEATURES else ("PCA" if x == "pca" else "Other"))
    df["Model"] = df["attn_method"].apply(lambda x: f"text+{x}")
    df["feature_name"] = df["feature_name"].replace({"attention_layer_0": "attention", "attention_layer_1": "attention"})
    df = df[df["TargetType"].isin(["Gaze", "PCA"])]
    df["abs_t"] = df["t"].abs()

    feature_order = sorted([f for f in df["feature_name"].unique() if f not in ["bert", "llama"]])
    model_order = [f"text+raw", f"text+flow", f"text+saliency"]
    task_order = sorted(df["task"].unique())

    n_rows = 2
    n_cols = len(task_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.2 * n_cols, 11 * n_rows), sharey='row')

    for row_idx, model_name in enumerate(["bert", "llama"]):
        for col_idx, task in enumerate(task_order):
            ax = axes[row_idx, col_idx] if n_cols > 1 else axes[row_idx]
            subdf = df[(df["llm_model"] == model_name) & (df["task"] == task)]
            sns.barplot(
                data=subdf,
                x="feature_name",
                y="abs_t",
                hue="Model",
                hue_order=model_order,
                order=feature_order,
                dodge=True,
                palette=CUSTOM_PALETTE[1:],
                errorbar="sd",
                ax=ax
            )
            if row_idx == 0:
                ax.set_title(f"Task {task[-1]}", fontweight='bold', pad=10)
            if col_idx == 0:
                ax.set_ylabel(f"|t| ({MODEL_TITLE[model_name]})")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            #ax.set_ylim(bottom=0)
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(title="Model", loc='upper right')
            else:
                ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", dpi=600, bbox_inches='tight', format='pdf')
    plt.close()

# --- MAIN EXECUTION ---
perf_path = "outputs/ols_unified_performance.csv"
plot_df = preprocess_perf_df(perf_path)

plot_metric(plot_df, "rsquared_adj", "outputs/ols_r2.pdf")
plot_attention_feature_importances(perf_path, "outputs/ols_feature_importances")
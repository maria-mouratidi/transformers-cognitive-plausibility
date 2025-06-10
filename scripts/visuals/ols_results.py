import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

all_perf = []
for perf_path in glob.glob("outputs/**/**/**/ols/model_performance.csv", recursive=True):
    df = pd.read_csv(perf_path)
    df["source_path"] = perf_path
    all_perf.append(df)

if all_perf:
    perf_df = pd.concat(all_perf, ignore_index=True)
    perf_df.to_csv("outputs/unified_model_performance.csv", index=False)
    print("Unified performance CSV created at outputs/unified_model_performance.csv")

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
plot_df = perf_df[perf_df["ModelType"].isin(["Attention", "TextOnly"])]

def plot_metric(plot_df, metric, filename):
    target_order = ["Gaze", "PCA"]
    task_order = sorted(plot_df["task"].unique())
    x_order = [f"{target} - {task}" for target in target_order for task in task_order]

    plot_df["x_label"] = plot_df["TargetType"] + " - " + plot_df["task"]

    g = sns.catplot(
        data=plot_df,
        kind="bar",
        x="x_label",
        y=metric,
        hue="ModelType",
        col="attention_method",
        ci="sd",
        height=5,
        aspect=1.3,
        dodge=True,
        order=x_order
    )

    g.set_axis_labels("Target & Task", f"Test {metric.upper()}")
    g.set_titles("Attention Method: {col_name}")
    g._legend.set_title("Model Type")

    # Wilcoxon annotation for R² and RMSE
    from scipy.stats import wilcoxon
    for ax, (attn_method, attn_df) in zip(g.axes.flat, plot_df.groupby("attention_method")):
        for xtick, (x_label, group) in enumerate(attn_df.groupby("x_label")):
            attn_vals = group[group["ModelType"] == "Attention"][metric]
            text_vals = group[group["ModelType"] == "TextOnly"][metric]
            if not attn_vals.empty and not text_vals.empty:
                try:
                    stat, p = wilcoxon(attn_vals, text_vals)
                    if p < 0.05:
                        max_height = max(attn_vals.mean(), text_vals.mean())
                        ax.annotate(
                            "*",
                            xy=(xtick, max_height),
                            xytext=(0, 8),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            color="red",
                            fontsize=18
                        )
                except Exception:
                    pass

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- PLOT FOR ALL MODELS EXCEPT BERT ---
not_bert_df = plot_df[~plot_df["source_path"].str.contains("bert", case=False, na=False)]
plot_metric(not_bert_df, "rsquared", "outputs/ols_barplot_r2.png")
plot_metric(not_bert_df, "rmse", "outputs/ols_barplot_rmse.png")

# --- PLOT FOR BERT ONLY ---
bert_df = plot_df[plot_df["source_path"].str.contains("bert", case=False, na=False)]
if not bert_df.empty:
    plot_metric(bert_df, "rsquared", "outputs/ols_barplot_r2_bert.png")
    plot_metric(bert_df, "rmse", "outputs/ols_barplot_rmse_bert.png")
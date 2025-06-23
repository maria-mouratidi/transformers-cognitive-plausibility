import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scripts.visuals.regression_plots import CUSTOM_PALETTE, MODEL_TITLES, TASK_TITLES, corr_plt_params

def plot_other_corr(combined_df, save_dir=None, log_file=None):
    """
    Plot a single barplot showing average Spearman r across features
    for each model-task-method combination (flow & saliency only).
    Also logs the mean and std values.
    """
    plt.rcParams.update(corr_plt_params)

    # Filter only 'flow' and 'saliency' methods
    bar_df = combined_df[combined_df['attn_method'].isin(['flow', 'saliency'])].copy()

    colors = {
        ('bert', 'task2'): '#1f77b4',      # blue
        ('bert', 'task3'): '#aec7e8',      # light blue
        ('llama', 'task2'): '#ff7f0e',     # orange
        ('llama', 'task3'): '#ffbb78',     # light orange
    }
    method_hatches = {'flow': '', 'saliency': '/'}

    # Prepare bar order
    bar_order = []
    for model in ['bert', 'llama']:
        for task in ['task2', 'task3']:
            bar_order.append((model, task))

    x = np.arange(len(bar_order))
    width = 0.35

    plt.figure(figsize=(8, 5))

    all_bars = []
    log_lines = []
    for i, method in enumerate(['flow', 'saliency']):
        bar_positions = x + (i - 0.5) * width  # offset left/right
        means, stds, bar_colors, hatches = [], [], [], []

        for model, task in bar_order:
            sub_df = bar_df[
                (bar_df['llm_model'].str.lower() == model) &
                (bar_df['task'] == task) &
                (bar_df['attn_method'] == method)
            ]
            grouped = sub_df.groupby('feature')['spearman_r'].mean()
            mean_val = grouped.mean()
            std_val = grouped.std()
            means.append(mean_val)
            stds.append(std_val)
            bar_colors.append(colors[(model, task)])
            hatches.append(method_hatches[method])
            log_lines.append(
                f"{MODEL_TITLES[model]} {TASK_TITLES[task]} {method}: mean={mean_val:.3f}, std={std_val:.3f}"
            )

        bars = plt.bar(
            bar_positions, means, width=width,
            yerr=stds,
            color=bar_colors,
            hatch=method_hatches[method],
            label=method
        )
        all_bars.extend(bars)

    # X-tick labels
    x_labels = [f"{MODEL_TITLES[model]}\n{TASK_TITLES[task]}" for model, task in bar_order]
    plt.xticks(x, x_labels)
    plt.ylabel("Spearman r")
    plt.ylim(-0.2, 0.8)
    plt.grid(False)  # Remove grid lines
    plt.legend(title="Method")
    plt.tight_layout()

    # Output log
    log_text = "\n".join(log_lines)
    if log_file:
        with open(log_file, "w") as f:
            f.write(log_text)
    else:
        print("Correlation summary for barplot:")
        print(log_text)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "correlations_nonraw.pdf"),
                    dpi=600, bbox_inches='tight', format='pdf')
        plt.close()

def plot_raw_corr(combined_df, save_dir=None, log_file=None):
    """
    Lineplots of mean correlation across features for raw attention
    Each model-task pair has its own color, std as shadow.
    Also logs the mean and std values per layer.
    """
    plt.rcParams.update(corr_plt_params)

    pair_colors = {
        ('bert', 'task2'): '#1f77b4',      # blue
        ('bert', 'task3'): '#aec7e8',      # light blue
        ('llama', 'task2'): '#ff7f0e',     # orange
        ('llama', 'task3'): '#ffbb78',     # light orange
    }
    model_labels = {'llama': 'Llama', 'bert': 'BERT'}
    task_labels = {'task2': 'Task 2', 'task3': 'Task 3'}

    plt.figure(figsize=(8, 5))
    log_lines = []
    for model in ['bert', 'llama']:
        for task in ['task2', 'task3']:
            task_df = combined_df[
                (combined_df['attn_method'] == 'raw') &
                (combined_df['task'] == task) &
                (combined_df['llm_model'].str.lower() == model)
            ]
            grouped = task_df.groupby('layer')['spearman_r']
            mean = grouped.mean()
            std = grouped.std()
            layers = mean.index
            label = f"{model_labels[model]} {task_labels[task]}"
            color = pair_colors[(model, task)]
            plt.plot(
                layers, mean,
                label=label,
                color=color,
                marker='o',
                linewidth=2,
                alpha=0.95,
                markerfacecolor=color,
                markeredgecolor=color
            )
            plt.fill_between(
                layers, mean - std, mean + std,
                color=color,
                alpha=0.15
            )
            for l in layers:
                log_lines.append(
                    f"{label} Layer {l}: mean={mean[l]:.3f}, std={std[l]:.3f}"
                )
    plt.xlabel("Layer")
    plt.ylabel("Spearman r")
    plt.ylim(-0.2, 0.8)
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
    plt.tight_layout()

    # Output log
    log_text = "\n".join(log_lines)
    if log_file:
        with open(log_file, "w") as f:
            f.write(log_text)
    else:
        print("Correlation summary for raw attention lineplot:")
        print(log_text)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"correlations_raw.pdf"),
                    dpi=600, bbox_inches='tight', format='pdf')
        plt.close()

def plot_text_attn_corr(corr_df, save_path=None):
    plt.rcParams.update(corr_plt_params)
    method_order = ["raw", "flow", "saliency"]

    # Prepare unique model-task pairs
    pairs = [(m, t) for m in ['bert', 'llama'] for t in ['task2', 'task3']]
    nrows, ncols = 2, 2  # 2 subplots in each row

    fig, axes = plt.subplots(nrows, ncols, figsize=(32.4, 22), sharey=True)
    axes = axes.flatten() 
    for idx, (model, task) in enumerate(pairs):
        ax = axes[idx]
        sub_df = corr_df[
            (corr_df['llm_model'] == model) &
            (corr_df['task'] == task)
        ].copy()
        sub_df["Text Feature"] = sub_df["text_feature"].str.capitalize()
        sns.barplot(
            data=sub_df,
            x="Text Feature",
            y="spearman_r",
            hue="attn_method",
            hue_order=method_order,
            palette=CUSTOM_PALETTE[1:],
            errorbar="sd",
            capsize=0.1,
            ci="sd",
            estimator=np.mean,
            ax=ax
        )
        ax.set_title(f"{model_labels[model]} - {task_labels[task]}", fontweight='bold')
        ax.set_xlabel("")
 
        ax.set_ylabel(f"Spearman r ({model_labels[model]})", fontsize=16)
        #ax.set_ylim(-0.4, 0.7)
        if idx != ncols - 1:
            ax.get_legend().remove()
        else:
            ax.legend(title="Attention Method")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format='pdf')
        plt.close()
    else:
        plt.show()
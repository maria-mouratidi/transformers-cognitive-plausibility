import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
palette = "RdBu_r"  #diverging color palette

def set_academic_rcparams():
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif']
    })

def plot_other_corr(combined_df, save_dir=None, log_file=None):
    """
    Plot a single barplot showing average Spearman r across features
    for each model-task-method combination (flow & saliency only).
    Also logs the mean and std values.
    """
    set_academic_rcparams()

    # Filter only 'flow' and 'saliency' methods
    bar_df = combined_df[combined_df['attn_method'].isin(['flow', 'saliency'])].copy()

    model_labels = {'llama': 'Llama', 'bert': 'BERT'}
    task_labels = {'task2': 'Task 2', 'task3': 'Task 3'}
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
                f"{model_labels[model]} {task_labels[task]} {method}: mean={mean_val:.3f}, std={std_val:.3f}"
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
    x_labels = [f"{model_labels[model]}\n{task_labels[task]}" for model, task in bar_order]
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
    set_academic_rcparams()

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
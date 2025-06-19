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

def plot_corr(combined_df, model_name, save_dir=None):
    """
    Plot a heatmap combining feature and PCA component correlations.
    Optimized for academic paper publication.
    """
    set_academic_rcparams()
    
    def reorder_columns_pc1_first(df):
        cols = df.columns.tolist()
        if 'PC1' in cols:
            cols.remove('PC1')
            cols = ['PC1'] + cols
            return df[cols]
        return df

    # Exclude 'raw' attention method from combined heatmap
    flow_saliency_results_df = combined_df[combined_df['attn_method'] != 'raw'].copy()
    flow_saliency_results_df.loc[:, 'attn_layer'] = flow_saliency_results_df['attn_method']
    pivot = flow_saliency_results_df.pivot(index=['task', 'attn_method'], columns='feature', values=f'spearman_r')
    pvals = flow_saliency_results_df.pivot(index=['task', 'attn_method'], columns='feature', values=f'spearman_p_value')
    mask = pvals >= 0.05
    pivot = reorder_columns_pc1_first(pivot)
    pvals = reorder_columns_pc1_first(pvals)

    plt.figure(figsize=(8, 4))
    sns.heatmap(pivot, mask=mask, annot=True, fmt=".2f", center=0, 
                cmap=palette, cbar_kws={'label': f'Spearman r', 'shrink': 0.7, 'aspect': 20, 'pad': 0.02},
                linewidths=0.5, linecolor='white')
    plt.xlabel('')
    plt.ylabel('')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"combined_corrs_{model_name}.pdf"), 
                   dpi=600, bbox_inches='tight', format='pdf')
        plt.close()


def plot_corr_lineplots(combined_df, save_dir=None):
    """
    Plot lineplots of raw correlations per task, one line per feature,
    solid for LLaMA, dashed for BERT, both models in the same plot.
    """
    set_academic_rcparams()
    features = [col for col in combined_df['feature'].unique() if col != 'type']
    line_styles = {'llama': '-', 'bert': '--'}
    palette_dict = dict(zip(features, sns.color_palette("tab10", n_colors=len(features))))
    model_labels = {'llama': 'LLaMA', 'bert': 'BERT'}

    for task in ['task2', 'task3']:
        plt.figure(figsize=(14, 14))
        task_df = combined_df[(combined_df['attn_method'] == 'raw') & (combined_df['task'] == task)]
        for feature in features:
            for model in ['llama', 'bert']:
                model_df = task_df[(task_df['feature'] == feature) & (task_df['llm_model'].str.lower() == model)]
                model_df = model_df.sort_values('layer')
                y = model_df['spearman_r'].where(model_df['spearman_p_value'] < 0.05, np.nan)
                plt.plot(
                    model_df['layer'], y,
                    label=f"{feature} ({model_labels[model]})",
                    linestyle=line_styles[model],
                    color=palette_dict[feature],
                    marker='o',
                    linewidth=2,
                    alpha=0.95
                )
        plt.title(f"Raw Attention Correlations: {task.title()}", fontweight='bold')
        plt.xlabel("Layer")
        plt.ylabel("Spearman r")
        plt.ylim(-0.2, 0.75)
        # Only show unique legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=13, ncol=2)
        plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"corr_lineplot_{task}.pdf"),
                        dpi=600, bbox_inches='tight', format='pdf')
            plt.close()

def plot_corr_lineplots_mean_shadow(combined_df, save_dir=None):
    """
    Plot mean correlation across features per LLM model, with std as shadow.
    Solid line for LLaMA, dashed for BERT.
    """
    set_academic_rcparams()
    tasks = ['task2', 'task3']
    models = ['llama', 'bert']
    line_styles = {'llama': '-', 'bert': '--'}
    colors = {'llama': '#1f77b4', 'bert': '#ff7f0e'}
    model_labels = {'llama': 'LLaMA', 'bert': 'BERT'}

    for task in tasks:
        plt.figure(figsize=(8, 5))
        task_df = combined_df[(combined_df['attn_method'] == 'raw') & (combined_df['task'] == task)]
        for model in models:
            model_df = task_df[task_df['model'].str.lower() == model]
            # Group by layer, aggregate mean and std across features
            grouped = model_df.groupby('layer')['spearman_r']
            mean = grouped.mean()
            std = grouped.std()
            # Only keep layers present in both mean and std
            layers = mean.index
            plt.plot(
                layers, mean, 
                label=model_labels[model],
                linestyle=line_styles[model],
                color=colors[model],
                marker='o',
                linewidth=2,
                alpha=0.95
            )
            plt.fill_between(
                layers, mean - std, mean + std,
                color=colors[model],
                alpha=0.2
            )
        plt.title(f"Raw Attention Correlations (Mean ± SD): {task.title()}", fontweight='bold')
        plt.xlabel("Layer")
        plt.ylabel("Spearman r")
        plt.ylim(-0.2, 1.0)
        plt.legend(fontsize=14)
        plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"corr_lineplot_mean_shadow_{task}.pdf"),
                        dpi=600, bbox_inches='tight', format='pdf')
            plt.close()

def plot_gaze_intercorr(human_df, pca_df, features, save_dir=None):
    """
    Plot heatmap of inter-correlations among gaze features and PCA components.
    """
    set_academic_rcparams()
    # Replace feature names for display
    combined = pd.concat([human_df[features], pca_df], axis=1)
    combined = combined.rename(columns={'meanPupilSize': 'mPS', 'nFixations': 'F'})
    corr_matrix = combined.corr(method='spearman')
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, center=0, cmap=palette)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f'intercorrs.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Saved inter-correlation heatmap to {save_path}")



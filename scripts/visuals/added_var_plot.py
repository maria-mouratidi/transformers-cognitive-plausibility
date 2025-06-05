import os
import matplotlib.pyplot as plt
from statsmodels.api import add_constant
from statsmodels.graphics.regressionplots import plot_partregress
import pandas as pd
import numpy as np

def partial_reg_plot(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    attention_feature: str,
    save_dir: str,
):
    """
    Generate and save partial regression plots of a specified attention feature
    against all PCA components.

    Parameters:
        X_scaled (np.ndarray): Scaled design matrix with attention + text features.
        y_pca (np.ndarray): PCA-transformed gaze data (n_samples, n_components).
        feature_names (list): List of feature names in X_scaled (must match order).
        attention_feature (str): Name of the attention feature to isolate.
        output_dir (str): Base output directory (e.g., "outputs").
        task (str): Task name used in subdirectory (e.g., "task3").
        attn_method (str): Attention method used in subdirectory (e.g., "raw").
    """
    
    n_components = y.shape[1]

    plot_dir = os.path.join(save_dir, "partreg_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for comp in range(n_components):
        y_comp = y.iloc[:, comp]
        response_name = f"PC{comp}"

        # Create DataFrame for modeling
        X_df = pd.DataFrame(X, columns=feature_names, index=y.index)
        X_df = add_constant(X_df)
        df_model = X_df.copy()
        df_model[response_name] = y_comp

        # Plot and save
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        plot_partregress(
            endog=response_name,
            exog_i=attention_feature,
            exog_others=[f for f in feature_names if f != attention_feature],
            data=df_model,
            obs_labels=False,
            ax=ax
        )

        ax.set_title(f'Partial Regression: {attention_feature} vs PCA Component {comp}')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'partreg_pca_comp{comp}_{attention_feature}.png'))
        plt.close()

    print(f"Partial regression plots saved to: {plot_dir}")

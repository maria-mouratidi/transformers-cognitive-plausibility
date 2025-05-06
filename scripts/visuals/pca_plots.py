from matplotlib import pyplot as plt
import seaborn as sns
import os
import pandas as pd

def plot_explained_variance(cumulative_variance, save_dir=None):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance vs Number of Components")
    plt.legend()
    plt.grid()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/explained_variance.png")
    plt.show()

def plot_pca_loadings(pca, features, save_dir=None):
    loadings = pd.DataFrame(pca.components_, columns=features, index=[f'PC{i+1}' for i in range(pca.n_components_)])
    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
    plt.title("PCA Component Loadings")
    plt.xlabel("Original Features")
    plt.ylabel("Principal Components")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/pca_loadings.png")
    plt.show()

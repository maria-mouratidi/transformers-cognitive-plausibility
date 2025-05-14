from sklearn.metrics import mean_squared_error
import numpy as np
import copy

def compute_permutation_importance(
    model, X: np.ndarray, y: np.ndarray, feature_names: list, 
    metric_func=mean_squared_error, n_repeats=10, random_state=42
):
    """
    Compute permutation importance for each feature using the specified regression model.

    Parameters:
        model: Trained regression model (e.g., LinearRegression).
        X (np.ndarray): Input features (n_samples, n_features).
        y (np.ndarray): Target values (can be multi-output).
        feature_names (list): Feature names in X.
        metric_func: Scoring function (default is mean_squared_error).
        n_repeats (int): Number of times to shuffle each feature.
        random_state (int): Reproducibility.

    Returns:
        List of (feature_name, mean_importance, std_dev) tuples.
    """
    rng = np.random.RandomState(random_state)
    baseline_score = metric_func(y, model.predict(X))
    
    importances = []

    for i, name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, i] = rng.permutation(X[:, i])
            score = metric_func(y, model.predict(X_permuted))
            scores.append(score - baseline_score)  # positive = performance worsened

        importances.append((name, np.mean(scores), np.std(scores)))

    importances.sort(key=lambda x: x[1], reverse=True)  # higher importance = more impact
    return importances

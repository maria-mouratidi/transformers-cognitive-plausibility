from scipy.stats import shapiro

def shapiro_test(human_df, model_values, features, layer_idx):
    print("Shapiro-Wilk Test Results:")
    for feature_name in features:
        stat_human, p_human = shapiro(human_df[feature_name])
        stat_model, p_model = shapiro(model_values)
        print(f"Feature: {feature_name}")
        print(f"  Eye-gaze - Stat: {stat_human:.4f}, p-value: {p_human:.4f}")
        print(f"  Model (Layer {layer_idx}) - Stat: {stat_model:.4f}, p-value: {p_model:.4f}\n")

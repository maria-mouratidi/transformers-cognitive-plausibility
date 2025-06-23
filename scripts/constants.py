FEATURES = ['nFixations', 'meanPupilSize', 'GD', 'TRT', 'FFD'] # Gaze features to analyze
ALL_FEATURES = FEATURES + ['SFD', 'GPT']  # Full feature list
MODEL_TITLES = {"bert": "BERT", "llama": "Llama"} # Styled titles for models
TASK_TITLES = {'task2': 'Task 2', 'task3': 'Task 3'}
CUSTOM_PALETTE = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00'] # Custom colors for the plots

corr_plt_params = {
        'font.size': 22,
        'axes.labelsize': 28,
        'axes.titlesize': 28,
        'xtick.labelsize': 24,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif']
    }

ols_plt_params = {
        'font.size': 36,
        'axes.labelsize': 40,
        'axes.titlesize': 42,
        'xtick.labelsize': 40,
        'ytick.labelsize': 44,
        'legend.fontsize': 34,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif']
    }
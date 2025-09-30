# Transformers Cognitive Plausibility: Eye-Tracking and Neural Attention Analysis

This repository contains code for analyzing the cognitive plausibility of transformer models by comparing their attention patterns with human eye-tracking data during reading comprehension tasks.

## Overview

The experiment investigates whether transformer attention patterns correlate with human cognitive processing during reading, using two tasks:
- **Task 2**: Natural reading Wikipedia sentences
- **Task 3**: Reading similar sentences wile looking for specifci semantic relation (e.g., job title)

For more information about the eye-tracking data see the [ZuCo paper](https://doi.org/10.1038/sdata.2018.291)

We probe two model types:
- **BERT** (encoder-only): `bert-base-uncased`
- **LLaMA** (decoder-only): `meta-llama/Llama-3.1-8b`

Using three probing techniques:
- **Raw**: Direct attention weights
- **Flow**: Attention flow between layers
- **Saliency**: Gradient-based saliency maps

## Understanding the repository

### 1. Data Processing Scripts

The experiment processes ZuCo eye-tracking data and sentence datasets through several scripts:

#### Core Data Processing (`scripts/data_processing/`)
- **`read_data.py`**: Reads raw ZuCo eye-tracking data with specific scaling and na filling parameters
- **`process_data.py`**: Processes participant data for both Task 2 and Task 3, extracting word-level reading measures
- **`get_relations.py`**: Restructure data from task 3 to contain their relations for ease
- **`utils_ZuCo.py`**: Utility functions for handling the original ZuCo dataset

#### Eye-tracking data and reading materials
- `data/task2/processed/all_participants.csv` - Eye-tracking data for Task 2 (natural reading)
- `data/task3/processed/all_participants.csv` - Eye-tracking data for Task 3 (relation extraction)
- `materials/sentences_task2.json` - Task 2 Sentences in List[List[str]] format
- `materials/sentences_task3.json` - Task 3 Sentences in List[List[str]] format
- `materials/relations_task_specific.csv` - Original data with relations

### 2. Model Probing Scripts

⚠️ **Note**: Pre-computed probing results are available in `outputs/`. Probing from scratch requires significant time and computational resources.

#### Probing Techniques (`scripts/probing/`)
The repository implements three attention analysis techniques for both BERT (encoder) and LLaMA (decoder):

##### Raw Attention
- **`raw_encoder.py`**: Extract raw attention weights from BERT
- **`raw_decoder.py`**: Extract raw attention weights from LLaMA

##### Flow Attention  
- **`flow_encoder.py`**: Compute attention flow between BERT layers
- **`flow_decoder.py`**: Compute attention flow between LLaMA layers

##### Saliency Maps
- **`saliency_encoder.py`**: Generate gradient-based saliency for BERT
- **`saliency_decoder.py`**: Generate gradient-based saliency for LLaMA

##### Support Files
- **`load_model.py`**: Model loading utilities for both BERT and LLaMA

#### Pre-computed Probing Results
All results are stored in `outputs/`:
- `outputs/raw/task2/bert/` - BERT raw attention for Task 2
- `outputs/raw/task2/llama/` - LLaMA raw attention for Task 2  
- `outputs/flow/task3/bert/` - BERT flow attention for Task 3
- `outputs/saliency/task3/llama/` - LLaMA saliency for Task 3
- (Similar structure for all reading task/model/ probing technique combinations)

Each directory contains:
- `attention_processed.pt` or `attention_flow_processed.pt` - Processed attention data
- `combined_corr_spearman.png` - Correlation visualizations
- `surprisals.json` - Model surprisal values for each word (for raw technique)

### 3. Analysis Scripts

#### Core Analysis (`scripts/analysis/`)
- **`load_attention.py`**: Loads and aligns human eye-tracking data with model attention data
- **`correlation.py`**: Computes Spearman correlations between human and model attention
- **`text_features.py`**: Extracts linguistic features from sentences (length, frequency, role etc.)
- **`regression.py`**: Runs various regression models with or without attention as a feature to predict human reading patterns
- **`perm_feat_imp.py`**: Computes permutation-based feature importance


### 4. Visualization Scripts

#### Plotting Scripts (`scripts/visuals/`)
- **`corr_plots.py`**: Generate correlation included in the paper
- **`regression_plots.py`**: Visualize regression results and model performance
- **`added_var_plot.py`**: Create added variable plots for regression analysis


## Repository Structure

```
├── scripts/                         # All analysis and processing code
│   ├── constants.py                 # Shared constants and configuration
│   ├── analysis/                    # Analysis modules
│   │   ├── correlation.py           # Spearman correlation analysis
│   │   ├── load_attention.py        # Data loading and alignment
│   │   ├── perm_feat_imp.py        # Permutation feature importance
│   │   ├── regression.py            # Regression models
│   │   └── text_features.py         # Feature extraction from sentences for regression modelss
│   ├── data_processing/             # Data processing modules  
│   │   ├── get_relations.py         # Relation extraction from processed data
│   │   ├── process_data.py          # Eye-tracking data processing
│   │   ├── read_data.py             # Raw data reading with specific settings
│   │   └── utils_ZuCo.py           # ZuCo dataset utilities
│   ├── probing/                     # Model attention probing
│   │   ├── flow_encoder.py          # BERT flow attention extraction
│   │   ├── flow_decoder.py          # LLaMA flow attention extraction  
│   │   ├── raw_encoder.py           # BERT raw attention extraction
│   │   ├── raw_decoder.py           # LLaMA raw attention extraction
│   │   ├── saliency_encoder.py      # BERT gradient saliency
│   │   ├── saliency_decoder.py      # LLaMA gradient saliency
│   │   └── load_model.py            # Model loading utilities
│   └── visuals/                     # Visualization and plotting
│       ├── added_var_plot.py        # Added variable plots for regression
│       ├── corr_plots.py            # Correlation heatmaps and plots
│       └── regression_plots.py      # Regression performance visualization
├── data/                            # Eye-tracking data (processed)
│   ├── task2/processed/
│   │   ├── all_participants.csv     # All participant data for Task 2
│   │   └── processed_participants.csv # Averaged and normalized participant data
│   └── task3/processed/
│       ├── all_participants.csv     # All participant data for Task 3  
│       └── processed_participants.csv # Averaged and normalized participant data
├── materials/                       # Experimental materials and features
│   ├── prompts.py                   # Task prompts and instructions
│   ├── relations_task_specific.csv  # Semantic relation mappings
│   ├── sentences_task2.json         # Relation classification sentences
│   ├── sentences_task3_unlabeled.json # Unlabeled Task 3 sentences
│   ├── sentences_task3.json         # Labeled Task 3 sentences  
│   ├── text_features_task2_bert.csv # Extracted sentence features for Task 2 (incl. BERT surprisal)
│   ├── text_features_task2_llama.csv # Extracted sentence features for Task 2 (incl. LLaMa surprisal)
│   ├── text_features_task3_bert.csv # Extracted sentence features for Task 3 (incl. BERT surprisal)
│   └── text_features_task3_llama.csv # Extracted sentence features for Task 3 (incl. LLaMa surprisal)
└── outputs/                         # Pre-computed results
    ├── ols_unified_performance.csv  # Regression model performance comprehensive results
    ├── flow/                        # Flow attention 
    │   ├── task2/{bert,llama}/      # Task 2 flow
    │   └── task3/{bert,llama}/      # Task 3 flow
    ├── raw/                         # Raw attention
    │   ├── task2/{bert,llama}/      # Task 2 raw
    │   └── task3/{bert,llama}/      # Task 3 raw
    └── saliency/                    # Saliency method
        ├── none/bert/               # Baseline saliency for testing
        ├── task2/{bert,llama}/      # Task 2 saliency 
        └── task3/{bert,llama}/      # Task 3 saliency
```

### Tips

1. **Use Pre-computed Results**: Skip probing by using files in `outputs/`
2. **GPU Acceleration**: Use CUDA for faster probing (especially flow attention)
3. Original processed data accessible [here, under answers](https://osf.io/q3zws/files/osfstorage#)

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{
}
```

## Contact

- First name: Maria
- Last name: Mouratidi
- Email: mouratidi.m@gmail.com
- Personal page: https://maria-mouratidi.github.io/

# Workflow to seamlessly use ML on tabular data at certain steps of computational workflow mainly for ranking features

- Motivated by applying ML on single-cell data where features would would be genes and samples would be single-cells
 
# Use cases

#### Computational biology

- Predicting control vs. phenotype and identifying important features
  + This effectively selects features too reducing dimensionality and also identifies most important features with biological relevance (results can be compared with conventional methods for analysing data)
  + Rather than applying on original data with each gene or protein as features, can apply on PCA dimensions to identify important PC components for predicting phenotype and then examine key features correlated with those important PC components

- Integrating multi-omic or modal data to predict phenotypes
  + When different modalities of data/measurements are available e.g. RNA and proteins, this pipeline can be used to train and select features from each modality and then do another training after combining each modalities with reduced features
  
# Method (current implementation)

- ML framework: LightGBM
  + Allows to easily switch between predicting continuous or categorical labels
  + Faster than XGBoost

- Automatic hyperparameter optimisation with optuna
  + Reduce spending time on hyperparameter optimisation
  + Set sensible ranges for optuna as defaults so user can focus on getting feature ranking
  + Optuna does multiple optimisation trials using different hyperparameters and identifies best trial

- Cross validation
  + To identify robust model
  + Increase robustness with feature importance metrics from multiple splits of data

# Outputs

The following are generated per optimisation trial

- Plots to assess model e.g. did it overfit, underfit, need more iteration
  + Assess whether you can trust the feature importance outputs or better to redo model training

- Pickle files of model per trial

- Parameters (SJON) per trial

- Plots to assess Optuna optimisation and to show learnings on relationship between features and performance during optimisation

- CSV of importance metrics for all features
  + 3 metrics: [shap](https://www.nature.com/articles/s42256-019-0138-9), lightGBM split and gain

- Plots showing most important features

# Sample data and outputs 

[link](https://drive.google.com/drive/folders/1aV484DkANCy8A3veGe1DN0B3YUz7HwoP?usp=share_link)

# Purpose

Create pipeline to seamlessly use ML at any step of bioinformatics workflow mainly for ranking features

- For my use case, features would mostly be genes/proteins and samples would be biological samples (bulk) or single-cells
 
### Potential use cases

- Predicting control vs. phenotype and identifying important features
  + This effectively selects features too reducing dimensionality and also identifies most important features with biological relevance (results can be compared with conventional methods for analysing data)
  + Rather than applying on original data with each gene or protein as features, can apply on PCA dimensions to identify important PC components for predicting phenotype and then examine key features correlated with those important PC components

- Integrating multi-omic or modal data to predict phenotypes
  + When different modalities of data/measurements are available e.g. RNA and proteins, this pipeline can be used to train and select features from each modality and then do another training after combining each modalities with reduced features
  
# Method

- Use ML framework that allows me to easily switch between predicting continuous or categorical labels (and easily switch metric accordingly) by changing a parameter in the pipeline
  + Considering LightGBM now over XGBoost for faster training times? Are there other frameworks that would be better for my purpose?
  
- optuna for automatic hyperparameter optimisation
  + Reduce spending time on hyperparameter optimisation
  + Set sensible ranges for optuna as defaults so user can focus on getting feature ranking

- Cross validation
  + To make sure that the model is robust while using all of my data especially when data is small
  + optuna has `optuna.integration.lightgbm.LightGBMTunerCV` so I guess this is possible when using LightGBM

# Outputs

- Panel of plots to assess model e.g. did not over or underfit
  + Figure out how I can get from `lightgbm.cv` output the training and validation metrics from each iteration for each fold so plots can be created
  + Figure out also how to assess optuna run
  + Essentially create a panel of plots to assess whether you can trust the feature importance output or not or better to redo model training

- Feature importance from each model for each fold
  + Not sure yet if this is possible with `lightgbm.cv` or it only gives aggregated feature importances (as well as metrics?)

- Panel of plots showing feature importances

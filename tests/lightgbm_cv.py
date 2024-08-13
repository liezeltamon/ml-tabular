# LightGBM CV + Optuna for tabular data

#%%

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from plotnine import ggplot, geom_point, aes, theme_classic, labs
from sklearn.datasets import load_breast_cancer, load_diabetes
import joblib, json, yaml

#%%
# Parameters

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

random_state = int(config["general"]["random_state"])
eval_metric = config["general"]["eval_metric"]
objective_type = config["general"]["objective_type"]
opt_direction = config["optimise"]["opt_direction"]
opt_n_trials = int(config["optimise"]["opt_n_trials"])
plot_n_trials = config["plot"]["plot_n_trials"] # If None, plot all trials
num_cores = int(config["general"]["num_cores"])
cv_nfold = int(config["cross_val"]["cv_nfold"])
cv_num_boost_round = int(config["cross_val"]["cv_num_boost_round"])
test = config["general"]["test"]

#%%
# Load data
if test and objective_type in ["binary", "multiclass"]:
    data, target = load_breast_cancer(return_X_y=True)
elif test and objective_type == "regression":
    data, target = load_diabetes(return_X_y=True)
elif not test:
    data = pd.read_csv("data.csv")
    target = pd.read_csv("target.csv")
else:
    raise ValueError("Problem with data loading")
#%%
# **Preprocessing**

#%%
# Optimise hyperparameters

def objective(trial, data=data, target=target, objective_type=objective_type, eval_metric=eval_metric, cv_num_boost_round=cv_num_boost_round, cv_nfold=cv_nfold):

    # Should be inside the objective function to not get this error:
    # LightGBMError: Reducing `min_data_in_leaf` with `feature_pre_filter=true` may cause unexpected behaviour for features that were pre-filtered by the larger `min_data_in_leaf`.
    dtrain = lgb.Dataset(data, label=target)

    param = {
        "objective": objective_type,
        "metric": eval_metric,
        "verbosity": -1,
        "boosting_type": "gbdt",
        "nfold": trial.suggest_int("nfold", 3, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    trial_results = lgb.cv(
        param,
        dtrain,
        num_boost_round=cv_num_boost_round,
        #nfold=cv_nfold,
        stratified=True,
        shuffle=True,
        metrics=eval_metric,
        seed=trial.number,
        eval_train_metric=True,
        return_cvbooster=True,
    )

    trial.set_user_attr("trial_results", trial_results)
    
    # Save the trial results and parameters
    joblib.dump(trial_results, f"trial_results_{trial.number}.pkl")
    with open(f"trial_params_{trial.number}.json", "w") as f:
        json.dump(trial_results["cvbooster"].boosters[0].params, f, indent=4)
    
    return max(trial_results["valid " + eval_metric + "-mean"])
#
#%%
# if __name__ == "__main__":
study = optuna.create_study(direction=opt_direction) # optuna.create_study(pruner=None) by default i.e. MedianPruner is used
study.optimize(objective, n_trials=opt_n_trials, n_jobs = num_cores)
############################################################################################################
#%%
# Assess whether you need to repeat optimisation or not

study_df = study.trials_dataframe()
study_df.sort_values("value", ascending=False, inplace=True)
study_df.to_csv("trials_study.csv", index=False)
study_df["number"] = pd.Categorical(study_df["number"], categories=study_df['number'].unique(), ordered=True)
study_df.head()

#%%
### 1) Plot showing rank of trials

p = (
    ggplot(study_df, aes(x="number", y="value"))
    + geom_point()
    + labs(y = "Optimisation value", x = "Trial number")
    + theme_classic()
)
p.save("trials_rank.png", dpi=300)

#%%
#### 2) Plot validation and training metrics across iterations to compare best and worst trials
if plot_n_trials is not None:
    plot_trials_inds = np.linspace(0, study_df.shape[0] - 1, num=int(plot_n_trials), endpoint=True, dtype=int)
    study_df = study_df.iloc[plot_trials_inds, :]

num_cols = min(study_df.shape[0], 3)
num_rows = np.ceil(study_df.shape[0] / num_cols).astype(int)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

# Get range of objective value for make y-axis consistent across all plots
minmax_array = np.full((study_df.shape[0], 2), None)
for i, trial_number in enumerate(study_df["number"]):
    trial_results = study.trials[int(trial_number)].user_attrs["trial_results"]
    all_metrics = trial_results["train " + eval_metric + "-mean"] + trial_results["valid " + eval_metric + "-mean"]
    minmax_array[i,0] = min(all_metrics)
    minmax_array[i,1] = max(all_metrics)
min_eval_metric = np.min(minmax_array)
max_eval_metric = np.max(minmax_array)

#%%
for i, trial_number in enumerate(study_df["number"]):
    #trial_results = joblib.load(f"trial_results_{trial_number}.pkl")
    trial_results = study.trials[int(trial_number)].user_attrs["trial_results"]
    train_metrics = trial_results["train " + eval_metric + "-mean"]
    valid_metrics = trial_results["valid " + eval_metric + "-mean"]
    iter_nums = range(0, len(train_metrics))
    
    row_idx = i // num_cols
    col_idx = i % num_cols
    axes[row_idx, col_idx].plot(iter_nums, train_metrics, color="turquoise", label="Train")
    axes[row_idx, col_idx].plot(iter_nums, valid_metrics, color="darkviolet", label="Validation")
    axes[row_idx, col_idx].set_xlabel("Iteration")
    axes[row_idx, col_idx].set_ylabel(f"Objective value: {eval_metric}")
    axes[row_idx, col_idx].set_title(f"Trial {trial_number}")
    axes[row_idx, col_idx].legend()
    axes[row_idx, col_idx].set_xlim(0, cv_num_boost_round)
    axes[row_idx, col_idx].set_ylim(min_eval_metric - (0.1 * min_eval_metric), max_eval_metric + (0.1 * max_eval_metric))
fig.tight_layout()
fig.savefig("trials_metrics.png", dpi=300)
plt.show()

#%%
# **Further explore study**
# Most important hyperparameters, performance for each range of heperparameters, etc.
# See https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html#visualization
# and https://optuna.readthedocs.io/en/stable/reference/visualization/matplotlib.html

#from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
plot_optimization_history(study)
plot_param_importances(study)

#%% 
# Explore feature importance
trial_number = 2
trial_results = joblib.load(f"trial_results_{trial_number}.pkl")
trial_cvbooster = trial_results["cvbooster"]
trial_cvbooster.feature_importance(importance_type="gain")
trial_cvbooster.feature_importance(importance_type="split")

# %%

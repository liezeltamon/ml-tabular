# LightGBM CV + Optuna for tabular data

#%%
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_timeline, plot_rank
import pandas as pd
from plotnine import ggplot, geom_point, aes, theme_classic, labs
import shap
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklego.dummy import RandomRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
#from statistics import mean, median
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
num_boost_round = int(config["predict"]["num_boost_round"])
nfold = int(config["predict"]["nfold"])
test = config["general"]["test"]
data_path = config["general"]["data_path"]
target_path = config["general"]["target_path"]

#%%
# Load data
if test and objective_type in ["binary", "multiclass"]:
    data, target = load_breast_cancer(return_X_y=True, as_frame=True)
elif test and objective_type == "regression":
    data, target = load_diabetes(return_X_y=True, as_frame=True)
elif not test:
    try:
        data = pd.read_csv(data_path)
        target = pd.read_csv("target_path")
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        data = pd.read_feather(data_path)
        target = pd.read_feather(target_path)
else:
    raise ValueError("Problem with data loading")

#%%
# **Preprocessing**

#### Missing values
data_missing = data.isnull().sum()
target_missing = target.isnull().sum()
print("Missing values in data:")
print(data_missing[data_missing > 0])
print("Missing values in target:")
print(target_missing[target_missing > 0])

#%%
# Cross-validation folds
folds = KFold(nfold, random_state=random_state, shuffle=True)

#%%
# Optimise hyperparameters

def objective(trial, data=data, target=target, objective_type=objective_type, eval_metric=eval_metric, num_boost_round=num_boost_round, folds=folds, random_state=random_state, opt_direction=opt_direction):

    if objective_type == "regression":
        stratified = False # Default is lightgbm.cv(stratified=True)
    elif objective_type in ["binary", "multiclass"]:
        stratified = True

    param = {
        "objective": objective_type,
        "metric": eval_metric,
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    # Should be inside the objective function to not get this error:
    # LightGBMError: Reducing `min_data_in_leaf` with `feature_pre_filter=true` may cause unexpected behaviour for features that were pre-filtered by the larger `min_data_in_leaf`.
    dtrain = lgb.Dataset(data, label=target)

    trial_results = lgb.cv(
        param,
        dtrain,
        num_boost_round=num_boost_round,
        folds=folds,
        stratified=stratified,
        shuffle=True, # Default
        eval_train_metric=True,
        return_cvbooster=True,
    )

    trial.set_user_attr("trial_results", trial_results)
    
    # Save the trial results and parameters
    joblib.dump(trial_results, f"trial_results_{trial.number}.pkl")
    with open(f"trial_params_{trial.number}.json", "w") as f:
        json.dump(trial_results["cvbooster"].boosters[0].params, f, indent=4)
    
    if opt_direction == "minimize":
        return min(trial_results["valid " + eval_metric + "-mean"])
    elif opt_direction == "maximize":
        return max(trial_results["valid " + eval_metric + "-mean"])
#
#%%
# if __name__ == "__main__":
study = optuna.create_study(direction=opt_direction) # optuna.create_study(pruner=None) by default i.e. MedianPruner is used
study.optimize(objective, n_trials=opt_n_trials, n_jobs = num_cores)

#%%
# Assess whether you need to repeat optimisation or not

study_df = study.trials_dataframe()
if opt_direction == "minimize":
    sort_direction = True
elif opt_direction == "maximize":
    sort_direction = False
study_df.sort_values("value", ascending=sort_direction, inplace=True)
study_df.to_csv("trials_study.csv", index=False)
study_df["number"] = pd.Categorical(study_df["number"], categories=study_df['number'].unique(), ordered=True)
study_df.head()

#%%
# Crate naive model/s and get metrics
if objective_type == "regression":
    naive_values_dict = {}
    for strat in ["mean", "median", "uniform", "normal"]:
        if strat in ["mean", "median"]:
            naive_target = DummyRegressor(strategy=strat).fit(data, target).predict(data)
        elif strat in ["uniform", "normal"]:
             # Based on https://koaning.github.io/scikit-lego/api/dummy/#sklego.dummy.RandomRegressor
            naive_target = RandomRegressor(strategy=strat, random_state=random_state).fit(data, target).predict(data)
        naive_values_dict[strat] = mean_squared_error(target, naive_target, squared=False)
    
elif objective_type in ["binary", "multiclass"]:
    naive_values_dict = {}
    for strat in ["most_frequent", "stratified", "uniform"]:
        naive_target = DummyClassifier(strategy=strat).fit(data, target).predict_proba(data)[:,1]
        naive_values_dict[strat] = roc_auc_score(target, naive_target)

#naive_metrics = {strat: np.full(num_boost_round, naive_value).tolist() for strat, naive_value in naive_values_dict.items()}

#%%
### 1) Plot showing rank of trials

p = (
    ggplot(study_df, aes(x="number", y="value"))
    + geom_point()
    + labs(y = "Optimisation value", x = "Trial number")
    + theme_classic()
)
p.save("trials_rank.pdf", dpi=300)

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
naive_values_list = [float(i) for i in naive_values_dict.values()]

for i, trial_number in enumerate(study_df["number"]):
    trial_results = study.trials[int(trial_number)].user_attrs["trial_results"]
    all_metrics = trial_results["train " + eval_metric + "-mean"] + trial_results["valid " + eval_metric + "-mean"] + naive_values_list
    minmax_array[i,0] = min(all_metrics)
    minmax_array[i,1] = max(all_metrics)
min_eval_metric = np.min(minmax_array)
max_eval_metric = np.max(minmax_array)

line_styles = {"mean": "--", "median": "-.", "uniform": ":", "normal": "-", "stratified": "--", "most_frequent": "-."}

for i, trial_number in enumerate(study_df["number"]):
    #trial_results = joblib.load(f"trial_results_{trial_number}.pkl")
    trial_results = study.trials[int(trial_number)].user_attrs["trial_results"]
    train_metrics = trial_results["train " + eval_metric + "-mean"]
    valid_metrics = trial_results["valid " + eval_metric + "-mean"]
    iter_nums = range(0, len(train_metrics))
    
    row_idx = i // num_cols
    col_idx = i % num_cols
    for strat, naive_value in naive_values_dict.items():
        axes[row_idx, col_idx].axhline(y=float(naive_value), color="grey", linestyle=line_styles[strat], label=f"Naive: {strat}")
    axes[row_idx, col_idx].plot(iter_nums, train_metrics, color="turquoise", label="Train")
    axes[row_idx, col_idx].plot(iter_nums, valid_metrics, color="darkviolet", label="Validation")
    axes[row_idx, col_idx].set_xlabel("Iteration")
    axes[row_idx, col_idx].set_ylabel(f"Objective value: {eval_metric}")
    axes[row_idx, col_idx].set_title(f"Trial {trial_number}")
    axes[row_idx, col_idx].legend()
    axes[row_idx, col_idx].set_xlim(0, num_boost_round)
    axes[row_idx, col_idx].set_ylim(min_eval_metric - (0.1 * min_eval_metric), max_eval_metric + (0.1 * max_eval_metric))
fig.tight_layout()
fig.savefig("trials_metrics.pdf", dpi=300)
plt.show()

#%%
#### 3) Optimisation diagnostic plots

fig = plot_optimization_history(study)
fig.write_image("plot_optimisation_history.pdf", height = 4*300, width = 6*300, engine="kaleido")

fig = plot_param_importances(study)
#n_mostimportant_param = fig.data[0]["y"][::-1][:5]
fig.write_image("plot_param_importances.pdf", height = 8*300, width = 6*300, engine="kaleido")

fig = plot_slice(study)
fig.write_image("plot_slice.pdf", height = 3*300, width = 12*300, engine="kaleido")

fig = plot_rank(study)
fig.write_image("plot_rank.pdf", height = 3*300, width = 12*300, engine="kaleido")

fig = plot_timeline(study)
fig.write_image("plot_timeline.pdf", height = 3*300, width = 4*300, engine="kaleido")

#%% 
# Save both split and gain from each trial so you can decide how to rank all features
# Get ave split and gain for each feature across all trials and folds
# then plot split x gain

folds_split_list = list(folds.split(data))

for i, trial_number in enumerate(study_df["number"]):
    trial_results = joblib.load(f"trial_results_{trial_number}.pkl")
    trial_cvbooster = trial_results["cvbooster"]
    
    if not all([trial_cvbooster.feature_name()[0] == i for i in trial_cvbooster.feature_name()]):
        raise ValueError(f"Trial number {trial_number}: Feature names are not the same across all folds")
    else:
        for j in ["gain", "split"]:
            df = pd.DataFrame(trial_cvbooster.feature_importance(importance_type=j)).T
            df.index = trial_cvbooster.feature_name()[0]
            df.to_csv(f"trial_{trial_number}_{j}.csv")

# %%
# Feature importance with shap values

with PdfPages('shap_beeswarm_plots.pdf') as pdf:

   fig, axes = plt.subplots(1, nfold, figsize=(20 * nfold, 5))
   plot_count = 0

   for i, trial_number in enumerate(study_df["number"]):
        # Using SHAP with Cross-Validation in Python - https://towardsdatascience.com/using-shap-with-cross-validation-d24af548fadc
        trial_results = joblib.load(f"trial_results_{trial_number}.pkl")
        boosters = trial_results["cvbooster"].boosters

        ave_shap_feature_list = []
        for j in range(nfold):
            explainer = shap.TreeExplainer(boosters[j])
            valid_inds = folds_split_list[j][1]
            explain_on_data = data.iloc[valid_inds, :] #data
        
            explainer_values = explainer(explain_on_data)
            shap_values = explainer_values.values # Same as shap_values = explainer.shap_values(explain_on_data)
            ave_shap_feature = np.mean(np.abs(shap_values), axis=0)
            ave_shap_feature_list.append(ave_shap_feature)

            #shap.plots.bar(explainer_values, max_display=30)
            shap.plots.beeswarm(explainer_values, max_display=30, show=False) #shap.summary_plot(shap_values, explain_on_data, max_display=30)
            f = plt.gcf()
            #shap_values_arr = np.hstack((explainer_values.values,
            #                             explainer_values.base_values.reshape(-1, 1)))
            #explainer_values.values.shape
            #explainer_values.base_values.shape

            plot_count += 1

            # Save the figure to the PDF every 3 plots
            if plot_count % nfold == 0:
                pdf.savefig(fig)
                plt.clf()
                fig, axes = plt.subplots(1, nfold, figsize=(20* nfold, 5))
   
   df = pd.DataFrame(ave_shap_feature_list).T
   df.index = boosters[j].feature_name()
   df.to_csv(f"trial_{trial_number}_shap.csv")

# %% 
# For testing
with PdfPages('shap_beeswarm_plots.pdf') as pdf:

    fig, axes = plt.subplots(1, nfold, figsize=(20 * nfold, 5))
    plot_count = 0

    for i, trial_number in enumerate(study_df["number"].iloc[0:2]):
        # Load trial results
        trial_results = joblib.load(f"trial_results_{trial_number}.pkl")
        boosters = trial_results["cvbooster"].boosters

        ave_shap_feature_list = []
        fig, axes = plt.subplots(1, nfold, figsize=(20 * nfold, 5))
        plot_count = 0

        for j in range(nfold):
            explainer = shap.TreeExplainer(boosters[j])
            valid_inds = folds_split_list[j][1]
            explain_on_data = data.iloc[valid_inds, :]  # Data for the current fold
            
            explainer_values = explainer(explain_on_data)
            shap_values = explainer_values.values  # SHAP values
            ave_shap_feature = np.mean(np.abs(shap_values), axis=0)
            ave_shap_feature_list.append(ave_shap_feature)

            # Generate the beeswarm plot
            plt.sca(axes[plot_count % nfold])  # Set the current axis
            shap.plots.beeswarm(explainer_values, max_display=30, show=False)
            
            plot_count += 1

            # Save the figure to the PDF every 3 plots
            if plot_count % nfold == 0:
                pdf.savefig(fig)
                plt.clf()
                fig, axes = plt.subplots(1, nfold, figsize=(20* nfold, 5))

        # Save any remaining plots
        if plot_count % nfold != 0:
            pdf.savefig(fig)
            plt.clf()

    df = pd.DataFrame(ave_shap_feature_list).T
    df.index = boosters[j].feature_name()
    df.to_csv(f"trial_{trial_number}_shap.csv")

# %%

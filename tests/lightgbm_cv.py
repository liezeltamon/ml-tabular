# Use xgboost_cv.py as reference - https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_cv.py

import lightgbm as lgb
import optuna
from sklearn.datasets import load_breast_cancer

# Parameters
eval_metric = "auc"
seed_value = int(290)
opt_direction = "maximize"
opt_n_trials = int(50)

def objective(trial):
    data, target = load_breast_cancer(return_X_y=True)
    dtrain = lgb.Dataset(data, label=target)

    param = {
        "objective": "binary",
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

    cv_results = lgb.cv(
        param,
        dtrain,
        num_boost_round=100,
        nfold=3,
        stratified=True,
        shuffle=True,
        metrics=eval_metric,
        seed=seed_value,
    )

    return max(cv_results["valid " + eval_metric + "-mean"])
#

if __name__ == "__main__":
    study = optuna.create_study(direction=opt_direction)
    study.optimize(objective, n_trials=opt_n_trials)
    #study.optimize(objective, n_trials=opt_n_trials, callbacks=[TerminatorCallback()])

    print(f"The number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value} (params: {study.best_params})")

############################################################################################################

# data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
# dtrain = lgb.Dataset(data, label=target)

# # param = {
# #     "objective": "binary",
# #     "metric": eval_metric,
# #     "verbosity": -1,
# #     "boosting_type": "gbdt",
# #     "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
# #     "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
# #     "num_leaves": trial.suggest_int("num_leaves", 2, 256),
# #     "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
# #     "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
# #     "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
# #     "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
# # }

# params = {
#     'objective': 'binary',              # Binary classification
#     'metric': 'auc',                    # Evaluation metric
#     'boosting_type': 'gbdt',            # Gradient Boosting Decision Tree
#     'num_leaves': 31,                   # Number of leaves in one tree
#     'learning_rate': 0.05,              # Learning rate
#     'feature_fraction': 0.9             # Fraction of features used for each tree
# }

# # Perform cross-validation
# cv_results = lgb.cv(
#     params,
#     dtrain,
#     num_boost_round = 100,
#     nfold=3,
#     stratified=True,
#     shuffle=True,
#     metrics='auc',
#     seed = seed_value,
#     eval_train_metric=True,
#     return_cvbooster=True
# )

# # Get length of values in a dictionary for each key
# for key, value in cv_results.items():
#     print(len(value))

# cv_boosters = cv_results['cvbooster']
# #cv_boosters.save_model('model.txt')
# print(f'Number of Boosters: {len(cv_boosters.boosters)}')

# # booster methods in - https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster
# booster_fold = cv_boosters.boosters[0]
# booster_fold.feature_name() # List of feature names
# booster_fold.feature_importance() # Numpy array of feature importance integers
# booster_fold.current_iteration()
# booster_fold.eval_train() # Dictionary of evaluation results on training data
# booster_fold.eval_valid() # Dictionary of evaluation results on validation data

# # Can be used to get train and valid metrics per iteration
# booster_fold.rollback_one_iter()
# booster_fold.current_iteration()
# booster_fold.eval_train()
# booster_fold.eval_valid()

# # Return the best score (maximum auc)
# #return max(cv_results['auc-mean'])

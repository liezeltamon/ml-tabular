# %% Benchmark model architectures
# env: ml-tabular-env
#sbatch -J benchmark_models -p short,long --mem=50G --output=%x.log.out --error=%x.log.err --wrap="python benchmark_models.py"

import os
import pandas as pd

from lazypredict.Supervised import LazyClassifier

# %% Parameters

benchmark_method = "lazypredict"  # "lazypredict" or "flaml"
lazypredict_sorter_key = "ROC AUC"  # "Accuracy", "Balanced Accuracy", "ROC AUC", "F1 Score", "Time Taken"
target_column = "label"

train_path = "../data/train.csv"
test_path = "../data/test.csv"

out_dir = "../results/benchmark_models"
os.makedirs(out_dir, exist_ok=True)

# %% ----- MAIN -----

# %% Load data

train_df = pd.read_csv(train_path, index_col=0)
test_df = pd.read_csv(test_path, index_col=0)

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

# %% Benchmark models

if benchmark_method == "lazypredict":
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    models = models.sort_values(by="ROC AUC", ascending=False)
    models = models.sort_values(by=lazypredict_sorter_key, ascending=False)

    print(models)
    models.reset_index().to_csv(
        os.path.join(out_dir, "lazypredict_results.csv"), index=False
    )

elif benchmark_method == "flaml":
    raise NotImplementedError(
        "FLAML workflow is still to be implemented and tested in benchmark_models.py."
    )

    # %% FLAML - Some code from https://medium.com/@mouse3mic3/a-practical-guide-to-automated-machine-learning-in-python-using-flaml-c73a714887a4

    # Check for using model outside of flaml - https://microsoft.github.io/FLAML/docs/FAQ/

    # import pickle
    # from flaml import AutoML
    # from sklearn.metrics import classification_report
    #
    # model_config = {
    #     "task": "classification",
    #     "time_budget": 300,     # in seconds
    #     "max_iter": 1000,       # num of model fitting allowed
    #     # "metric": sensitivity_metric,   # custom metric function
    #     # "estimator_list": ["lgbm", "histgb", "svm"],  # list of estimators used
    #     "eval_method": "cv",    # use RepeatedKFold
    #     # "split_type": RepeatedKFold(n_splits=3, n_repeats=10),
    #     # "custom_hp": custom_params,
    #     "ensemble": True,       # combine the two (or more) best performing estimators into one, effectively making an ensemble of classifiers
    #     "verbose": 3,
    #     "log_file_name": "training_log",
    #     "log_type": "all",
    # }
    #
    # automl = AutoML()
    # automl.add_learner("svm", SVM) # SVM is custom learner to be defined by user, check link how
    # automl.fit(X_train, y_train, **model_config)
    #
    # # Save Model
    # with open("flaml.pkl", "wb") as f:
    #     pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    #
    # # automl.add_learner("mylgbm", MyLGBMEstimator)
    # # automl.fit(
    # #     X_train=X_train,
    # #     y_train=y_train,
    # #     task="classification",
    # #     #metric=custom_metric,
    # #     estimator_list=["mylgbm"],
    # #     time_budget=60,
    # # )
    #
    # # Best model and best hyperparameters
    # automl.model.estimator
    #
    # # Best configuration per estimator
    # automl.best_config_per_estimator
    #
    # # Display which iteration resulted in the best model and how long it takes to find it.
    # print(
    #     "The ",
    #     automl.best_iteration,
    #     "-th iteration is the best, completed in ",
    #     round(automl.time_to_find_best_model, 1),
    #     " seconds.",
    #     sep="",
    # )
    #
    # # Predict test data
    # y_test_pred = automl.predict(X_test)
    #
    # # Display metrics
    # print(classification_report(y_test, y_test_pred))

else:
    raise ValueError("benchmark_method must be 'lazypredict' or 'flaml'")

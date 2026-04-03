# Check if you can reproduce training and validaiton metrics from each trial by rerunning using parameters and original folds
constant_param = {
        "objective": objective_type,
        "metric": eval_metric,
        "verbosity": -1,
        "boosting_type": "gbdt",
}

study.best_trial
is_equal = []
for i in range(len(study.trials)):
    trial = study.trials[i]
    dtrain = lgb.Dataset(data, label=target)
    trial_results = lgb.cv(
            {**constant_param, **trial.params},
            dtrain,
            num_boost_round=num_boost_round,
            folds=folds,
            #nfold=nfold,
            stratified=False,
            shuffle=True, # Default
            #seed=random_state,
            eval_train_metric=True,
            return_cvbooster=True,
    )
    print(trial_results['train rmse-mean'][0])
    is_equal = is_equal + [trial_results['train rmse-mean'] == trial.user_attrs["trial_results"]['train rmse-mean'],
    trial_results['train rmse-stdv'] == trial.user_attrs["trial_results"]['train rmse-stdv'],
    trial_results['valid rmse-mean'] == trial.user_attrs["trial_results"]['valid rmse-mean'],
    trial_results['valid rmse-stdv'] == trial.user_attrs["trial_results"]['valid rmse-stdv']]
is_equal
all(is_equal)

# Check that folds from all trials same with original folds
folds_split_list = list(folds.split(data))

is_equal = []
for i in range(len(study.trials)):
    #i = 1
    trial = study.trials[i]
    trial_folds_split_list = trial.user_attrs["folds"]
    for a in range(nfold):
        for b in range(2):
            is_equal = is_equal + [np.array_equal(folds_split_list[a][b], trial_folds_split_list[a][b])]
is_equal
all(is_equal)
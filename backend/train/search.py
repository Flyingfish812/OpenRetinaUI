import optuna
from backend.train.search_utils import (
    build_model_from_config, 
    train_and_evaluate_model, 
    get_model_params_name
)

def make_objective_function(model_class, fixed_config: dict, trial_config: dict, dataloader, eval_metric: str):
    def objective(trial):
        model_config = fixed_config.copy()
        train_config = {}

        for param_name, trial_def in trial_config.items():
            method = trial_def["method"]
            args = trial_def["args"]

            if method == "loguniform":
                value = trial.suggest_float(param_name, *args, log=True)
            elif method == "uniform":
                value = trial.suggest_float(param_name, *args)
            elif method == "int":
                value = trial.suggest_int(param_name, *args)
            elif method == "categorical":
                value = trial.suggest_categorical(param_name, args)
            else:
                raise ValueError(f"Unsupported optuna method: {method}")

            # 区分模型参数 vs 训练参数
            if param_name in get_model_params_name(model_class):
                model_config[param_name] = value
            else:
                train_config[param_name] = value

        model = build_model_from_config(model_class, model_config)
        return train_and_evaluate_model(model, dataloader, train_config, monitor=eval_metric)

    return objective

def run_hyperparameter_search(model_class, fixed_config, trial_config, dataloader, n_trials=30, eval_metric="val_correlation"):
    """
    执行 optuna 超参搜索
    """
    direction = "maximize" if "corr" in eval_metric.lower() else "minimize"
    study = optuna.create_study(direction=direction)

    objective = make_objective_function(model_class, fixed_config, trial_config, dataloader, eval_metric)
    study.optimize(objective, n_trials=n_trials)

    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials_df": study.trials_dataframe(),
        "study": study,
    }



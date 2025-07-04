# utils/hypersearch.py

import torch.nn as nn
import optuna
from inspect import signature, _empty
from utils.train import create_logger, create_early_stopping, create_lr_monitor, create_checkpoint, configure_external_optimizers
from utils.model import *  # 扩展热点
from lightning.pytorch import Trainer
from openretina.data_io.cyclers import LongCycler, ShortCycler

def build_model_from_config(model_class, config: dict):
    """
    根据模型类和参数字典实例化模型
    """
    return model_class(**config)

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

def get_model_params_name(model_class):
    sig = signature(model_class.__init__)
    return [name for name in sig.parameters if name != "self"]

def train_and_evaluate_model(model, dataloader, train_config=None, monitor="val_correlation"):
    train_config = train_config or {}

    logger = create_logger("data/log")
    patience = train_config.get("early_stop_patience", 15)
    min_delta = train_config.get("early_stop_min_delta", 1e-5)

    early_stopping = create_early_stopping(
        monitor=monitor,
        patience=patience,
        mode="max" if "corr" in monitor else "min",
        verbose=False,
        min_delta=min_delta,
    )

    lr_monitor = create_lr_monitor("epoch")
    checkpoint = create_checkpoint(
        monitor=monitor,
        mode="max" if "corr" in monitor else "min",
        save_weights_only=True
    )

    max_epochs = train_config.get("max_epochs", 100)
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stopping, lr_monitor, checkpoint],
        enable_progress_bar=False,
    )
    train_loader = LongCycler(dataloader["train"])
    val_loader = ShortCycler(dataloader["validation"])

    # 优化器配置（允许传入 start_lr 等）
    model.configure_optimizers = lambda: configure_external_optimizers(
        model,
        factor=train_config.get("scheduler_factor", 0.1),
        patience=train_config.get("scheduler_patience", 10),
        threshold=train_config.get("scheduler_threshold", 5e-4),
        start_lr=train_config.get("lr", 1e-3),
        min_lr=train_config.get("min_lr", 1e-6)
    )

    trainer.fit(model, train_loader, val_loader)

    val_metrics = trainer.callback_metrics
    val_score = val_metrics.get(monitor, None)
    if val_score is None:
        raise ValueError(f"Metric {monitor} not found in callback_metrics.")

    return val_score.item()

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


def get_model_params(model_class):
    """
    获取模型构造函数的参数名、类型和默认值
    """
    sig = signature(model_class.__init__)
    param_info = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        default = param.default if param.default is not _empty else None
        param_info.append({
            "name": name,
            "default": default,
            "type": str(param.annotation)
        })
    return param_info


def get_available_model_classes():
    """
    返回支持的模型名与类定义
    """
    return {
        "KlindtCoreReadout2D": KlindtCoreReadout2D,
        # "OtherModel": OtherModelClass
    }

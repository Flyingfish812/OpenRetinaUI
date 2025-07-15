import torch
from inspect import signature, _empty
from backend.train.trainer_utils import (
    create_logger, 
    create_early_stopping, 
    create_lr_monitor, 
    create_checkpoint, 
    configure_external_optimizers
)
from backend.model import *
from lightning.pytorch import Trainer
from openretina.data_io.cyclers import LongCycler, ShortCycler

def build_model_from_config(model_class, config: dict):
    """
    Instantiate a model class with the given configuration.
    """
    return model_class(**config)

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

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    max_epochs = train_config.get("max_epochs", 100)
    trainer = Trainer(
        accelerator=accelerator,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stopping, lr_monitor, checkpoint],
        enable_progress_bar=False,
    )
    train_loader = LongCycler(dataloader["train"])
    val_loader = ShortCycler(dataloader["validation"])

    model.configure_optimizers = lambda: configure_external_optimizers(
        model,
        factor=train_config.get("scheduler_factor", 0.1),
        patience=train_config.get("scheduler_patience", 10),
        threshold=train_config.get("scheduler_threshold", 1e-4),
        start_lr=train_config.get("lr", 1e-2),
        min_lr=train_config.get("min_lr", 1e-6)
    )

    trainer.fit(model, train_loader, val_loader)

    val_metrics = trainer.callback_metrics
    val_score = val_metrics.get(monitor, None)
    if val_score is None:
        raise ValueError(f"Metric {monitor} not found in callback_metrics.")

    return val_score.item()

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

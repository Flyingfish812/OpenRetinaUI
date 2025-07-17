import os
import torch
import datetime
import types
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# 设置 TensorBoardLogger
def create_logger(log_dir: str, run_name: str | None = None) -> TensorBoardLogger:
    if run_name is None:
        run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=run_name  # 这会在 log_dir/run_name 下保存日志
    )

    print(f"Log saved to: {os.path.join(log_dir, run_name)}")
    return logger

def get_tensorboard_url(port=6006):
    return f"http://127.0.0.1:{port}"

def create_early_stopping(monitor, patience, mode, verbose, min_delta):
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        verbose=verbose,
        min_delta=min_delta,
    )


def create_lr_monitor(interval):
    return LearningRateMonitor(logging_interval=interval)


def create_checkpoint(monitor, mode, save_weights_only):
    return ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_weights_only=save_weights_only
    )

def configure_external_optimizers(model, factor = 0.5, patience = 25, threshold = 0.0005, start_lr = 1e-4, min_lr = 1e-6):
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=1e-4)

    scheduler = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode="abs",
            min_lr=min_lr,
        ),
        "monitor": "val_loss",
        "interval": "epoch",
        "frequency": 1
    }

    return {"optimizer": optimizer, "lr_scheduler": scheduler}

def bind_configure_optimizers(model, config):
    def configure_optimizers_fn(self):
        return configure_external_optimizers(
            self,
            config["sr_factor"],
            config["sr_patience"],
            config["sr_threshold"],
            config["sr_startlr"],
            config["sr_minlr"]
        )
    # 绑定为实例方法
    model.configure_optimizers = types.MethodType(configure_optimizers_fn, model)
import os
import datetime
from ui.global_settings import LOG_SAVE_DIR, global_state
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

    print(f"📁 日志保存目录: {os.path.join(log_dir, run_name)}")
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
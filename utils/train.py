import os
import torch
import datetime
from ui.global_settings import LOG_SAVE_DIR, global_state
from utils.train_callbacks import LiveLogCallback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import Trainer
from openretina.data_io.cyclers import LongCycler, ShortCycler

# 设置 TensorBoardLogger
def create_logger(log_dir: str, run_name: str | None = None) -> TensorBoardLogger:
    if run_name is None:
        run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=run_name  # 这会在 log_dir/run_name 下保存日志
    )

    print(f"日志保存目录: {os.path.join(log_dir, run_name)}")
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

def trigger_train_from_config(config: dict):
    logger = create_logger(LOG_SAVE_DIR, config["train_name"])
    early_stopping = create_early_stopping(
        config["monitor"], config["patience"], config["mode"],
        config["verbose"], config["min_delta"]
    )
    lr_monitor = create_lr_monitor(config["interval"])
    model_checkpoint = create_checkpoint(
        config["monitor"], config["mode"], config["save_weights_only"]
    )
    live_log_callback = LiveLogCallback()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        accelerator=accelerator,
        max_epochs=config["max_epochs"],
        logger=logger,
        callbacks=[early_stopping, lr_monitor, model_checkpoint, live_log_callback],
    )

    dataloader = global_state["flattened_dataloader"] if config["input_2d"] else global_state["dataloader"]
    train_loader = LongCycler(dataloader["train"])
    val_loader = ShortCycler(dataloader["validation"])

    model = global_state["model"]
    model.configure_optimizers = lambda: configure_external_optimizers(
        model,
        config["sr_factor"],
        config["sr_patience"],
        config["sr_threshold"],
        config["sr_startlr"],
        config["sr_minlr"]
    )

    global_state["training_logs"] = []
    trainer.fit(model, train_loader, val_loader)

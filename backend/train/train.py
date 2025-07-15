import torch
from backend.utils import LOG_SAVE_DIR, global_state
from backend.train.train_callbacks import LiveLogCallback
from backend.train.trainer_utils import (
    create_logger,
    create_early_stopping,
    create_lr_monitor,
    create_checkpoint,
    configure_external_optimizers
)
from lightning.pytorch import Trainer
from openretina.data_io.cyclers import LongCycler, ShortCycler

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
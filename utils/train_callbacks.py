from lightning.pytorch.callbacks import Callback
from ui.global_settings import global_state

class LiveLogCallback(Callback):
    def __init__(self):
        super().__init__()
        self.logs = []

    def on_train_start(self, trainer, pl_module):
        msg = "Training start...\nTo check the full training logs, please find them your terminal."
        self.logs.append(msg)
        global_state["training_logs"].append(msg)

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        msg = f"Epoch {trainer.current_epoch} ends, val_correlation = {metrics.get('val_correlation', 'N/A')}\n"
        self.logs.append(msg)
        global_state["training_logs"].append(msg)

    def on_train_end(self, trainer, pl_module):
        msg = "Training finished.\nTo check the training info, use the button 'Activate TensorBoard' below."
        self.logs.append(msg)
        global_state["training_logs"].append(msg)

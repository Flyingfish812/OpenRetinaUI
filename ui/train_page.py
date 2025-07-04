import subprocess
import threading
import time
import io
import contextlib
import gradio as gr
import torch
from ui.global_settings import global_state, LOG_SAVE_DIR
from utils.data_io import print_dataloader_info
from utils.train import *

log_messages_train = []

def append_log_train(new_msg: str):
    log_messages_train.append(new_msg)
    return "\n".join(log_messages_train)

def check_training_materials(input_2d=False):
    log = []
    success_count = 0
    total_count = 2

    # 检查 dataloader
    dataloader = global_state.get("flattened_dataloader") if input_2d else global_state.get("dataloader")
    if dataloader is None:
        log.append("❌ DataLoader missing, please first prepare the data")
    else:
        success_count += 1
        log.append("✅ DataLoader found, with details:\n")
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            print_dataloader_info(dataloader)
            log.append(buf.getvalue())

    # 检查模型
    model = global_state.get("model")
    if model is None:
        log.append("❌ Model instance missing, please first build the model")
    else:
        success_count += 1
        log.append("✅ Model instance found, with details:\n")
        log.append(str(model))

    log.append(f"\n {success_count}/{total_count} checks passed.")

    return "\n".join(log)

def start_tensorboard(logdir=LOG_SAVE_DIR, port=6006):
    def run():
        subprocess.run([
            "tensorboard", f"--logdir={logdir}", f"--port={port}"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # 用线程启动，避免阻塞主进程
    threading.Thread(target=run, daemon=True).start()
    time.sleep(2)  # 稍作等待确保启动成功


def trigger_train(monitor, input_2d, train_name, patience, mode, verbose, min_delta, interval, save_weights_only, max_epochs,
                  sr_factor, sr_patience, sr_threshold, sr_startlr, sr_minlr):
    try:
        config = {
            "monitor": monitor,
            "input_2d": input_2d,
            "train_name": train_name,
            "patience": patience,
            "mode": mode,
            "verbose": verbose,
            "min_delta": min_delta,
            "interval": interval,
            "save_weights_only": save_weights_only,
            "max_epochs": max_epochs,
            "sr_factor": sr_factor,
            "sr_patience": sr_patience,
            "sr_threshold": sr_threshold,
            "sr_startlr": sr_startlr,
            "sr_minlr": sr_minlr
        }
        trigger_train_from_config(config)

        return "✅ Model trained successfully"
    except Exception as e:
        return f"❌ Training failed: {str(e)}"


def build_train_ui():
    with gr.Blocks() as train_page:
        gr.Markdown("# Training Settings")
        gr.Markdown("## Callbacks")
        gr.Markdown("Parameters for training callbacks: Early stopping, learning rate monitor, model checkpoint.")

        with gr.Row():
            monitor = gr.Textbox(label="Monitor Choice", value="val_loss")
            mode = gr.Dropdown(choices=["max", "min"], value="min", label="Optimization Mode")
            interval = gr.Dropdown(choices=["epoch", "step"], value="epoch", label="Learning Rate Monitor Interval")

        with gr.Row():
            patience = gr.Slider(1, 20, value=15, step=1, label="Early Stopping Patience")
            min_delta = gr.Number(value=1e-5, label="Early Stopping min_delta")
            verbose = gr.Checkbox(value=False, label="Logs")

        with gr.Row():
            save_weights_only = gr.Checkbox(value=False, label="Weights Only")
            max_epochs = gr.Slider(1, 300, value=100, step=1, label="Max Epochs")
        
        gr.Markdown("## Optimizer and Scheduler")
        gr.Markdown("Parameters for optimizer and scheduler.")

        with gr.Row():
            sr_factor = gr.Slider(0.1, 1.0, value=0.1, step=0.1, label="Scheduler Reduce Factor")
            sr_patience = gr.Slider(1, 30, value=15, step=1, label="Scheduler Patience")
        
        with gr.Row():
            sr_threshold = gr.Number(value=0.0001, label="Scheduler Threshold")
            sr_startlr = gr.Number(value=1e-2, label="Optimizer Start Learning Rate")
            sr_minlr = gr.Number(value=1e-6, label="Scheduler Min Learning Rate")

        gr.Markdown("## Materials Check and Training Session")

        with gr.Column():
            input_2d = gr.Checkbox(label="Use 2D Input", value=False)
            check_btn = gr.Button("Check Training Materials")
            check_output = gr.Textbox(lines=10, max_lines=10, label="Results", interactive=False)

        check_btn.click(fn=check_training_materials, inputs=[input_2d], outputs=check_output)

        def update_log_display():
            return "\n".join(global_state.get("training_logs", []))

        with gr.Row():
            train_name = gr.Textbox(label="Run Name (for TensorBoard)", placeholder="e.g., my_run_001")
            train_btn = gr.Button("Fit Model")
        
        output_log = gr.Textbox(label="Train log", lines=5, max_lines=5, interactive=False)
        timer = gr.Timer(value=1.0, active=False)
        timer.tick(fn=update_log_display, outputs=output_log)

        train_btn.click(
            fn=trigger_train,
            inputs=[monitor, input_2d, train_name, patience, mode, verbose, min_delta, interval, save_weights_only, max_epochs,
                    sr_factor, sr_patience, sr_threshold, sr_startlr, sr_minlr],
            outputs=output_log,
            show_progress=True
        )

        gr.Markdown("## TensorBoard Log")

        start_tb_btn = gr.Button("Activate TensorBoard")
        tb_frame = gr.HTML(label="TensorBoard Dashboard")

        def launch_tensorboard():
            start_tensorboard()

            html_code = """
            <script>
                window.open("http://127.0.0.1:6006", "_blank");
            </script>
            <a href="http://127.0.0.1:6006" target="_blank">Open TensorBoard</a>
            """
            return html_code

        start_tb_btn.click(fn=launch_tensorboard, outputs=tb_frame)


    return train_page
import os
import gradio as gr
from frontend import *
from backend.utils import global_state, favicon_abs_path
from backend.model import backup

def exit_app():
    os._exit(0)

css = """
footer {
    display: none !important;
}
#custom-footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    padding: 20px;
    margin-top: 40px;
}
"""

with gr.Blocks(title="Open Retina UI", css=css) as demo:
    with gr.Tabs():
        with gr.Tab("Data I/O"):
            build_dataio_ui()

        with gr.Tab("Dataloader"):
            build_dataloader_ui(demo)

        with gr.Tab("Search"):
            build_search_ui()

        with gr.Tab("Model"):
            build_model_instance_ui()

        with gr.Tab("Train"):
            build_train_ui()

        with gr.Tab("Visualizer"):
            build_visualizer_ui()

        with gr.Tab("Compare"):
            build_compare_ui()
        
        with gr.Row():
            b_exit = gr.Button("Quit Application")
            b_exit.click(exit_app)

        gr.HTML("<div id='custom-footer'>© 2025 Open Retina UI - Version 0.4.3 - Powered by Flyingfish812</div>")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, favicon_path=favicon_abs_path)

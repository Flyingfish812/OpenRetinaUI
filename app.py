import os
import gradio as gr
from ui.global_settings import *
from ui.data_io_page import *
from ui.dataloader_page import *
from ui.model_page import *
from ui.train_page import *

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
            gr.Markdown("# Load and Check Existing Dataloader")

            with gr.Row():
                dataloader_dropdown = gr.Dropdown(choices=list_dataloader_files(), label="Choose Dataloader File")
                metadata_autofill = gr.Textbox(label="Auto-fill Metadata File", interactive=False)
                read_button = gr.Button("Load Dataloader")

            output_dataloader = gr.Textbox(label="Console", lines=10, max_lines=10, interactive=False)

            dataloader_dropdown.change(fn=infer_metadata_filename, inputs=dataloader_dropdown, outputs=metadata_autofill)
            demo.load(fn=infer_metadata_filename, inputs=[dataloader_dropdown], outputs=metadata_autofill)
            read_button.click(fn=read_and_print_dataloader_info, inputs=dataloader_dropdown, outputs=output_dataloader)

        with gr.Tab("Model"):
            build_model_instance_ui()

        with gr.Tab("Train"):
            build_train_ui()
        
        with gr.Row():
            b_exit = gr.Button("Quit Application")
            b_exit.click(exit_app)

        gr.HTML("<div id='custom-footer'>© 2025 Open Retina UI — Powered by Flyingfish812</div>")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, favicon_path=favicon_abs_path)

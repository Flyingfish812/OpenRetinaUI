import os
import gradio as gr
from ui.global_settings import METRICS_SAVE_DIR
from utils.data_io import Unpickler
from utils.compare import *

def list_metrics_files():
    if not os.path.exists(METRICS_SAVE_DIR):
        return []
    return [f for f in os.listdir(METRICS_SAVE_DIR) if f.endswith('_metrics.pkl')]

def load_multiple_metrics(filenames):
    metrics_list = []
    errors = []
    for filename in filenames:
        filepath = os.path.join(METRICS_SAVE_DIR, filename)
        try:
            with open(filepath, 'rb') as f:
                metrics = Unpickler(f).load()
            short_name = filename.replace('_metrics.pkl', '')
            metrics_list.append((short_name, metrics))
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
    return metrics_list, "\n".join([f"✅ Loaded: {f}" for f, _ in metrics_list] + [f"❌ {e}" for e in errors])

def build_compare_ui():
    with gr.Blocks() as compare_page:
        gr.Markdown("## Compare Metrics Across Models")

        with gr.Row():
            refresh_btn = gr.Button("Refresh Metric Files")
            metrics_dropdown = gr.Dropdown(label="Select Metric Files", multiselect=True, choices=list_metrics_files())

        with gr.Row():
            load_btn = gr.Button("Load Selected Files")
            console = gr.Textbox(label="Console Output", lines=5, interactive=False)

        with gr.Row():
            boxplot_btn = gr.Button("Boxplot of Corrected R²")
            barchart_btn = gr.Button("Bar Chart of Corrected R²")
        
        with gr.Row():
            plot_output_box = gr.Image(type="pil", label="Comparison Plot")
            plot_output_bar = gr.Image(type="pil", label="Comparison Plot")

        with gr.Column():
            with gr.Row():
                scatter_index1 = gr.Number(label="Model Index A", value=0, precision=0)
                scatter_index2 = gr.Number(label="Model Index B", value=1, precision=0)
                scatter_btn = gr.Button("Scatter Comparison (First Two)")
            plot_output_scatter = gr.Image(type="pil", label="Comparison Plot")
        metrics_state = gr.State()

        refresh_btn.click(fn=list_metrics_files, outputs=metrics_dropdown)
        load_btn.click(fn=load_multiple_metrics, inputs=metrics_dropdown, outputs=[metrics_state, console])
        boxplot_btn.click(fn=plot_corrected_r2_boxplot, inputs=metrics_state, outputs=plot_output_box)
        barchart_btn.click(fn=plot_corrected_r2_barchart, inputs=metrics_state, outputs=plot_output_bar)

        def _indexed_scatter(metrics_list, index1, index2):
            index1 = int(index1)
            index2 = int(index2)
            if len(metrics_list) < 2:
                raise gr.Error("Need at least two models loaded.")
            if index1 >= len(metrics_list) or index2 >= len(metrics_list) or index1 < 0 or index2 < 0:
                raise gr.Error("Selected indices are out of range.")
            name1, metrics1 = metrics_list[index1]
            name2, metrics2 = metrics_list[index2]
            return plot_corrected_r2_scatter(name1, name2, metrics1["corrected_r2"], metrics2["corrected_r2"])

        scatter_btn.click(
            fn=_indexed_scatter,
            inputs=[metrics_state, scatter_index1, scatter_index2],
            outputs=plot_output_scatter
        )

    return compare_page

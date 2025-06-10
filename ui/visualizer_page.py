import gradio as gr
from ui.global_settings import global_state
from utils.visualizer import evaluate_model_final

log_messages_visualizer = []

def append_log_visualizer(msg: str):
    log_messages_visualizer.append(msg)
    return "\n".join(log_messages_visualizer)

def run_visualizer(n_bootstrap, save_plots, prefix):
    if global_state.get("model") is None:
        return append_log_visualizer("❌ Model not loaded.")
    if global_state.get("merged_data") is None:
        return append_log_visualizer("❌ Dataset not prepared.")
    results = evaluate_model_final(
        global_state["model"],
        global_state["merged_data"],
        n_bootstrap=int(n_bootstrap),
        save_plots=save_plots,
        plot_prefix=prefix
    )
    global_state["visualization_results"] = results
    return append_log_visualizer("✅ Visualization complete.")

def build_visualizer_ui():
    with gr.Blocks() as visualizer_page:
        gr.Markdown("# Visualizer")
        with gr.Row():
            n_bootstrap = gr.Number(value=100, label="Bootstrap Samples")
            save_plots = gr.Checkbox(value=True, label="Save Plots")
            prefix = gr.Textbox(label="Plot Prefix", value="final_")
            run_btn = gr.Button("Run Evaluation")
        output_box = gr.Textbox(lines=10, max_lines=10, interactive=False, label="Console")
        run_btn.click(run_visualizer, [n_bootstrap, save_plots, prefix], output_box)

    return visualizer_page

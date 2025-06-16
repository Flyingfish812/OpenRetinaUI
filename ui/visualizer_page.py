import os
import pickle
import gradio as gr
from ui.global_settings import global_state, METRICS_SAVE_DIR
from utils.data_io import Unpickler
from utils.visualizer import (
    compute_evaluation_metrics,
    plot_correlation_distribution,
    plot_reliability_distribution,
    plot_corr_vs_reliability,
    plot_corr_vs_reliability_with_indice,
    plot_fraction_of_ceiling,
    plot_example_prediction,
    plot_grid_predictions,
    plot_lsta,
    fig_to_buffer
)

log_messages_visualizer = []

def append_log_visualizer(msg: str):
    log_messages_visualizer.append(msg)
    return "\n".join(log_messages_visualizer)

def save_metrics_to_file(filename):
    try:
        metrics = global_state.get("metrics")
        if metrics is None:
            return "❌ No metrics available to save."
        filepath = os.path.join(METRICS_SAVE_DIR, f"{filename}_metrics.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(metrics, f)
        return append_log_visualizer("✅ Metrics saved successfully to " + filepath)
    except Exception as e:
        return append_log_visualizer(f"❌ Error saving metrics: {str(e)}")

def load_metrics_from_file(filename):
    try:
        filepath = os.path.join(METRICS_SAVE_DIR, filename)
        with open(filepath, 'rb') as f:
            metrics = Unpickler(f).load()

        global_state["metrics"] = metrics
        
        return append_log_visualizer("✅ Metrics loaded successfully from " + filepath)
    except Exception as e:
        return append_log_visualizer(f"❌ Error loading metrics: {str(e)}")

def list_metrics_files():
    if not os.path.exists(METRICS_SAVE_DIR):
        return []
    return [f for f in os.listdir(METRICS_SAVE_DIR) if f.endswith('_metrics.pkl')]

def run_metric_computation(is_2d_input: bool):
    try:
        model = global_state.get("model")
        dataloader = global_state.get("flattened_dataloader" if is_2d_input else "dataloader")
        normalized_data = global_state.get("normalized_data")

        if model is None:
            return append_log_visualizer("❌ Model not loaded.")
        if dataloader is None:
            return append_log_visualizer("❌ Dataloader not available.")
        if normalized_data is None:
            return append_log_visualizer("❌ Not available to build response data.")
        
        response_data = normalized_data.get("responses_test_by_trial")

        session_dict = dataloader.get("test")
        if session_dict is None or not session_dict:
            return append_log_visualizer("❌ Test set not found in dataloader.")

        result = compute_evaluation_metrics(model, session_dict, response_data, is_2d=is_2d_input)
        global_state["metrics"] = result
        append_log_visualizer("Model Evaluation Results:")
        append_log_visualizer(f"  MSE: {result['mse']:.4f}")
        append_log_visualizer(f"  RMSE: {result['rmse']:.4f}")
        append_log_visualizer(f"  Mean Correlation: {result['mean_correlation']:.4f}")
        append_log_visualizer(f"  Median Correlation: {result['median_correlation']:.4f}")
        append_log_visualizer(f"  Mean Fraction of Ceiling: {result['mean_fraction_of_ceiling']:.4f}")
        append_log_visualizer(f"  Median Fraction of Ceiling: {result['median_fraction_of_ceiling']:.4f}")
        return append_log_visualizer("✅ Metric computation complete.")
    except Exception as e:
        return append_log_visualizer(f"❌ Failed to compute metrics: {str(e)}")

def draw_metrics_summary():
    metrics = global_state.get("metrics")
    if metrics is None:
        return "❌ No metrics available."
    
    fig1 = plot_correlation_distribution(
        metrics["correlations"],
        metrics["mean_correlation"],
        metrics["median_correlation"]
    )
    fig2 = plot_reliability_distribution(metrics["reliability"])
    fig3 = plot_corr_vs_reliability(
        metrics["reliability"],
        metrics["correlations"]
    )
    fig4 = plot_corr_vs_reliability_with_indice(
        metrics["reliability"],
        metrics["correlations"]
    )
    fig5 = plot_fraction_of_ceiling(
        metrics["fraction_of_ceiling"],
        metrics["mean_fraction_of_ceiling"],
        metrics["median_fraction_of_ceiling"]
    )
    figs = [fig_to_buffer(fig1), fig_to_buffer(fig2), fig_to_buffer(fig3), fig_to_buffer(fig4), fig_to_buffer(fig5)]
    return figs, "✅ Metrics summary plots generated."

def draw_example_prediction():
    metrics = global_state.get("metrics")
    if metrics is None:
        return [], "❌ No metrics available."
    fig = plot_example_prediction(
        metrics["predictions"],
        metrics["targets"],
        metrics["reliability"],
        metrics["correlations"],
        metrics["fraction_of_ceiling"]
    )
    return fig_to_buffer(fig), "✅ Example prediction plot done."

def draw_grid_predictions():
    metrics = global_state.get("metrics")
    if metrics is None:
        return [], "❌ No metrics available."

    fig_list = plot_grid_predictions(
        predictions=metrics["predictions"],
        targets=metrics["targets"],
        correlations=metrics["correlations"]
    )

    images = [fig_to_buffer(fig) for fig in fig_list]
    return images, f"✅ Grid Prediction plot done, {len(images)} images in total."

def parse_index_string(index_string):
    index_string = index_string.strip()
    if not index_string:
        cell_indexs = None
    else:
        cell_indexs = []
        for part in index_string.split(','):
            part = part.strip()
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    cell_indexs.extend(range(start, end + 1))
                except ValueError:
                    continue
            else:
                try:
                    i = int(part)
                    cell_indexs.append(i)
                except ValueError:
                    continue
        cell_indexs = sorted(set(cell_indexs))
    return cell_indexs

def draw_lstas(index):
    metrics = global_state.get("metrics")
    if metrics is None:
        return [], "❌ No metrics available."

    cell_indexs = parse_index_string(index)
    fig_list = plot_lsta(
        images=metrics["images"],
        lstas=metrics["lsta"],
        cell_indexs=cell_indexs
    )

    images = [fig_to_buffer(fig) for fig in fig_list]
    return images, f"✅ LSTA plot done, {len(images)} images in total."

def build_visualizer_ui():
    with gr.Blocks() as visualizer_page:
        gr.Markdown("# Visualizer")

        with gr.Row():
            is_2d_input = gr.Checkbox(label="Use 2D Input", value=False)
            run_btn = gr.Button("Run Evaluation")
        with gr.Row():
            save_filename = gr.Textbox(label="Save as", value="test")
            save_btn = gr.Button("Save Metrics")
        with gr.Row():
            metric_dropdown = gr.Dropdown(label="Load Metrics", choices=list_metrics_files(), interactive=True)
            load_btn = gr.Button("Load Metrics")
        output_box = gr.Textbox(lines=10, max_lines=10, interactive=False, label="Console")
        save_btn.click(save_metrics_to_file, inputs=[save_filename], outputs=output_box)
        load_btn.click(load_metrics_from_file, inputs=[metric_dropdown], outputs=output_box)
        run_btn.click(run_metric_computation, inputs=[is_2d_input], outputs=output_box)

        gr.Markdown("## Visualization Plots")
        gr.Markdown("Summary plots include `Correlation`, `Reliability`, `Correlation vs Reliability`, and `Fraction of Ceiling`.")
        with gr.Column():
            summary_btn = gr.Button("Generate Summary Plots")
            summary_imgs = gr.Gallery(label="Summary Plots", columns=2, height="auto", preview=False)
        gr.Markdown("Example neuron predictions vs actual")
        with gr.Column():
            example_btn = gr.Button("Plot Example Prediction")
            example_img = gr.Image(label="Example Prediction Plot", type="pil", interactive=False)
        gr.Markdown("Grid of prediction plots for all neurons")
        with gr.Column():
            grid_btn = gr.Button("Plot Grid Predictions")
            grid_imgs = gr.Gallery(label="Grid Prediction Plots", columns=4, height="auto", preview=False)
        gr.Markdown("Grid of LSTAs for all neurons")
        with gr.Column():
            with gr.Row():
                lsta_indices = gr.Textbox(label="Cell indices (e.g. 0,1,2 or 0-3,5)", placeholder="Leave blank to plot all (may consume a lot of memory)")
                lsta_btn = gr.Button("Plot LSTAs")
            lsta_imgs = gr.Gallery(label="LSTA Plots", columns=4, height="auto", preview=False)

        status_box = gr.Textbox(label="Status", max_lines=1, interactive=False)

        summary_btn.click(draw_metrics_summary, outputs=[summary_imgs, status_box])
        example_btn.click(draw_example_prediction, outputs=[example_img, status_box])
        grid_btn.click(draw_grid_predictions, outputs=[grid_imgs, status_box])
        lsta_btn.click(draw_lstas, inputs=[lsta_indices], outputs=[lsta_imgs, status_box])

    return visualizer_page
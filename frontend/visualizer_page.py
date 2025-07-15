import os
import pickle
import gradio as gr
import numpy as np
from backend.utils import global_state, METRICS_SAVE_DIR, APP_DIR
from backend.dataio import Unpickler
from backend.viz import (
    compute_evaluation_metrics,
    fig_to_buffer
)
from backend.viz.plots import *

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
        normalized_data = global_state.get("normalized_data") or global_state.get("converted_data")

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

        # test_std = global_state.get("std_dict", {}).get("images_test", 1.0)
        test_std = 1.0
        result = compute_evaluation_metrics(model, session_dict, response_data, is_2d=is_2d_input, test_std=test_std)
        global_state["metrics"] = result
        append_log_visualizer("Model Evaluation Results:")
        append_log_visualizer(f"  MSE: {result['mse']:.4f}")
        append_log_visualizer(f"  RMSE: {result['rmse']:.4f}")
        append_log_visualizer(f"  Mean Correlation: {result['mean_correlation']:.4f}")
        append_log_visualizer(f"  Median Correlation: {result['median_correlation']:.4f}")
        if result.get("reliability") is not None:
            append_log_visualizer(f"  Mean Fraction of Ceiling: {result['mean_fraction_of_ceiling']:.4f}")
            append_log_visualizer(f"  Median Fraction of Ceiling: {result['median_fraction_of_ceiling']:.4f}")
            append_log_visualizer(f"  Mean Corrected R2: {result['mean_corrected_r2']:.4f}")
            append_log_visualizer(f"  Median Corrected R2: {result['median_corrected_r2']:.4f}")
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
    figs = [fig1, fig2, fig3, fig4, fig5]
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
    return fig, "✅ Example prediction plot done."

def draw_grid_predictions():
    metrics = global_state.get("metrics")
    if metrics is None:
        return [], "❌ No metrics available."

    fig_list = plot_grid_predictions(
        predictions=metrics["predictions"],
        targets=metrics["targets"],
        correlations=metrics["correlations"]
    )

    return fig_list, f"✅ Grid Prediction plot done, {len(fig_list)} images in total."

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
    lsta_data = global_state.get("lsta_data")
    if metrics is None or lsta_data is None:
        return [], "❌ Metrics and LSTA should both be available."

    cell_indexs = parse_index_string(index)
    fig_list = plot_lsta(
        images=metrics["images"],
        lsta_data=lsta_data["lsta"],
        lsta_model=metrics["lsta"],
        ellipses=lsta_data["ellipses"],
        image_indices=lsta_data["image_indices"],
        cell_indexs=cell_indexs
    )
    return fig_list, f"✅ LSTA plot done, {len(fig_list)} images in total."

def draw_kernels(is_2d=False, channels=None, time_frames=None):
    model = global_state.get("model")
    if model is None:
        return [], "❌ No model available."
    
    channel_indexs = parse_index_string(channels)
    time_frame_indexs = parse_index_string(time_frames)
    fig_list = plot_convolutional_kernels(
        model=model,
        is_2d=is_2d,
        channels=channel_indexs,
        time_frames=time_frame_indexs
    )

    return fig_list, f"✅ Kernel plot done, {len(fig_list)} images in total."

def draw_spatial_masks(cells):
    model = global_state.get("model")
    if model is None:
        return None, "❌ No model available."
    
    cell_indexs = parse_index_string(cells)

    image = plot_spatial_masks(
        model=model,
        cell_indices=cell_indexs
    )

    return image, f"✅ Spatial mask plot done."

def draw_feature_weights():
    model = global_state.get("model")
    if model is None:
        return None, "❌ No model available."

    image = plot_feature_weights(
        model=model
    )

    return image, f"✅ Feature weights plot done."

def draw_response_curves(indices):
    metrics = global_state.get("metrics")
    if metrics is None:
        return None, "❌ No model available."
    
    predictions = metrics.get("predictions")
    targets = metrics.get("targets")
    cell_indexs = parse_index_string(indices)
    image = plot_response_curves(
        predictions=predictions,
        targets=targets,
        cell_indices=cell_indexs
    )

    return image, "✅ Response curves plot done."

LSTA_DIR = os.path.join(APP_DIR, "data", "lsta")

def list_lsta_files():
    files = os.listdir(LSTA_DIR)
    return [f for f in files if f.endswith(".npz")]

def load_selected_lsta(file_name):
    file_path = os.path.join(LSTA_DIR, file_name)
    if not os.path.exists(file_path):
        return gr.update(), f"❌ File not found: {file_name}"

    data = np.load(file_path)
    global_state["lsta_data"] = dict(data)
    info = "\n".join([f"{k}: shape {v.shape}" for k, v in data.items()])
    return f"✅ Loaded {file_name}" + info

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

        gr.Markdown("## Model Structures")
        gr.Markdown("This part is for showing the structures of `Kernels`, `Spatial Mask`, and `Feature Weights`.")
        with gr.Column():
            with gr.Row():
                kernel_channels = gr.Textbox(label="Channels (e.g. 0,1,2 or 0-3,5)", placeholder="Leave blank to plot all (may consume a lot of memory)")
                kernel_time_frames = gr.Textbox(label="Time Frames (e.g. 0,1,2 or 0-3,5)", placeholder="Leave blank to plot all (may consume a lot of memory)")
                kernel_btn = gr.Button("Plot Kernels")
            kernel_imgs = gr.Gallery(label="Kernel Plots", columns=4, height="auto", preview=False)
        with gr.Column():
            with gr.Row():
                mask_cells = gr.Textbox(label="Cells (e.g. 0,1,2 or 0-3,5)", placeholder="Leave blank to plot all (may consume a lot of memory)")
                mask_btn = gr.Button("Plot Spatial Masks")
            mask_imgs = gr.Image(label="Spatial Mask Plot", type="pil", interactive=False)
        with gr.Column():
            feature_btn = gr.Button("Plot Feature Weights")
            feature_imgs = gr.Image(label="Feature Weights Plot", type="pil", interactive=False)
        
        model_box = gr.Textbox(lines=1, max_lines=1, interactive=False, label="Model Console")
        
        kernel_btn.click(draw_kernels, inputs=[is_2d_input, kernel_channels, kernel_time_frames], outputs=[kernel_imgs, model_box])
        mask_btn.click(draw_spatial_masks, inputs=[mask_cells], outputs=[mask_imgs, model_box])
        feature_btn.click(draw_feature_weights, outputs=[feature_imgs, model_box])

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
        status_box = gr.Textbox(label="Status", max_lines=1, interactive=False)
        with gr.Column():
            with gr.Row():
                response_indices = gr.Textbox(label="Cell indices (e.g. 0,1,2 or 0-3,5)",
                                          value="",
                                          placeholder="Leave blank to plot all (may consume a lot of memory)")
                response_btn = gr.Button("Plot Responses")
            response_imgs = gr.Image(label="Response Plots", type="pil", interactive=False)
        response_box = gr.Textbox(label="Response Status", max_lines=1, interactive=False)
        response_btn.click(draw_response_curves, inputs=[response_indices], outputs=[response_imgs, response_box])
        
        gr.Markdown("## LSTA")

        with gr.Row():
            lsta_file_dropdown = gr.Dropdown(label="Select LSTA dataset file", choices=list_lsta_files())
            lsta_load_btn = gr.Button("Load Selected File")
        lsta_file_info = gr.Textbox(label="LSTA File Contents", lines=5, interactive=False)

        with gr.Column():
            with gr.Row():
                lsta_indices = gr.Textbox(label="Cell indices (e.g. 0,1,2 or 0-3,5)",
                                          value="0,1,5,9,17,19,21,26,28,36,39",
                                          placeholder="Leave blank to plot all (may consume a lot of memory)")
                lsta_btn = gr.Button("Plot LSTAs")
            lsta_imgs = gr.Gallery(label="LSTA Plots", columns=4, height="auto", preview=False)
        lsta_box = gr.Textbox(label="LSTA Status", max_lines=1, interactive=False)

        lsta_load_btn.click(fn=load_selected_lsta,
                            inputs=lsta_file_dropdown,
                            outputs=[lsta_file_info])

        summary_btn.click(draw_metrics_summary, outputs=[summary_imgs, status_box])
        example_btn.click(draw_example_prediction, outputs=[example_img, status_box])
        grid_btn.click(draw_grid_predictions, outputs=[grid_imgs, status_box])
        lsta_btn.click(draw_lstas, inputs=[lsta_indices], outputs=[lsta_imgs, lsta_box])

    return visualizer_page
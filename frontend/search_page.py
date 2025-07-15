import gradio as gr
import pandas as pd
import os
import yaml
from backend.utils import global_state, MODEL_SAVE_DIR
from backend.train import get_available_model_classes, run_hyperparameter_search

# UI 状态缓存
search_log = []

# 日志输出
def append_log(msg: str):
    search_log.append(msg)
    return "\n".join(search_log)

# === yaml 读取 ===
def load_yaml_settings(file_path):
    try:
        with open(file_path, "r") as f:
            settings = yaml.safe_load(f)
        fixed_args = settings.get("fixed_args", settings.get("args", {}))
        search_args = settings.get("search_args", {})
        fixed_table = [[k, repr(v)] for k, v in fixed_args.items()]
        search_table = [[k, v["method"], repr(v["args"])] for k, v in search_args.items()]
        return fixed_table, search_table, f"[LOAD] Read {os.path.basename(file_path)}"
    except Exception as e:
        return [], [], f"[ERROR] Fail to read YAML: {str(e)}"
    
# === yaml 写入 ===
def fallback_representer(dumper, data):
    return dumper.represent_scalar("!str", str(data))

def save_yaml_settings(model_name, fixed_table, param_table):
    yaml.SafeDumper.add_multi_representer(object, fallback_representer)
    try:
        fixed_config = {k: eval(v) for k, v in fixed_table.values.tolist() if k}
        search_config = {k: {"method": m, "args": eval(a)} for k, m, a in param_table.values.tolist() if k}
        config = {
            "model_name": model_name,
            "fixed_args": fixed_config,
            "search_args": search_config
        }
        filename = f"{model_name}_search.yaml"
        path = os.path.join(MODEL_SAVE_DIR, filename)
        with open(path, "w") as f:
            yaml.safe_dump(config, f)
        return append_log(f"[SAVE] Settings saved to {filename}"), filename
    except Exception as e:
        return append_log(f"[ERROR] Fail to save YAML: {e}"), ""

def build_search_ui():
    with gr.Blocks() as search_page:
        gr.Markdown("# Parameter Search")

        with gr.Row():
            model_selector = gr.Dropdown(
                choices=list(get_available_model_classes().keys()),
                value="KlindtCoreReadout2D",
                label="Select Model",
            )
            n_trials = gr.Slider(1, 100, value=20, step=1, label="Trials Count")
            eval_metric = gr.Dropdown(choices=["val_loss", "val_correlation"], value="val_correlation", label="Metrics")

        param_table = gr.Dataframe(
            headers=["Parameter", "method", "args (tuple)"],
            datatype=["str", "str", "str"],
            row_count=(3, "dynamic"),
            label="Search Parameters and ranges",
        )

        fixed_table = gr.Dataframe(
            headers=["Parameter", "Value"],
            datatype=["str", "str"],
            row_count=(3, "dynamic"),
            label="Fixed Parameters"
        )

        with gr.Row():
            yaml_file = gr.Dropdown(
                choices=[f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith("_search.yaml")],
                label="Read YAML config"
            )
            load_btn = gr.Button("Read YAML")
            save_btn = gr.Button("Save YAML")

        with gr.Row():
            input_2d = gr.Checkbox(label="Use 2D Input", value=False)
            run_btn = gr.Button("Search")

        with gr.Row():
            best_result = gr.Textbox(label="Optimal Result", lines=10)
            console_log = gr.Textbox(label="Search Log", lines=10)

        def on_load_yaml(yaml_name):
            path = os.path.join(MODEL_SAVE_DIR, yaml_name)
            fixed, search, log = load_yaml_settings(path)
            return fixed, search, append_log(log)
        
        def on_save_yaml(model_name, fixed_table, param_table):
            log, _ = save_yaml_settings(model_name, fixed_table, param_table)
            return log

        load_btn.click(fn=on_load_yaml, inputs=[yaml_file], outputs=[fixed_table, param_table, console_log])
        save_btn.click(fn=on_save_yaml, inputs=[model_selector, fixed_table, param_table], outputs=[console_log])

        def launch_search(model_name, input_2d, n_trials, eval_metric, param_table, fixed_table):
            try:
                append_log(f"[INFO] Start searching {n_trials} trials using {model_name}...")

                model_class = get_available_model_classes()[model_name]
                dataloader = global_state.get("flattened_dataloader") if input_2d else global_state.get("dataloader")
                if dataloader is None:
                    return "", append_log("[ERROR] Cannot find dataloader, please load data first.")

                trial_config = {}
                for row in param_table.values.tolist():
                    try:
                        name, method, args = row
                        trial_config[name] = {
                            "method": method,
                            "args": eval(args)
                        }
                    except Exception as e:
                        append_log(f"Error parsing trial config row {row}: {e}")
                        continue

                fixed_config = {}
                for row in fixed_table.values.tolist():
                    try:
                        name, value = row
                        fixed_config[name] = eval(value)
                    except Exception as e:
                        append_log(f"Error parsing fixed config row {row}: {e}")
                        continue

                result = run_hyperparameter_search(
                    model_class=model_class,
                    fixed_config=fixed_config,
                    trial_config=trial_config,
                    dataloader=dataloader,
                    n_trials=n_trials,
                    eval_metric=eval_metric
                )

                msg = f"Optimal value: {result['best_value']:.4f}\nOptimal parameters:\n{result['best_params']}"
                return msg, append_log("[SUCCESS] Search completed")
            except Exception as e:
                return "", append_log(f"[ERROR] Fail to search: {str(e)}")

        run_btn.click(
            fn=launch_search,
            inputs=[model_selector, input_2d, n_trials, eval_metric, param_table, fixed_table],
            outputs=[best_result, console_log]
        )

    return search_page
import gradio as gr
import os
import yaml
from backend.utils import global_state, MODEL_SAVE_DIR
from backend.train import get_available_model_classes, run_hyperparameter_search
from frontend.utils import cast_value, extract_model_init_params, quote_string_fields

# UI 日志缓存
search_log = []
def append_log(msg: str):
    search_log.append(msg)
    return "\n".join(search_log)

# 自动根据模型提取参数并生成配置行
def auto_fill_config(model_name):
    model_class = get_available_model_classes()[model_name]
    params = extract_model_init_params(model_class)
    rows = []
    for name, ptype, default in params:
        # 默认值字符串化
        if isinstance(default, str):
            val_repr = repr(default)
        else:
            val_repr = str(default)
        rows.append([name, val_repr, False, "", ""])
    return rows

# 加载 YAML 到统一表格
def load_yaml_unified(yaml_name):
    path = os.path.join(MODEL_SAVE_DIR, yaml_name)
    settings = yaml.safe_load(open(path, 'r'))
    model_name = settings.get('model_name')
    fixed_args = settings.get('fixed_args', {})
    search_args = settings.get('search_args', {})
    rows = []
    params = extract_model_init_params(get_available_model_classes()[model_name])
    for name, ptype, default in params:
        # 固定参数优先，否则用默认
        val = fixed_args.get(name, default)
        val_repr = repr(val) if isinstance(val, str) else str(val)
        if name in search_args:
            search_flag = True
            method = search_args[name].get('method', '')
            args_repr = repr(search_args[name].get('args', ''))
        else:
            search_flag = False
            method = ''
            args_repr = ''
        rows.append([name, val_repr, search_flag, method, args_repr])
    return rows, append_log(f"[LOAD] Read {yaml_name}")

# 保存统一表格到 YAML
def save_yaml_unified(model_name, config_data):
    try:
        rows = config_data.values.tolist() if hasattr(config_data, 'values') else config_data
        fixed_config = {}
        search_config = {}
        # 类型映射用于固定参数转换
        params = extract_model_init_params(get_available_model_classes()[model_name])
        type_map = {n: t for n, t, _ in params}
        for row in rows:
            name, value, search_flag, method, args_str = row
            if not name:
                continue
            if search_flag:
                try:
                    search_config[name] = {"method": method, "args": eval(args_str)}
                except Exception as e:
                    append_log(f"Error parsing search args for {name}: {e}")
            else:
                try:
                    fixed_config[name] = cast_value(value, name, type_map.get(name, str))
                except Exception as e:
                    append_log(f"Error parsing fixed value for {name}: {e}")
        config = {
            "model_name": model_name,
            "fixed_args": fixed_config,
            "search_args": search_config,
        }
        # 强制字符串字段加引号
        config = quote_string_fields(config)
        filename = f"{model_name}_search.yaml"
        path = os.path.join(MODEL_SAVE_DIR, filename)
        with open(path, 'w') as f:
            yaml.safe_dump(config, f)
        return append_log(f"[SAVE] Settings saved to {filename}")
    except Exception as e:
        return append_log(f"[ERROR] Fail to save YAML: {e}")

# 执行搜索
def launch_search_unified(model_name, input_2d, n_trials, eval_metric, config_data):
    try:
        rows = config_data.values.tolist() if hasattr(config_data, 'values') else config_data
        append_log(f"[INFO] Start searching {n_trials} trials using {model_name}...")
        model_class = get_available_model_classes()[model_name]
        dataloader = global_state.get('flattened_dataloader') if input_2d else global_state.get('dataloader')
        if dataloader is None:
            return "", append_log("[ERROR] Cannot find dataloader, please load data first.")
        fixed_config = {}
        trial_config = {}
        params = extract_model_init_params(model_class)
        type_map = {n: t for n, t, _ in params}
        for row in rows:
            name, value, search_flag, method, args_str = row
            if not name:
                continue
            if search_flag:
                try:
                    trial_config[name] = {"method": method, "args": eval(args_str)}
                except Exception as e:
                    append_log(f"Error parsing trial config row {row}: {e}")
            else:
                try:
                    fixed_config[name] = cast_value(value, name, type_map.get(name, str))
                except Exception as e:
                    append_log(f"Error parsing fixed config row {row}: {e}")
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

# 构建 Gradio UI

def build_search_ui():
    with gr.Blocks() as search_page:
        gr.Markdown("# Parameter Search")
        with gr.Row():
            model_selector = gr.Dropdown(
                choices=list(get_available_model_classes().keys()),
                value=list(get_available_model_classes().keys())[0],
                label="Select Model",
            )
            n_trials = gr.Slider(1, 100, value=20, step=1, label="Trials Count")
            eval_metric = gr.Dropdown(
                choices=["val_loss", "val_correlation"],
                value="val_correlation",
                label="Metrics"
            )
        # 统一参数表格
        initial_rows = auto_fill_config(model_selector.value)
        config_table = gr.Dataframe(
            headers=["Parameter", "Value", "Search?", "Method", "Args"],
            datatype=["str", "str", "bool", "str", "str"],
            value=initial_rows,
            row_count=(len(initial_rows), "dynamic"),
            label="Parameter Configuration"
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

        # 事件绑定
        # 模型切换自动填充表格
        model_selector.change(
            fn=auto_fill_config,
            inputs=[model_selector],
            outputs=[config_table]
        )
        # 加载 YAML
        load_btn.click(
            fn=load_yaml_unified,
            inputs=[yaml_file],
            outputs=[config_table, console_log]
        )
        # 保存 YAML
        save_btn.click(
            fn=save_yaml_unified,
            inputs=[model_selector, config_table],
            outputs=[console_log]
        )
        # 运行搜索
        run_btn.click(
            fn=launch_search_unified,
            inputs=[model_selector, input_2d, n_trials, eval_metric, config_table],
            outputs=[best_result, console_log]
        )

    return search_page

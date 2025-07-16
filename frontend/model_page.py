import gradio as gr
import numpy as np
import os
import torch
import torch.nn as nn
import yaml
from torchinfo import summary
from frontend.utils import cast_value, quote_string_fields, extract_model_init_params
from backend.utils import global_state, MODEL_SAVE_DIR
from backend.model import *
from openretina.models.core_readout import CoreReadout

MAX_PARAMS = 50

# 可选模型注册表
available_models = {
    "Klindt Core Readout 2D": KlindtCoreReadout2D,
    "Klindt Core Readout 3D": KlindtCoreReadout3D,
    "Linear-Nonlinear 2D": LNCoreReadout2D,
    "Linear-Nonlinear 3D": LNCoreReadout3D,
    "LNLN 2D": LNLNCoreReadout2D,
    "(Open Retina Default) Core Readout": CoreReadout,
}

log_messages_model = []

def append_log_model(new_msg: str):
    log_messages_model.append(new_msg)
    return "\n".join(log_messages_model)

def render_param_fields(model_name, param_container, input_widgets, input_component_list):
    model_class = available_models[model_name]
    params = extract_model_init_params(model_class)
    
    # 确保有足够的预分配控件
    if len(input_component_list) < MAX_PARAMS:
        # 初始化时填充足够的 Textbox
        with param_container:
            with gr.Row():
                for i in range(len(input_component_list), MAX_PARAMS):
                    tb = gr.Textbox(visible=False, label=f"param_{i}", interactive=True)
                    input_component_list.append(tb)
    
    # 更新控件：显示需要的，隐藏多余的
    updates = []
    for i, (name, ptype, default) in enumerate(params):
        if i >= MAX_PARAMS:
            break  # 超过最大支持参数数，忽略
        input_widgets[name] = input_component_list[i]
        updates.append(gr.update(
            visible=True,
            label=name,
            value=str(default),
        ))
    
    # 隐藏未使用的控件
    for i in range(len(params), MAX_PARAMS):
        updates.append(gr.update(visible=False))
    
    return updates

def trigger_build_model(*args, model_name, input_widgets):
    try:
        model_class = available_models[model_name]
        append_log_model(f"Model name: {model_name}, Model class: {model_class}")
        params = extract_model_init_params(model_class)
        typed_kwargs = {
            name: cast_value(value, name, ptype)
            for (name, ptype, _), value in zip(params, args)
        }
        append_log_model(f"\nBuilding models with following parameters: \n{typed_kwargs}")
        model = model_class(**typed_kwargs)
        global_state["model"] = model
        global_state["model_settings"] = {"name": model_name, "args": typed_kwargs}
        return append_log_model(f"\n✅ Model successfully built: \n{model}")
    except Exception as e:
        return append_log_model(f"\n❌ Fail to build model: {str(e)}")
    
# YAML Read/Write
def save_model_and_settings(filename):
    try:
        model = global_state.get("model")
        settings = global_state.get("model_settings")
        if model is None or settings is None:
            return append_log_model("❌ Please build the model first")

        torch.save(model, os.path.join(MODEL_SAVE_DIR, f"{filename}_model.pth"))

        # 强制加引号包裹字符串字段
        quoted_settings = quote_string_fields(settings)

        with open(os.path.join(MODEL_SAVE_DIR, f"{filename}_settings.yaml"), "w") as f:
            yaml.safe_dump(quoted_settings, f)

        return append_log_model(f"✅ Saved model to: {filename}_model.pth\nSetting parameters to: {filename}_settings.yaml")
    except Exception as e:
        return append_log_model(f"❌ Fail to save: {str(e)}")

def load_settings_and_fill(settings_path, model_selector, param_container, input_widgets, input_component_list):
    try:
        with open(settings_path, "r") as f:
            settings = yaml.safe_load(f)

        model_name = settings.get("name")
        args = settings.get("args")
        if model_name not in available_models:
            return [gr.update()] * (1 + len(input_component_list)) + [
                append_log_model(f"❌ Model `{model_name}` in settings is not available in the model list")
            ]

        render_param_fields(model_name, param_container, input_widgets, input_component_list)

        model_selector_update = gr.update(value=model_name)
        param_updates = []
        params = extract_model_init_params(available_models[model_name])

        for i, (name, _, _) in enumerate(params):
            widget = input_widgets.get(name)
            if widget:
                v = args.get(name, "")
                if isinstance(v, str) and not (v.startswith('"') or v.startswith("'")):
                    v = f'"{v}"'  # 在界面中加引号，提示用户是字符串
                param_updates.append(gr.update(value=str(v)))
            else:
                param_updates.append(gr.update())
        for _ in range(len(params), MAX_PARAMS):
            param_updates.append(gr.update(visible=False))

        log_text = append_log_model(f"✅ Setting loaded successfully with model built: {model_name}")

        global_state["model_settings"] = args
        global_state["model"] = torch.load(settings_path.replace("_settings.yaml", "_model.pth"), weights_only=False)

        return [model_selector_update] + param_updates + [log_text]
    except Exception as e:
        return [gr.update()] * (1 + len(input_component_list)) + [
            append_log_model(f"❌ Fail to load settings: {str(e)}")
        ]
    
def build_summary(input_size):
    try:
        model = global_state.get("model")
        input_size = cast_value(input_size, "", "tuple")
        model_stats = summary(model, input_size=input_size, verbose=0)
        summary_str = str(model_stats)
        return f"Model summary:\n" + summary_str
    except Exception as e:
        return f"Error generating model summary: {str(e)}"

def build_model_instance_ui():
    with gr.Blocks() as model_ui:
        gr.Markdown("# Model Constructor")

        # 顶部：读取设置文件
        with gr.Row():
            settings_file = gr.Dropdown(
                label="Read Settings File",
                choices=[f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith("_settings.yaml")],
                interactive=True
            )
            read_settings_btn = gr.Button("Read Settings")

        model_selector = gr.Dropdown(
            choices=list(available_models.keys()),
            label="Choose a model",
            value="Klindt Core Readout 3D"
        )

        param_container = gr.Column()
        input_widgets = {}
        input_component_list = []

        output_log = gr.Textbox(label="Console", lines=12, max_lines=20, show_copy_button=True)
        build_btn = gr.Button("Build Model")

        with gr.Column():
            with gr.Row():
                input_size = gr.Textbox(
                    label="Input Size",
                    placeholder="(B,C,H,W) for 2D or (B,C,T,H,W) for 3D",
                    value="(1, 1, 108, 108)"
                )
                summary_btn = gr.Button("Build Summary")
            summary_output = gr.Textbox(label="Model Summary", lines=10, max_lines=20, show_copy_button=True)
        summary_btn.click(
            fn=lambda input_size: build_summary(input_size),
            inputs=[input_size],
            outputs=summary_output
        )

        with gr.Row():
            save_name = gr.Textbox(label="Save as", placeholder="ex. my_model")
            save_btn = gr.Button("Save Model and Settings")
        save_btn.click(
            fn=save_model_and_settings,
            inputs=[save_name],
            outputs=output_log
        )

        # 初始化参数控件
        init_updates = render_param_fields(model_selector.value, param_container, input_widgets, input_component_list)
        model_ui.load(fn=lambda: init_updates,
            outputs=input_component_list
        )
        
        # 点击创建模型按钮
        build_btn.click(
            fn=lambda *args: trigger_build_model(*args[:-1], model_name=args[-1], input_widgets=input_widgets),
            inputs=input_component_list + [model_selector],
            outputs=output_log
        )

        # 更换模型时刷新参数控件
        def update_on_model_change(model_name):
            return render_param_fields(model_name, param_container, input_widgets, input_component_list)

        model_selector.change(fn=update_on_model_change, inputs=model_selector, outputs=input_component_list)

        # 点击读取设置按钮
        read_settings_btn.click(
            fn=lambda fname: load_settings_and_fill(
                os.path.join(MODEL_SAVE_DIR, fname),
                model_selector,
                param_container,
                input_widgets,
                input_component_list
            ),
            inputs=[settings_file],
            outputs=[model_selector, *input_component_list, output_log]
        )

        

    return model_ui
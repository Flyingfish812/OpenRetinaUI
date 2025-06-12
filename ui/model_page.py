import gradio as gr
import numpy as np
import os
import pickle
import yaml
from inspect import signature, _empty
from typing import get_type_hints
from ui.global_settings import global_state, MODEL_SAVE_DIR
from utils.model3d import *
from utils.model import *
from openretina.models.core_readout import CoreReadout

MAX_PARAMS = 50

# 可选模型注册表
available_models = {
    "Klindt Core Readout 3D": KlindtCoreReadout3D,
    "Klindt Core Readout 2D": KlindtCoreReadout2D,
    "(Open Retina Default) Core Readout": CoreReadout,
}

log_messages_model = []

def append_log_model(new_msg: str):
    log_messages_model.append(new_msg)
    return "\n".join(log_messages_model)

def extract_model_init_params(model_class):
    sig = signature(model_class.__init__)
    type_hints = get_type_hints(model_class.__init__)
    params = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        ptype = type_hints.get(name, str)
        default = param.default if param.default is not _empty else ""
        params.append((name, ptype, default))
    return params

def cast_value(value, target_type):
    try:
        if value in ("None", "", None):
            return None
        if target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            return value.lower() in ["true", "1"]
        elif target_type in [list, tuple] or str(target_type).startswith(("list", "tuple")):
            result = eval(value, {"np": np})
            return list(result) if isinstance(result, tuple) else result
        elif "np.array" in value:
            return eval(value, {"np": np})
        else:
            return eval(value)
    except Exception as e:
        raise ValueError(f"Parameter `{value}` cannot be cast to {target_type}: {e}")

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
    # try:
    model_class = available_models[model_name]
    append_log_model(f"Model name: {model_name}, Model class: {model_class}")
    params = extract_model_init_params(model_class)
    typed_kwargs = {
        name: cast_value(value, ptype)
        for (name, ptype, _), value in zip(params, args)
    }
    append_log_model(f"\nBuilding models with following parameters: \n{typed_kwargs}")
    model = model_class(**typed_kwargs)
    global_state["model"] = model
    global_state["model_settings"] = {"name": model_name, "args": typed_kwargs}
    return append_log_model(f"\n✅ Model successfully built: \n{model}")
    # except Exception as e:
        # return append_log_model(f"\n❌ Fail to build model: {str(e)}")

def fallback_representer(dumper, data):
        return dumper.represent_scalar("!str", str(data))

def save_model_and_settings(filename):
    yaml.SafeDumper.add_multi_representer(object, fallback_representer)
    try:
        model = global_state.get("model")
        settings = global_state.get("model_settings")
        if model is None or settings is None:
            return append_log_model("❌ Please build the model first")

        # with open(os.path.join(MODEL_SAVE_DIR, f"{filename}_model.pkl"), "wb") as f:
        #     pickle.dump(model, f)
        torch.save(model, os.path.join(MODEL_SAVE_DIR, f"{filename}_model.pth"))

        with open(os.path.join(MODEL_SAVE_DIR, f"{filename}_settings.yaml"), "w") as f:
            yaml.safe_dump(settings, f)

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

        # 重建参数控件
        render_param_fields(model_name, param_container, input_widgets, input_component_list)

        # 构建输出更新列表
        model_selector_update = gr.update(value=model_name)
        param_updates = []
        params = extract_model_init_params(available_models[model_name])
        for i, (name, _, _) in enumerate(params):
            widget = input_widgets.get(name)
            if widget:
                param_updates.append(gr.update(value=str(args.get(name, ""))))
            else:
                param_updates.append(gr.update())
        # 填满剩余控件
        for _ in range(len(params), MAX_PARAMS):
            param_updates.append(gr.update(visible=False))

        log_text = append_log_model(f"✅ Setting loaded successfully with model built: {model_name}")

        global_state["model_settings"] = args
        # global_state["model"] = available_models[model_name](**args)
        global_state["model"] = torch.load(settings_path.replace("_settings.yaml", "_model.pth"), weights_only=False)

        return [model_selector_update] + param_updates + [log_text]
    except Exception as e:
        return [gr.update()] * (1 + len(input_component_list)) + [
            append_log_model(f"❌ Fail to load settings: {str(e)}")
        ]

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
from inspect import signature, _empty
from typing import get_type_hints
import numpy as np
from typing import Union, Iterable, get_origin, get_args
from backend.utils import global_state

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

def quote_string_fields(settings_dict):
    """
    给 settings_dict["args"] 中所有字符串值加引号（防止后续 YAML 转储出错）
    """
    args = settings_dict.get("args", {})
    for k, v in args.items():
        if isinstance(v, str) and not (v.startswith('"') or v.startswith("'")):
            args[k] = f'"{v}"'
    return settings_dict

def resolve_base_type(target_type):
    """
    递归解析typing类型注解，提取所有基础类型（如 int, float, list, str, bool）。
    """
    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin is Union:
        types = set()
        for arg in args:
            types |= resolve_base_type(arg)
        return types
    elif origin in (list, tuple, Iterable):
        return {list}
    elif target_type in (int, float, bool, str):
        return {target_type}
    else:
        return {str}

def cast_value(value, name, target_type):
    """
    安全地将字符串值转换为目标类型，兼容Gradio输入和typing复杂类型（如Union、Iterable等）。
    """
    if value in ("None", "", None):
        return None

    value = str(value).strip()
    accepted_types = resolve_base_type(target_type)

    # 特例处理：init_mask
    if name == "init_mask":
        if value.lower() == "default":
            return global_state.get("init_mask", None)
        try:
            return eval(value, {"np": np})
        except:
            return value

    # 布尔识别
    if bool in accepted_types:
        if value.lower() in ["true", "false", "1", "0"]:
            return value.lower() in ["true", "1"]

    # 整数识别
    if int in accepted_types:
        try:
            return int(value)
        except:
            pass

    # 浮点识别
    if float in accepted_types:
        try:
            return float(value)
        except:
            pass

    # 列表/元组/np.array识别
    if list in accepted_types:
        try:
            return eval(value, {"np": np})
        except:
            pass
    if "np.array" in value:
        try:
            return eval(value, {"np": np})
        except:
            pass

    # 字符串识别（防止eval("abs") → <built-in function abs>）
    if str in accepted_types:
        try:
            result = eval(value, {"np": np})
            if callable(result):
                return value  # 避免函数体
            return result
        except:
            return value

    # 最终兜底：尝试eval（可能是表达式或数组）
    try:
        return eval(value, {"np": np})
    except Exception as e:
        raise ValueError(f"❌ `{name}` = `{value}` cannot be cast to {target_type}: {e}")

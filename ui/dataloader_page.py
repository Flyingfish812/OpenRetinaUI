import os
import io
import contextlib
import pickle
from utils.data_io import (
    Unpickler,
    load_dataloader,
    load_metadata,
    print_dataloader_info,
)
from ui.global_settings import LOADER_DATA_DIR, global_state

log_messages_dataloader = []

def append_log_dataloader(new_msg: str):
    log_messages_dataloader.append(new_msg)
    return "\n".join(log_messages_dataloader)

def list_dataloader_files():
    return [f for f in os.listdir(LOADER_DATA_DIR) if f.endswith("dataloader.pkl")]

def infer_metadata_filename(dataloader_filename: str):
    if "_flattened_" in dataloader_filename:
        base = dataloader_filename.replace("_flattened_dataloader.pkl", "")
    elif "_unflattened_" in dataloader_filename:
        base = dataloader_filename.replace("_unflattened_dataloader.pkl", "")
    else:
        return ""
    return f"{base}_metadata.pkl"

def read_and_print_dataloader_info(dataloader_filename: str):
    try:
        dataloader_path = os.path.join(LOADER_DATA_DIR, dataloader_filename)
        metadata_path = os.path.join(LOADER_DATA_DIR, infer_metadata_filename(dataloader_filename))
        dataloader_obj = load_dataloader(dataloader_path)
        metadata_obj = load_metadata(metadata_path)

        global_state["metadata"] = metadata_obj
        if metadata_obj["flattened"]:
            append_log_dataloader(f"Attention: Metadata indicates this is a flattened DataLoader. You should use 2D models to train it.")
            global_state["flattened_dataloader"] = dataloader_obj
        else:
            global_state["dataloader"] = dataloader_obj
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            print_dataloader_info(dataloader_obj)
            info = buf.getvalue()
        return append_log_dataloader(info + "\n✅ DataLoader loaded successfully")
    except Exception as e:
        return append_log_dataloader(f"❌ 读取失败: {str(e)}")
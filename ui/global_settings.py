import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
favicon_abs_path = os.path.join(APP_DIR, "retina.ico")

RAW_DATA_DIR = os.path.join(APP_DIR, "data", "raw")
LOADER_DATA_DIR = os.path.join(APP_DIR, "data", "dataloader")
MODEL_SAVE_DIR = os.path.join(APP_DIR, "data", "model")
LOG_SAVE_DIR = os.path.join(APP_DIR, "data", "log")
METRICS_SAVE_DIR = os.path.join(APP_DIR, "data", "metrics")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(LOADER_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_SAVE_DIR, exist_ok=True)

global_state = {
    "raw_data": None,
    "converted_data": None,
    "normalized_data": None,
    "init_mask": None,
    "lsta_data": None,
    "merged_data": None,
    "metadata": None,
    "dataloader": None,
    "flattened_dataloader": None,
    "model": None,
    "model_settings": None,
    "training_logs": [],
    "metrics": None,
}
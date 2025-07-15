from .compare_page import build_compare_ui
from .data_io_page import build_dataio_ui
from .dataloader_page import build_dataloader_ui
from .model_page import build_model_instance_ui
from .search_page import build_search_ui
from .train_page import build_train_ui
from .visualizer_page import build_visualizer_ui

__all__ = [
    "build_compare_ui",
    "build_dataio_ui",
    "build_dataloader_ui",
    "build_model_instance_ui",
    "build_search_ui",
    "build_train_ui",
    "build_visualizer_ui",
]
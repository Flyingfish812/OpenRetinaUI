import io
import os
import contextlib
import pickle
import gradio as gr
from backend.dataio import (
    DatasetLoader,
    convert_format,
    normalize_data,
    prepare_data_and_metadata,
    build_train_test_splits,
    print_dataloader_info,
    strip_all_dataloaders
)
from openretina.data_io.base_dataloader import multiple_movies_dataloaders
from backend.utils import RAW_DATA_DIR, LOADER_DATA_DIR, global_state

# Find all raw .pkl files in the RAW_DATA_DIR
def list_datasets():
    return [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".pkl")]

log_messages_dataio = []

def append_log_dataio(new_msg: str):
    log_messages_dataio.append(new_msg)
    return "\n".join(log_messages_dataio)

# Read raw data from a .pkl file
def step_load_raw(filename):
    path = os.path.join(RAW_DATA_DIR, filename)
    loader = DatasetLoader()
    try:
        raw_data = loader.load_numpy_arrays(path)
        global_state["raw_data"] = raw_data

        info = []
        info.append(f"Training size: {raw_data['images_train'].shape}")
        info.append(f"Training responses size: {raw_data['responses_train'].shape}")
        info.append(f"Validation size: {raw_data['images_val'].shape}")
        info.append(f"Validation responses size: {raw_data['responses_val'].shape}")
        info.append(f"Test size: {raw_data['images_test'].shape}")
        info.append(f"Test responses size: {raw_data['responses_test'].shape}")

        image_size = raw_data["images_train"].shape[1]
        num_neurons = raw_data["responses_train"].shape[1]
        info.append(f"Image shape: {image_size}")
        info.append(f"Number of neurons: {num_neurons}")

        info.append("✅ Data loaded successfully")

        return append_log_dataio("\n".join(info))
    except Exception as e:
        return append_log_dataio(f"❌ Data load failed: {str(e)}")

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

# Convert the raw data format
def step_convert_format(indices=None):
    if global_state["raw_data"] is None:
        return append_log_dataio("❌ Please read the data first")
    try:
        cell_indexs = parse_index_string(indices)
        converted = convert_format(global_state["raw_data"], cell_indexs)
        global_state["converted_data"] = converted
        global_state["init_mask"] = converted.get("mask", None)

        info = []
        info.append(f"Training size: {converted['images_train'].shape}")
        info.append(f"Training responses size: {converted['responses_train'].shape}")
        info.append(f"Validation size: {converted['images_val'].shape}")
        info.append(f"Validation responses size: {converted['responses_val'].shape}")
        info.append(f"Test size: {converted['images_test'].shape}")
        if converted.get('responses_test_by_trial', None) is not None:
            info.append(f"Test responses with trail: {converted['responses_test_by_trial'].shape}")
        else:
            info.append(f"Test responses with trail: {converted.get('responses_test_by_trial', None)}")
        info.append("✅ Format convert successfully")
        return append_log_dataio("\n".join(info))
    except Exception as e:
        return append_log_dataio(f"❌ Format convert failed: {str(e)}")

# Normalize the converted data
def step_normalize():
    if global_state["converted_data"] is None:
        return append_log_dataio("❌ Please convert the format first")
    try:
        normalized, _, _ = normalize_data(global_state["converted_data"])
        global_state["normalized_data"] = normalized
        return append_log_dataio("✅ Normalization Success")
    except Exception as e:
        return append_log_dataio(f"❌ Normalization failed: {str(e)}")

# Merge data (train and validation)
def step_prepare(train_chunk_size, batch_size, seed, clip_length):
    if global_state["converted_data"] is None:
        return append_log_dataio("❌ Please convert the format first")
    try:
        data = global_state.get("normalized_data") or global_state.get("converted_data")
        merged_data, metadata = prepare_data_and_metadata(
            normalized_data=data,
            train_chunk_size=int(train_chunk_size),
            batch_size=int(batch_size),
            seed=int(seed),
            clip_length=int(clip_length)
        )
        global_state["merged_data"] = merged_data
        global_state["metadata"] = metadata

        info = []
        info.append("✅ Create and merge data successfully")
        for k, v in metadata.items():
            info.append(f"{k}: {v}")

        return append_log_dataio("\n".join(info))
    except Exception as e:
        return append_log_dataio(f"❌ Merge failed: {str(e)}")

# Build DataLoader from the merged data
def build_dataloader():
    if global_state["merged_data"] is None:
        return append_log_dataio("❌ Please prepare the data first")
    try:
        merged_data = global_state["merged_data"]
        metadata = global_state["metadata"]
        movie, response = build_train_test_splits(merged_data)
        movie_stimuli = {"default": movie}
        response_stimuli = {"default": response}
        test_loaders = multiple_movies_dataloaders(
            neuron_data_dictionary=response_stimuli,
            movies_dictionary=movie_stimuli,
            train_chunk_size=metadata["train_chunk_size"],
            batch_size=metadata["batch_size"],
            seed=metadata["seed"],
            clip_length=metadata["clip_length"],
            num_val_clips=metadata["num_val_clips"],
            val_clip_indices=metadata["val_clip_indices"],
            allow_over_boundaries=True  # 默认值，根据需要可设为False
        )
        global_state["dataloader"] = test_loaders
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            print_dataloader_info(test_loaders)
            printed_output = buf.getvalue()

        return append_log_dataio(printed_output + "\n✅ DataLoader successfully built")
    except Exception as e:
        return append_log_dataio(f"❌ Build DataLoader failed: {str(e)}")

# Optional: Flatten the DataLoader for 2D use
def flatten_dataloader():
    if global_state["dataloader"] is None:
        return append_log_dataio("❌ Please build the DataLoader first")
    try:
        dataloader = global_state["dataloader"]
        flattened = strip_all_dataloaders(dataloader)
        global_state["flattened_dataloader"] = flattened
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            print_dataloader_info(flattened)
            printed_output = buf.getvalue()
        return append_log_dataio(printed_output + "\n✅ DataLoader Flattened")
    except Exception as e:
        return append_log_dataio(f"❌ Flatten DataLoader failed: {str(e)}")

# Save the DataLoader and metadata to files
def save_metadata_and_dataloader(filename, save_flattened):
    if global_state["merged_data"] is None or global_state["metadata"] is None:
        return append_log_dataio("❌ Please prepare the data first")
    try:
        metadata = global_state["metadata"]

        if save_flattened:
            dataloader_to_save = global_state.get("flattened_dataloader")
            metadata["flattened"] = True
            suffix = "flattened"
        else:
            dataloader_to_save = global_state.get("dataloader")
            suffix = "unflattened"

        if dataloader_to_save is None:
            return append_log_dataio(f"❌ DataLoader ({suffix}) does not exist, please build it first")

        base_name = filename.replace(".pkl", "")
        metadata_path = os.path.join(LOADER_DATA_DIR, f"{base_name}_metadata.pkl")
        dataloader_path = os.path.join(LOADER_DATA_DIR, f"{base_name}_{suffix}_dataloader.pkl")

        with open(os.path.join(LOADER_DATA_DIR, f"{base_name}_metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
        with open(os.path.join(LOADER_DATA_DIR, f"{base_name}_{suffix}_dataloader.pkl"), "wb") as f:
            pickle.dump(dataloader_to_save, f)
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            print_dataloader_info(dataloader_to_save)
            loader_info = buf.getvalue()

        return append_log_dataio(
            f"✅ Save DataLoader ({suffix}) to: {dataloader_path}\n"
            f"✅ Metadata saved to: {metadata_path}\n"
            f"✅ DataLoader structure: \n{loader_info}"
        )
    except Exception as e:
        return append_log_dataio(f"❌ Save DataLoader failed: {str(e)}")
    
def build_dataio_ui():
    gr.Markdown("# Pre-Processing Data I/O")
    with gr.Row():
        dataset_dropdown = gr.Dropdown(choices=list_datasets(), label="Choose a dataset (.pkl)", interactive=True)

    pick_indices = gr.Textbox(label="Cell indices (e.g. 0,1,2 or 0-3,5)",
                                value="",
                                placeholder="Leave blank to pick all")

    with gr.Row():
        b1 = gr.Button("Step 1: Raw Data Input")
        b2 = gr.Button("Step 2: Format Conversion")
        b3 = gr.Button("Step 3: Normalization")

    with gr.Row():
        input_chunk = gr.Number(label="Train Chunk Size", value=1)
        input_batch = gr.Number(label="Batch Size", value=32)
        input_seed = gr.Number(label="Random Seed", value=42)
        input_clip = gr.Number(label="Validation Clip Length", value=1)

    with gr.Row():
        b4 = gr.Button("Step 4: Merge Data + Generate Metadata")
        b5 = gr.Button("Step 5: Build DataLoader")
        b6 = gr.Button("Optional: Flatten DataLoader (2D use)")

    with gr.Row():
        save_filename = gr.Textbox(label="Save as", value="backup_data.pkl")
        save_flattened_check = gr.Checkbox(label="Save flattened version", value=False)
        b_save = gr.Button("Save Dataloader and Metadata")

    output_dataio = gr.Textbox(label="Console", lines=10, max_lines=10, interactive=False, show_copy_button=True)

    b1.click(step_load_raw, inputs=dataset_dropdown, outputs=output_dataio)
    b2.click(step_convert_format, inputs=[pick_indices], outputs=output_dataio)
    b3.click(step_normalize, outputs=output_dataio)
    b4.click(step_prepare, inputs=[input_chunk, input_batch, input_seed, input_clip], outputs=output_dataio)
    b5.click(build_dataloader, outputs=output_dataio)
    b6.click(flatten_dataloader, outputs=output_dataio)
    b_save.click(save_metadata_and_dataloader, inputs=[save_filename, save_flattened_check], outputs=output_dataio)
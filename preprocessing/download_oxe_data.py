import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import numpy as np
import json
import argparse

import tensorflow_datasets as tfds
import tensorflow as tf
from .utils import get_settings

def feature_to_metadata(feature):
    # Extract feature metadata from the builder.
    # Use to get the metadata about the actions and transform it into dictionary for saving with hdf5.
    if isinstance(feature, tfds.features.FeaturesDict):
        return {
            key: feature_to_metadata(value)
            for key, value in feature.items()
        }

    metadata = {
        "type": type(feature).__name__,
    }

    if hasattr(feature, "shape") and feature.shape is not None:
        metadata["shape"] = list(feature.shape)

    if hasattr(feature, "dtype") and feature.dtype is not None:
        metadata["dtype"] = str(feature.dtype)

    if hasattr(feature, "doc") and feature.doc:
        metadata["description"] = str(feature.doc)

    if isinstance(feature, tfds.features.ClassLabel):
        metadata["num_classes"] = feature.num_classes
        metadata["names"] = feature.names

    return metadata


def open_dataset(builder_dir: str, n_episodes: int, verbose=True):
    builder = tfds.builder_from_directory(
        builder_dir=builder_dir
    )

    print("====================================================")

    action_feature = builder.info.features["steps"]["action"]

    if verbose:
        print(f"Action feature:\n{action_feature}")

    # Convert TFDS feature objects into plain Python dictionaries
    action_metadata = feature_to_metadata(action_feature)

    if verbose:
        print("Action metadata:")
        print(action_metadata)

    dataset = builder.as_dataset(
        split=f"train[:{n_episodes}]"
    )

    print(
        f"Entire data size on disk: "
        f"{builder.info.dataset_size / 1e9:.2f} GB"
    )

    return dataset, action_metadata

def _get_candidate_keys(dimension: str):
    # Get keys for actions and images depending on datasets

    if dimension == "image":
        return ["image", "rgb_gripper", "rgb_static"]
    elif dimension == "language":
        return ["language_instruction", "natural_language_instruction"]
    # TODO: Incorporate a ValueError loop -> dimension has to be "image" or "language"

#### save data slice as hdf5

# transform into hdf5 format. Useful for uploading with an environment without tensorflow.
def save_hdf5_dataset(path_data: str, data_name_lst: list, n_episodes: int, oxe_dataset_name: str):
    with h5py.File(f"{path_data}/{oxe_dataset_name}.h5", "w") as f:
        for data_name in data_name_lst:
            print("\n\n====================================================")
            print(f"Dataset: {data_name}")
            dataset, action_metadata = open_dataset(builder_dir=f"gs://gresearch/robotics/{data_name}/0.1.0", n_episodes=n_episodes, verbose=False)
            dataset_grp = f.create_group(f"{data_name}")

            for i, episode in enumerate(dataset.take(n_episodes)):
                grp_episode = dataset_grp.create_group(f"episode_{i}")
                steps = list(episode["steps"])

                # Collect across all steps first
                ## images
                candidate_image_keys = _get_candidate_keys(dimension="image") # keys that might contain images.
                grp_images = grp_episode.create_group("images")
                for key in candidate_image_keys:
                    if key in steps[0]['observation'].keys():
                        images = np.stack([s["observation"][key].numpy() for s in steps])
                        grp_images.create_dataset(key, data=images)

                ## actions
                grp_action = grp_episode.create_group("actions")
                if isinstance(steps[0]["action"], dict):
                    for key in steps[0]['action'].keys():
                        action = np.stack([s["action"][key].numpy() for s in steps])
                        grp_action.create_dataset(key, data=action)
                else:
                    action = np.stack([s["action"].numpy() for s in steps])
                    grp_action.create_dataset("actions", data=action)
                
                ## language
                candidate_language_keys = _get_candidate_keys(dimension="language")
                grp_language = grp_episode.create_group("language")
                if any("language" in key for key in steps[0].keys()):
                    for key in candidate_language_keys:
                        if key in steps[0].keys():
                            language_instructions = np.stack([s[key].numpy() for s in steps])
                            grp_language.create_dataset(key, data=language_instructions)
                else:
                    for key in candidate_language_keys:
                        if key in steps[0]['observation'].keys():
                            language_instructions = np.stack([s["observation"][key].numpy() for s in steps])
                            grp_language.create_dataset(key, data=language_instructions)
            
            dataset_grp.attrs["actions_metadata"] = json.dumps(action_metadata)
    
    # Print data size
    file_path = Path(f"{path_data}/{oxe_dataset_name}.h5")
    size_bytes = file_path.stat().st_size
    print(f"Size of file {file_path}: {size_bytes / (1024 ** 2):.2f} MB")
    print(f"Size of file {file_path}: {size_bytes / (1024 ** 3):.2f} GB")




# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # python download_oxe_data.py --settings cfg.json
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, default="cfg.json")
    args = parser.parse_args()

    settings = get_settings(settings=str(Path(__file__).resolve().parent.parent / f"{args.settings}"))    
    data_name_lst = settings["data_name_lst"] # characteristics: Franka, Single Arm, EEF Position.
    n_episodes = settings["n_episodes"]
    oxe_dataset_name = settings["oxe_dataset_name"]

    save_hdf5_dataset(path_data=str(Path(__file__).resolve().parent.parent / "data"), 
                    data_name_lst=data_name_lst,
                    n_episodes=n_episodes,
                    oxe_dataset_name=oxe_dataset_name)

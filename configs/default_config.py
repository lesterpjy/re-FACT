#!/usr/bin/env python3
"""
default_config.py

A placeholder configuration file. You can store default or custom
settings for your project here. In practice, you might have multiple
config files for different experiments or environments.
"""

CONFIG = {
    # Model / architecture
    "model_type": "MyAwesomeModel",
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 10,

    # Data / paths
    "train_data_path": "/local/data/train",   # adjust to your directory structure
    "val_data_path":   "/local/data/val",     # adjust to your directory structure

    # Device settings
    "device": "cuda",  # or "cpu"

    # Misc
    "random_seed": 42,
}

def get_config():
    """
    Returns a copy of the default configuration dictionary.
    In a real-world project, you might add logic here to
    load different configs based on environment variables,
    command-line flags, or HPC cluster settings.
    """
    return CONFIG.copy()


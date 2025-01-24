import os
import sys
import importlib.util
from dotenv import load_dotenv
from loguru import logger

import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from huggingface_hub import login

from .data_utils import EAPDataset, prepare_bias_corrupt
from .utils import get_metric
from .config import Config


def load_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA (NVIDIA GPU)
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS (Apple Silicon GPU)
        print("Using MPS device")
    else:
        device = torch.device("cpu")  # Fallback to CPU
        print("Using CPU device")
    return device


def load_model(config: Config):
    logger.info("Loading model and tokenizer...")
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    model_name = config.model_name  # Access the model_name directly from the config
    model_name_noslash = model_name.split("/")[-1]
    config.model_name_noslash = model_name_noslash
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        device=config.device,
        fold_ln=True,
    )
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    config.model = model
    config.tokenizer = model.tokenizer
    logger.info("Model and tokenizer loaded.")


def load_dataset(config: Config):
    logger.info("Loading dataset...")
    ds = EAPDataset(config)
    config.dataloader = ds.to_dataloader(config.batch_size)

    config.task_metric = get_metric(config.metric_name, config.task, model=config.model)
    config.kl_div = get_metric("kl_divergence", config.task, model=config.model)
    logger.info("Dataset and metrics loaded.")


def load_config(config_path: str, process_data: bool = False) -> Config:
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if hasattr(config, "CONFIG"):
        config_dict = config.CONFIG
    else:
        raise KeyError("CONFIG dictionary not found in the config file.")

    # Create a Config instance with the necessary attributes
    config_obj = Config(
        model_name=config_dict["model_name"],
        random_seed=config_dict["random_seed"],
        data_dir=config_dict["data_dir"],
        work_dir=config_dict["work_dir"],
        debug=config_dict["debug"],
        labels=config_dict["labels"],
        task=config_dict["task"],
        data_split=config_dict["data_split"],
        metric_name=config_dict["metric_name"],
        batch_size=config_dict["batch_size"],
        run=config_dict["run"],
        datapath=config_dict.get("dataset_path", None),
        from_generated_graphs=config_dict.get("from_generated_graphs", False),
    )

    # Dynamically load the device
    config_obj.device = load_device()

    # Load the model with the updated config object
    load_model(config_obj)
    config_obj.configure_logger()

    if process_data and config_obj.task == "bias":
        prepare_bias_corrupt(config_obj)
    load_dataset(config_obj)

    return config_obj

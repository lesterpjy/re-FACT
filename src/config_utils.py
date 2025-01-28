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

from .data_utils import EAPDataset, prepare_bias_corrupt, prepare_toxicity_corrupt
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


def load_token():
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in .env file")
    return token


def load_model(config: Config):
    logger.info("Loading model and tokenizer...")
    login(token="")
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


def load_config(config_path: str) -> Config:
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
        process_data=config_dict.get("process_data", False),
        from_generated_graphs=config_dict.get("from_generated_graphs", False),
        tiny_sample=config_dict.get("tiny_sample", None),
    )
    logger.info(f"loaded task name: {config_obj.task}")
    logger.info(f"loaded path name: {config_obj.datapath}")
    # Dynamically load the device
    config_obj.device = load_device()

    # Load the model with the updated config object
    load_model(config_obj)
    config_obj.configure_logger()
    # '../data/circuit_identification_data/bias_tiny'

    if config_obj.process_data and "bias" in config_obj.task:
        prepare_bias_corrupt(config_obj)
    if config_obj.process_data and "toxicity" in config_obj.task:
        prepare_toxicity_corrupt(config_obj)
    load_dataset(config_obj)

    return config_obj

import os
import sys
import importlib.util
from typing import Optional, List, Callable
from dotenv import load_dotenv
from loguru import logger

import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from huggingface_hub import login

from data_utils import EAPDataset, prepare_bias_corrupt
from utils import get_metric


class Config:
    def __init__(
        self,
        model_name: str,
        random_seed: int,
        data_dir: str,
        work_dir: str,
        debug: bool,
        labels: List[str],
        task: str,
        data_split: int,
        metric_name: str,
        batch_size: int,
        datapath: Optional[str] = None,
    ):
        self.model_name: str = model_name
        self.random_seed: int = random_seed
        self.data_dir: str = data_dir
        self.work_dir: str = work_dir
        self.debug: bool = debug
        self.labels: List[str] = labels
        self.task: str = task
        self.data_split: int = data_split
        self.metric_name: str = metric_name
        self.batch_size: int = batch_size
        self.datapath: Optional[str] = datapath

        self.device: torch.device = torch.device(
            "cpu"
        )  # Default to CPU, will be updated in load_device
        self.model_name_noslash: Optional[str] = None
        self.model: Optional[HookedTransformer] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.dataloader: Optional[DataLoader] = None
        self.task_metric: Optional[Callable] = None

    def configure_logger(self):
        if self.debug:
            logger.remove()  # Remove any default handlers
            logger.add(sys.stderr, level="DEBUG")  # Enable DEBUG level
        else:
            logger.remove()  # Remove any default handlers
            logger.add(
                sys.stderr, level="INFO"
            )  # Disable DEBUG, only show INFO and higher


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


def load_dataset(config: Config):
    ds = EAPDataset(config)
    config.dataloader = ds.to_dataloader(config.batch_size)

    config.task_metric = get_metric(config.metric_name, config.task, model=config.model)
    config.kl_div = get_metric("kl_divergence", config.task, model=config.model)


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
        datapath=config_dict.get("dataset_path", None),
    )

    # Dynamically load the device
    config_obj.device = load_device()

    # Load the model with the updated config object
    load_model(config_obj)
    config_obj.configure_logger()

    if config_obj.task == "bias":
        prepare_bias_corrupt(config_obj)
    load_dataset(config_obj)

    return config_obj

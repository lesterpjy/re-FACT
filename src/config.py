from typing import Optional, List, Callable
import sys
from loguru import logger
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


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
        run: List[str],
        from_generated_graphs: bool,
        tiny_sample: Optional[int] = None,
        datapath: Optional[str] = None,
        process_data: Optional[bool] = False,
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
        self.run: List[str] = run
        self.datapath: Optional[str] = datapath
        self.process_data: Optional[bool] = process_data
        self.from_generated_graphs: bool = from_generated_graphs
        self.tiny_sample: Optional[int] = tiny_sample

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

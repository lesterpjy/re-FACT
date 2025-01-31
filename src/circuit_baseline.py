from typing import Callable, List, Union
import pandas as pd
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
from functools import partial
from .config_utils import load_config
from .config import Config


def evaluate_baseline(
    config: Config,
    model: HookedTransformer,
    dataloader: DataLoader,
    metrics: List[Callable[[Tensor], Tensor]],
    run_corrupted=False,
):
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False

    results = [[] for _ in metrics]

    for clean, corrupted, label in tqdm(dataloader):
        logger.debug(f"Evaluting baseline with run corrupted={run_corrupted}")
        logger.debug(f"clean: {clean}")
        logger.debug(f"corrupted: {corrupted}")
        logger.debug(f"label: {label}")
        tokenized = model.tokenizer(
            clean, padding="longest", return_tensors="pt", add_special_tokens=False
        )
        input_lengths = 1 + tokenized.attention_mask.sum(1)
        with torch.inference_mode():
            additional = (
                torch.tensor([25] * len(corrupted)).unsqueeze(1).to(config.device)
            )
            corrupted_logits = model(
                torch.cat((model.to_tokens(corrupted), additional), dim=1)
            )
            logger.debug(f"corrupted_logits: {corrupted_logits}")
            additional = torch.tensor([25] * len(clean)).unsqueeze(1).to(config.device)
            logits = model(torch.cat((model.to_tokens(clean), additional), dim=1))
            logger.debug(f"logits: {logits}")
        for i, metric in enumerate(metrics):
            if run_corrupted:
                r = metric(corrupted_logits, logits, input_lengths, label).cpu()
            else:
                r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results

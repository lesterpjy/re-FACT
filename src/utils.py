import pandas as pd
import torch
from torch.nn.functional import kl_div
from functools import partial
from transformers import PreTrainedTokenizer
from typing import Optional, List, Union, Literal, Tuple
from transformer_lens import HookedTransformer
from loguru import logger


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    logits = logits[idx, input_length - 1]
    return logits


def js_div(p: torch.tensor, q: torch.tensor):
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    return 0.5 * (
        kl_div(m, p.log(), log_target=True, reduction="none").mean(-1)
        + kl_div(m, q.log(), log_target=True, reduction="none").mean(-1)
    )


def divergence(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    divergence_type: Union[Literal["kl"], Literal["js"]] = "kl",
    mean=True,
    loss=True,
):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    probs = torch.softmax(logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    if divergence_type == "kl":
        results = kl_div(
            probs.log(), clean_probs.log(), log_target=True, reduction="none"
        ).mean(-1)
    elif divergence_type == "js":
        results = js_div(probs, clean_probs)
    else:
        raise ValueError(
            f"Expected divergence_type of 'kl' or 'js', but got '{divergence_type}'"
        )
    return results.mean() if mean else results


def logit_diff(
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean=True,
    prob=False,
    loss=False,
):
    logger.debug("DEBUG logit_diff:")
    logger.debug(f"clean_logits.shape: {clean_logits.shape}")
    logger.debug(f"corrupted_logits.shape: {corrupted_logits.shape}")
    logger.debug(f"input_length.shape: {input_length.shape}, values: {input_length}")
    logger.debug(f"labels.shape: {labels.shape}, values: {labels}")

    clean_logits = get_logit_positions(clean_logits, input_length)
    logger.debug(f"clean_logits after get_logit_positions: {clean_logits.shape}")

    if prob:
        logger.info("logit_diff: using softmax")

    cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
    logger.debug(f"cleans.shape: {cleans.shape}, prob: {prob}")
    logger.debug(
        f"labels.to(cleans.device).shape: {labels.to(cleans.device).shape}, values: {labels.to(cleans.device)}"
    )
    logger.debug(f"cleans.device: {cleans.device}, labels.device: {labels.device}")

    try:
        good_bad = torch.gather(cleans, -1, labels.to(cleans.device))
        logger.debug(f"cleans: {cleans}")
        logger.debug(f"labels: {labels}")
        logger.debug(f"labels.to(cleans.device): {labels.to(cleans.device)}")
        logger.debug(f"good_bad: {good_bad}")
    except RuntimeError as e:
        logger.error(f"Error in torch.gather: {e}")
        logger.error(
            f"Expected cleans.shape[-1]: {cleans.shape[-1]}, but labels values are: {labels}"
        )
        raise

    logger.info(f"good_bad.shape: {good_bad.shape}, values: {good_bad}")
    logger.info(
        f"good_bad[:, 0].shape: {good_bad[:, 0].shape}, values: {good_bad[:, 0]}"
    )
    logger.info(
        f"good_bad[:, 1].shape: {good_bad[:, 1].shape}, values: {good_bad[:, 1]}"
    )
    results = good_bad[:, 0] - good_bad[:, 1]
    logger.debug(f"results.shape: {results.shape}, values: {results}")

    if loss:
        # remember it's reversed to make it a loss
        results = -results
        logger.debug(f"results after loss inversion: {results}")

    if mean:
        logger.debug(f"results before mean: {results}")
        results = results.mean()
        logger.debug(f"results after mean: {results}")

    return results


def logit_diff_toxicity(
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: List[torch.Tensor],
    mean=True,
    prob=False,
    loss=False,
):
    clean_logits = get_logit_positions(clean_logits, input_length)
    cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits

    results = []
    for i, (ls, corrupted_ls) in enumerate(labels):
        r = (
            cleans[i][ls.to(cleans.device)].sum()
            - cleans[i][corrupted_ls.to(cleans.device)].sum()
        )
        results.append(r)
    results = torch.stack(results)

    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean:
        results = results.mean()
    return results


def get_metric(
    metric_name: str,
    task: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model: Optional[HookedTransformer] = None,
):
    if metric_name == "kl_divergence" or metric_name == "kl":
        return partial(divergence, divergence_type="kl")
    elif metric_name == "js_divergence" or metric_name == "js":
        return partial(divergence, divergence_type="js")
    elif metric_name == "logit_diff" or metric_name == "prob_diff":
        prob = metric_name == "prob_diff"
        if "toxicity" in task:
            logit_diff_fn = logit_diff_toxicity
        else:
            logit_diff_fn = logit_diff
        return partial(logit_diff_fn, prob=prob)
    elif metric_name == "logit_diff_ablate":
        prob = False
        logit_diff_fn = logit_diff
        return partial(logit_diff_fn, prob=prob)
    else:
        raise ValueError(f"got bad metric_name: {metric_name}")

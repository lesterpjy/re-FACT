from functools import partial
from typing import Callable, List, Union, Optional
from functools import partial
from pathlib import Path
from tqdm import tqdm
from loguru import logger

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from einops import einsum
from transformer_lens import HookedTransformer

from .graph import Graph, AttentionNode, LogitNode, InputNode
from .config_utils import load_config
from .config import Config

allowed_aggregations = {"sum", "mean", "l2"}


def get_npos_input_lengths(model, inputs):
    tokenized = model.tokenizer(
        inputs, padding="longest", return_tensors="pt", add_special_tokens=True
    )

    logger.debug("DEBUG get_npos_input_lengths:")
    logger.debug(f"  tokenized['input_ids'].shape = {tokenized['input_ids'].shape}")
    logger.debug(f"  attention_mask.shape          = {tokenized.attention_mask.shape}")
    logger.debug(f"  attention_mask sums           = {tokenized.attention_mask.sum(1)}")

    # Confirm what you actually assign to n_pos here:
    n_pos = 1 + tokenized.attention_mask.size(1)
    logger.debug(f"  => n_pos = {n_pos}")

    input_lengths = 1 + tokenized.attention_mask.sum(1)
    logger.debug(f"  => input_lengths = {input_lengths}")

    logger.debug("-" * 40)

    return n_pos, input_lengths


def make_hooks_and_matrices(
    config: Config,
    model: HookedTransformer,
    graph: Graph,
    batch_size: int,
    n_pos: int,
    scores,
):
    activation_difference = torch.zeros(
        (batch_size, n_pos, graph.n_forward, model.cfg.d_model),
        device=config.device,
        dtype=model.cfg.dtype,
    )
    # print("DEBUG make_hooks_and_matrices: Creating activation_difference")
    # print("  batch_size =", batch_size)
    # print("  n_pos      =", n_pos)
    # print("  -> activation_difference.shape (intial) =", activation_difference.shape)

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []

    def activation_hook(index, activations, hook, add: bool = True):
        # print(f"Entering activation_hook with index={index}", flush=True)
        # print(f"Activation shape: {activations.shape}", flush=True)
        # print(f"activation difference (in activation hook)): {activation_difference.shape}", flush=True)
        try:
            acts = activations.detach()
            # print(f"{hook.name} shapes: difference={activation_difference[:, :, index].shape}, acts={acts.shape}", flush=True)
        except Exception as e:
            print("Caught exception in activation_hook:", e, flush=True)

        if not add:
            acts = -acts
        try:
            activation_difference[:, :, index] += acts
        except Exception as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e

    def gradient_hook(
        fwd_index: Union[slice, int],
        bwd_index: Union[slice, int],
        gradients: torch.Tensor,
        hook,
    ):
        # print(f"Entering gradient_hook with fwd_index={fwd_index} and bwd_index={bwd_index}", flush=True)
        # print(f"activation difference (in gradient hook), {activation_difference.shape}", flush=True)
        # print(f"Gradients shape: {gradients.shape}", flush=True)
        # print(f"score shape: {scores.shape}", flush=True)
        try:
            grads = gradients.detach()
            # print(f"{hook.name} shapes: difference={activation_difference[:, :, :fwd_index].shape}, grads={grads.shape}", flush=True)
        except Exception as e:
            print("Caught exception in activation_hook:", e, flush=True)
        try:
            if isinstance(fwd_index, slice):
                fwd_index = fwd_index.start
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            s = einsum(
                activation_difference[:, :, :fwd_index],
                grads,
                "batch pos forward hidden, batch pos backward hidden -> forward backward",
            )
            s = s.squeeze(1)
            # print(f"score shape: {scores.shape}", flush=True)
            # print(f"indices: {fwd_index}, {bwd_index}", flush=True)
            # print(f"S shape: {s.shape}", flush=True)
            scores[:fwd_index, bwd_index] += s
        except RuntimeError as e:
            print("=" * 10)
            print(hook.name, activation_difference.size(), grads.size())
            print("=" * 10)
            raise e

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward
        fwd_index = graph.forward_index(node)
        # print("Registering forward hook on", node.out_hook, "for node:", name)
        # print("Fwd index:", fwd_index)
        if not isinstance(node, LogitNode):
            fwd_hooks_corrupted.append(
                (node.out_hook, partial(activation_hook, fwd_index))
            )
            fwd_hooks_clean.append(
                (node.out_hook, partial(activation_hook, fwd_index, add=False))
            )
        # print("Registering backward hook on", node.in_hook, "for node:", name)
        if not isinstance(node, InputNode):
            if isinstance(node, AttentionNode):
                for i, letter in enumerate("qkv"):
                    bwd_index = graph.backward_index(node, qkv=letter)
                    # print(f"doing {letter}")
                    # print("*"*20)
                    # print("Fwd index:", fwd_index)
                    # print("Bwd index:", bwd_index)
                    bwd_hooks.append(
                        (
                            node.qkv_inputs[i],
                            partial(gradient_hook, fwd_index, bwd_index),
                        )
                    )
            else:
                bwd_index = graph.backward_index(node)
                bwd_hooks.append(
                    (node.in_hook, partial(gradient_hook, fwd_index, bwd_index))
                )

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def strip_trailing_eos(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, eos_id: int = 128001
):
    """
    In-place removal of all trailing <|end_of_text|> tokens from each row.
    If a sequence has multiple EOS in a row, it removes them all.
    """
    batch_size, seq_len = input_ids.size()
    for i in range(batch_size):
        # Keep removing EOS tokens if present at the end
        while True:
            # Count how many tokens are actually present
            length = attention_mask[i].sum().item()  # number of non-pad tokens
            if length == 0:
                # Sequence is entirely empty/pad
                break

            last_idx = length - 1
            if input_ids[i, last_idx] == eos_id:
                # Zero out the final position
                input_ids[i, last_idx] = 0
                attention_mask[i, last_idx] = 0
            else:
                # The last token is not EOS; stop removing
                break


def get_scores(
    config: Config,
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    quiet=False,
):
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device=config.device, dtype=model.cfg.dtype
    )

    total_items = 0
    dataloader = dataloader if quiet else dataloader
    for clean, corrupted, label in tqdm(dataloader):

        # 1) Tokenize
        clean_tokenized = model.tokenizer(
            clean, padding="longest", return_tensors="pt", add_special_tokens=True
        )
        corrupted_tokenized = model.tokenizer(
            corrupted, padding="longest", return_tensors="pt", add_special_tokens=True
        )

        # 2) Log tokenization details
        logger.debug("DEBUG Tokenization:")
        logger.debug(f"  input_ids.shape = {clean_tokenized['input_ids'].shape}")
        logger.debug(
            f"  attention_mask.shape = {clean_tokenized['attention_mask'].shape}"
        )
        logger.debug(
            f"  attention_mask sums = {clean_tokenized['attention_mask'].sum(1)}"
        )

        # 3) Log the token IDs and their decoded versions
        for i in range(len(clean)):
            logger.debug(f"=== Example {i} ===")
            logger.debug(f"Raw text: {clean[i]}")
            logger.debug(f"Token IDs: {clean_tokenized['input_ids'][i].tolist()}")
            logger.debug(
                f"Decoded tokens: {model.tokenizer.decode(clean_tokenized['input_ids'][i])}"
            )

        batch_size = len(clean)
        total_items += batch_size
        logger.debug(f"DEBUG: batch_size = {batch_size}")

        n_pos, input_lengths = get_npos_input_lengths(model, clean)
        logger.debug(f"DEBUG: Just got n_pos = {n_pos} for clean input")
        logger.debug(f"DEBUG: input_lengths = {input_lengths.tolist()}")

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(config, model, graph, batch_size, n_pos, scores)
        )

        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            logger.debug(f"DEBUG corrupted sent: {corrupted}")
            logger.debug(f"DEBUG corrupted tokens: {model.to_tokens(corrupted)}")
            additional = (
                torch.tensor([25] * len(corrupted)).unsqueeze(1).to(config.device)
            )

            corrupted_logits = model(
                torch.cat((model.to_tokens(corrupted), additional), dim=1)
            )
            logger.debug(
                f"DEBUG corrupted model output shape: {corrupted_logits.shape}"
            )

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logger.debug(f"DEBUG clean sent: {clean}")
            logger.debug(f"DEBUG clean tokens: {model.to_tokens(clean)}")
            additional = torch.tensor([25] * len(clean)).unsqueeze(1).to(config.device)

            logits = model(torch.cat((model.to_tokens(clean), additional), dim=1))
            logger.debug(f"DEBUG clean model output shape: {logits.shape}")

            metric_value = metric(logits, corrupted_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores


def get_scores_ig(
    config: Config,
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    steps=30,
    quiet=False,
):
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device=config.device, dtype=model.cfg.dtype
    )

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        n_pos, input_lengths = get_npos_input_lengths(model, clean)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(config, model, graph, batch_size, n_pos, scores)
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                additional = (
                    torch.tensor([25] * len(corrupted)).unsqueeze(1).to(config.device)
                )
                _ = model(torch.cat((model.to_tokens(corrupted), additional), dim=1))

            input_activations_corrupted = activation_difference[
                :, :, graph.forward_index(graph.nodes["input"])
            ].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                additional = (
                    torch.tensor([25] * len(clean)).unsqueeze(1).to(config.device)
                )

                clean_logits = model(
                    torch.cat((model.to_tokens(clean), additional), dim=1)
                )

            input_activations_clean = (
                input_activations_corrupted
                - activation_difference[:, :, graph.forward_index(graph.nodes["input"])]
            )

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (
                    input_activations_clean - input_activations_corrupted
                )
                new_input.requires_grad = True
                return new_input

            return hook_fn

        total_steps = 0
        for step in range(1, steps + 1):
            total_steps += 1
            with model.hooks(
                fwd_hooks=[
                    (graph.nodes["input"].out_hook, input_interpolation_hook(step))
                ],
                bwd_hooks=bwd_hooks,
            ):
                logits = model(clean)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    scores /= total_steps

    return scores


def attribute(
    config: Config,
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    aggregation="sum",
    integrated_gradients: Optional[int] = None,
    quiet=False,
):
    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"aggregation must be in {allowed_aggregations}, but got {aggregation}"
        )

    if integrated_gradients is None:
        scores = get_scores(config, model, graph, dataloader, metric, quiet=quiet)
    else:
        assert (
            integrated_gradients > 0
        ), f"integrated_gradients gives positive # steps (m), but got {integrated_gradients}"
        scores = get_scores_ig(
            config,
            model,
            graph,
            dataloader,
            metric,
            steps=integrated_gradients,
            quiet=quiet,
        )

        if aggregation == "mean":
            scores /= model.cfg.d_model
        elif aggregation == "l2":
            scores = torch.linalg.vector_norm(scores, ord=2, dim=-1)

    scores = scores.cpu().numpy()

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        edge.score = scores[
            graph.forward_index(edge.parent, attn_slice=False),
            graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False),
        ]

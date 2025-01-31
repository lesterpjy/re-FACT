import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from functools import partial
import pandas as pd
from typing import Callable, List
from loguru import logger
import pickle

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from .config_utils import Config
from .graph import Graph, Node, InputNode, LogitNode, MLPNode, AttentionNode

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def evaluate_graph(
    config: Config,
    graph: Graph,
    dataloader: DataLoader,
    metrics: List[Callable[[Tensor], Tensor]],
    prune: bool = True,
    quiet=False,
):
    """
    Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune beforehand to make sure your circuit is valid.
    """
    model = config.model
    if prune:
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)

    empty_circuit = not graph.nodes["logits"].in_graph

    fwd_names = {edge.parent.out_hook for edge in graph.edges.values()}
    fwd_filter = lambda x: x in fwd_names

    corrupted_fwd_cache, corrupted_fwd_hooks, _ = model.get_caching_hooks(fwd_filter)
    mixed_fwd_cache, mixed_fwd_hooks, _ = model.get_caching_hooks(fwd_filter)

    nodes_in_graph = [
        node
        for node in graph.nodes.values()
        if node.in_graph
        if not isinstance(node, InputNode)
    ]

    # For each node in the graph, construct its input (in the case of attention heads, multiple inputs) by corrupting the incoming edges that are not in the circuit.
    # We assume that the corrupted cache is filled with corresponding corrupted activations, and that the mixed cache contains the computed activations from preceding nodes in this forward pass.
    def make_input_construction_hook(node: Node, qkv=None):
        def input_construction_hook(activations, hook):
            for edge in node.parent_edges:
                if edge.qkv != qkv:
                    continue

                parent: Node = edge.parent
                if not edge.in_graph:
                    activations[edge.index] -= mixed_fwd_cache[parent.out_hook][
                        parent.index
                    ]
                    activations[edge.index] += corrupted_fwd_cache[parent.out_hook][
                        parent.index
                    ]
            return activations

        return input_construction_hook

    input_construction_hooks = []
    for node in nodes_in_graph:
        if isinstance(node, InputNode):
            pass
        elif isinstance(node, LogitNode) or isinstance(node, MLPNode):
            input_construction_hooks.append(
                (node.in_hook, make_input_construction_hook(node))
            )
        elif isinstance(node, AttentionNode):
            for i, letter in enumerate("qkv"):
                input_construction_hooks.append(
                    (node.qkv_inputs[i], make_input_construction_hook(node, qkv=letter))
                )
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    # and here we actually run / evaluate the model
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    results = [[] for _ in metrics]

    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        tokenized = model.tokenizer(
            clean, padding="longest", return_tensors="pt", add_special_tokens=True
        )
        input_lengths = 1 + tokenized.attention_mask.sum(1)
        with torch.inference_mode():
            with model.hooks(corrupted_fwd_hooks):
                additional = (
                    torch.tensor([25] * len(corrupted)).unsqueeze(1).to(config.device)
                )
                corrupted_logits = model(
                    torch.cat((model.to_tokens(corrupted), additional), dim=1)
                )
                # corrupted_logits = model(corrupted)

            with model.hooks(mixed_fwd_hooks + input_construction_hooks):
                if empty_circuit:
                    # if the circuit is totally empty, so is nodes_in_graph
                    # so we just corrupt everything manually like this
                    additional = (
                        torch.tensor([25] * len(corrupted))
                        .unsqueeze(1)
                        .to(config.device)
                    )
                    logits = model(
                        torch.cat((model.to_tokens(corrupted), additional), dim=1)
                    )
                else:
                    additional = (
                        torch.tensor([25] * len(clean)).unsqueeze(1).to(config.device)
                    )
                    logits = model(
                        torch.cat((model.to_tokens(clean), additional), dim=1)
                    )
                    logger.debug("DEBUG model output shape:", logits.shape)

        for i, metric in enumerate(metrics):
            r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results


def load_graph_from_json(file_path):
    try:
        return Graph.from_json(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading graph from JSON: {e}")
        return None




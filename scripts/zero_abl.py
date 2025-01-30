import sys
import torch
import pandas as pd
import pickle
import numpy as np

from src.graph import Graph, Node, Edge, load_graph_from_json
from transformer_lens import HookedTransformer
from huggingface_hub import login
from loguru import logger

login(token="")


class Config:
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_name_noslash = "Llama-3.2-1B-Instruct"
    data_dir = "../data"
    work_dir = "../work"
    debug = False
    graph_name = "graph_0.json"


config = Config()

if config.debug:
    logger.remove()  # Remove any default handlers
    logger.add(sys.stderr, level="DEBUG")  # Enable DEBUG level
else:
    logger.remove()  # Remove any default handlers
    logger.add(sys.stderr, level="INFO")  # Disable DEBUG, only show INFO and higher


def find_bos_token(tokens_all: torch.Tensor, bos_token: int = 128009):
    """
    Finds the position of the BOS token in each sequence of a batch of tokenized sentences.

    Parameters:
    -----------
    tokens_all : Tensor
        A batch of tokenized sentences, shape (batch_size, seq_len).
    bos_token : int, optional
        The ID of the BOS token to find. Default is 128009.

    Returns:
    --------
    List[int]
        A list of positions for the BOS token in each sentence. If not found,
        the position is set to len(tokens) - 1.
    """
    # Check where the BOS token exists in each sentence
    bos_positions = (tokens_all == bos_token).nonzero(as_tuple=False)

    # Create a result array initialized with len(tokens) - 1 for each sentence
    result = [len(tokens) - 1 for tokens in tokens_all]

    # Update with BOS token positions
    for sentence_idx, token_idx in reversed(bos_positions):
        if token_idx == 0:
            continue
        result[sentence_idx] = token_idx.item() - 1

    return result


def filter_answer_logits(
    logits_all: torch.Tensor,
    tokens_all: torch.Tensor,
    needed_tokens: list[int] = [0, 1],
):
    """
    Filters the logits for the answer tokens in each sentence.

    Parameters:
    -----------
    logits_all : Tensor
        A batch of logits, shape (batch_size, seq_len, vocab_size).
    tokens_all : Tensor
        A batch of tokenized sentences, shape (batch_size, seq_len).
    needed_tokens : List[int]
        A list of token positions to extract logits from.

    Returns:
    --------
    Tensor
        A tensor of logits for the answer tokens in each sentence, shape (batch_size, len(needed_tokens)).
    """

    x = find_bos_token(tokens_all)

    x = torch.tensor(x, device=logits_all.device, dtype=torch.long)
    logits_answer = torch.stack(
        [logits[idx, needed_tokens] for logits, idx in zip(logits_all, x)]
    )

    return logits_answer


def compute_logit_diff_2(
    logits_all: torch.Tensor,
    tokens_all: torch.Tensor,
    correct_answers: list[int],
    needed_tokens: list[int] = [0, 1],
    average: bool = True,
):
    """
    Computes the difference between the logits of the correct and incorrect answers.

    Parameters:
    -----------
    logits_all : Tensor
        A batch of logits, shape (batch_size, seq_len, vocab_size).
    tokens_all : Tensor
        A batch of tokenized sentences, shape (batch_size, seq_len).
    correct_answers : List[int]
        A list of correct answer positions in the vocab.
    needed_tokens : List[int], optional
        A list of token positions to extract logits from. Default is [0, 1].
    average : bool, optional
        If True, returns the average logit difference. Default is True.

    Returns:
    --------
    Tensor
        A tensor of logit differences for the correct and incorrect answers in each sentence.
    """

    logits = filter_answer_logits(logits_all, tokens_all, needed_tokens)
    logit_diffs = (logits[:, 0] - logits[:, 1]) * torch.tensor(correct_answers).to(
        logits.device
    )
    return logit_diffs.mean() if average else logit_diffs


# TODO: change path
dfs = pd.read_csv(
    f"{config.data_dir}/circuit_identification_data/final_toxicity_prompts_0.csv"
)
logeer.info("Loaded data")
logger.info("loaded data lengths is: ", len(dfs))

batch_size = 2
threshold_logit = -0.63

logger.info("Loaded data")

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # default_prepend_bos = False
    # refactor_factored_attn_matrices=True
)

logger.info("Loaded model")

prompts = dfs["prompt_final"].tolist()
toxicity = dfs["toxicity"].to_list()

y_sampled = np.array(toxicity) > 0.5
answer_tokens = [1 for _ in y_sampled]  # always l(true) - l(false)
needed_tokens = [model.to_tokens(" true")[0][1], model.to_tokens(" false")[0][1]]

try:
    # TODO: change path
    file_path_g1 = (
        f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.graph_name}"
    )
    graph = load_graph_from_json(file_path_g1)
    logger.info(f"Loaded ablated graph from {file_path_g1}")
except FileNotFoundError:
    logger.error(
        f"Error: Graphs not found in {config.work_dir}/graphs/{config.model_name_noslash}/{config.graph_name}"
    )
    raise
except Exception as e:
    logger.error(f"Error: {e}")
    raise

logger.info("Loaded graph")

fwd_names = {edge.parent.out_hook for edge in graph.edges.values()}
fwd_filter = lambda x: x in fwd_names

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

            if edge.in_graph:

                activations[edge.index] -= mixed_fwd_cache[parent.out_hook][
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

logger.info("Constructed input hooks")

results = []

logger.info("Starting inference...")
for i in range(0, len(prompts), batch_size):

    clean = prompts[i : i + batch_size]
    tokens_org = model.to_tokens(clean)

    with torch.inference_mode():

        with model.hooks(mixed_fwd_hooks + input_construction_hooks):

            logits = model(tokens_org)

        r = compute_logit_diff_2(
            logits,
            tokens_org,
            answer_tokens[: len(clean)],
            needed_tokens=needed_tokens,
            average=False,
        )

        results.append(r)

with open(f"results_abl_{config.graph_name}.pkl", "wb") as f:
    pickle.dump(results, f)

logger.info("Finished inference")

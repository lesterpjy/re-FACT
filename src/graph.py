import cmapy
from typing import List, Dict, Union, Tuple, Literal, Optional, Set
from transformer_lens import HookedTransformer, HookedTransformerConfig
import heapq
import json
import numpy as np
import torch
import pygraphviz as pgv
from loguru import logger

EDGE_TYPE_COLORS = {
    "q": "#FF00FF",  # Purple
    "k": "#00FF00",  # Green
    "v": "#0000FF",  # Blue
    None: "#000000",  # Black
}


def generate_random_color(colorscheme: str) -> str:
    """
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """

    def rgb2hex(rgb):
        """
        https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
        """
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    return rgb2hex(cmapy.color(colorscheme, np.random.randint(0, 256), rgb_order=True))


def load_graph_from_json(file_path):
    try:
        # Assuming Graph class has a from_dict method
        return Graph.from_json(file_path)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading graph from JSON: {e}")
        return None


class Node:
    """
    A node in our computational graph. The in_hook is the TL hook into its inputs,
    while the out_hook gets its outputs.
    """

    name: str
    layer: int
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set["Node"]
    parent_edges: Set["Edge"]
    children: Set["Node"]
    child_edges: Set["Edge"]
    in_graph: bool
    qkv_inputs: Optional[List[str]]

    def __init__(
        self,
        name: str,
        layer: int,
        in_hook: List[str],
        out_hook: str,
        index: Tuple,
        qkv_inputs: Optional[List[str]] = None,
    ):
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook
        self.index = index
        self.in_graph = True
        self.parents = set()
        self.children = set()
        self.parent_edges = set()
        self.child_edges = set()
        self.qkv_inputs = qkv_inputs

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Node({self.name}, in_graph: {self.in_graph})"

    def __hash__(self):
        return hash(self.name)


class LogitNode(Node):
    def __init__(self, n_layers: int):
        name = "logits"
        index = slice(None)
        super().__init__(
            name, n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", "", index
        )


class MLPNode(Node):
    def __init__(self, layer: int):
        name = f"m{layer}"
        index = slice(None)
        super().__init__(
            name,
            layer,
            f"blocks.{layer}.hook_mlp_in",
            f"blocks.{layer}.hook_mlp_out",
            index,
        )


class AttentionNode(Node):
    head: int

    def __init__(self, layer: int, head: int):
        name = f"a{layer}.h{head}"
        self.head = head
        index = (slice(None), slice(None), head)
        super().__init__(
            name,
            layer,
            f"blocks.{layer}.hook_attn_in",
            f"blocks.{layer}.attn.hook_result",
            index,
            [f"blocks.{layer}.hook_{letter}_input" for letter in "qkv"],
        )


class InputNode(Node):
    def __init__(self):
        name = "input"
        index = slice(None)
        super().__init__(
            name, 0, "", "hook_embed", index
        )  # "blocks.0.hook_resid_pre", index)


class Edge:
    name: str
    parent: Node
    child: Node
    hook: str
    index: Tuple
    score: Optional[float]
    in_graph: bool

    def __init__(
        self,
        parent: Node,
        child: Node,
        qkv: Union[None, Literal["q"], Literal["k"], Literal["v"]] = None,
    ):
        self.name = (
            f"{parent.name}->{child.name}"
            if qkv is None
            else f"{parent.name}->{child.name}<{qkv}>"
        )
        self.parent = parent
        self.child = child
        self.qkv = qkv
        self.score = None
        self.in_graph = True
        if isinstance(child, AttentionNode):
            if qkv is None:
                raise ValueError(
                    f"Edge({self.name}): Edges to attention heads must have a non-none value for qkv."
                )
            self.hook = f"blocks.{child.layer}.hook_{qkv}_input"
            self.index = (slice(None), slice(None), child.head)
        else:
            self.index = child.index
            self.hook = child.in_hook

    def get_color(self):
        if self.qkv is not None:
            return EDGE_TYPE_COLORS[self.qkv]
        elif self.score < 0:
            return "#FF0000"
        else:
            return "#000000"

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Edge({self.name}, score: {self.score}, in_graph: {self.in_graph})"

    def __hash__(self):
        return hash(self.name)


class Graph:
    nodes: Dict[str, Node]
    edges: Dict[str, Edge]
    n_forward: int
    n_backward: int
    cfg: HookedTransformerConfig

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.n_forward = 0
        self.n_backward = 0

    def add_edge(
        self,
        parent: Node,
        child: Node,
        qkv: Union[None, Literal["q"], Literal["k"], Literal["v"]] = None,
    ):
        edge = Edge(parent, child, qkv)
        self.edges[edge.name] = edge
        parent.children.add(child)
        parent.child_edges.add(edge)
        child.parents.add(parent)
        child.parent_edges.add(edge)

    def forward_index(self, node: Node, attn_slice=True):
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
        elif isinstance(node, MLPNode):
            return 1 + node.layer * (self.cfg["n_heads"] + 1) + self.cfg["n_heads"]
        elif isinstance(node, AttentionNode):
            # Forward indexing remains the same. We assume one forward slot
            # per attention head, so we use n_heads.
            i = 1 + node.layer * (self.cfg["n_heads"] + 1)
            return slice(i, i + self.cfg["n_heads"]) if attn_slice else i + node.head
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def backward_index(self, node: Node, qkv=None, attn_slice=True):
        """
        Modified so that Q uses n_heads=32, while K and V each use n_key_value_heads=8.
        """
        if isinstance(node, InputNode):
            raise ValueError("No backward for input node")

        elif isinstance(node, LogitNode):
            return -1

        elif isinstance(node, MLPNode):
            # --- CHANGE: MLP node is offset after Q, K, V in each layer ---
            # We define a layer stride as: Q( n_heads ) + K( n_kv ) + V( n_kv ) + 1 (for MLP).
            layer_stride = (
                self.cfg["n_heads"]
                + 2 * self.cfg["n_key_value_heads"]
                + 1  # Q + K + V + MLP
            )
            i = node.layer * layer_stride
            offset_for_mlp = self.cfg["n_heads"] + 2 * self.cfg["n_key_value_heads"]
            return i + offset_for_mlp

        elif isinstance(node, AttentionNode):
            assert qkv in "qkv", f"Must give qkv for AttentionNode, got {qkv}"

            # --- CHANGE: separate Q from K/V ---
            # Q uses n_heads (e.g. 32), K/V each use n_key_value_heads (e.g. 8).
            n_h = self.cfg["n_heads"]
            n_kv = self.cfg["n_key_value_heads"]

            # Each layer has Q( n_h ) + K( n_kv ) + V( n_kv ) + 1( MLP ) = total
            layer_stride = n_h + 2 * n_kv + 1

            # Starting index for this layer
            i = node.layer * layer_stride

            if qkv == "q":
                offset = 0
                size = n_h
            elif qkv == "k":
                offset = n_h
                size = n_kv
            else:  # qkv == 'v'
                offset = n_h + n_kv
                size = n_kv

            i += offset
            return slice(i, i + size) if attn_slice else i + node.head

        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def scores(self, nonzero=False, in_graph=False, sort=True):
        s = (
            torch.tensor(
                [
                    edge.score
                    for edge in self.edges.values()
                    if edge.score != 0 and (edge.in_graph or not in_graph)
                ]
            )
            if nonzero
            else torch.tensor([edge.score for edge in self.edges.values()])
        )
        return torch.sort(s).values if sort else s

    def count_included_edges(self):
        return sum(edge.in_graph for edge in self.edges.values())

    def count_included_nodes(self):
        return sum(node.in_graph for node in self.nodes.values())

    def apply_threshold(self, threshold: float, absolute: bool):
        threshold = float(threshold)
        for node in self.nodes.values():
            node.in_graph = True

        for edge in self.edges.values():
            edge.in_graph = (
                abs(edge.score) >= threshold if absolute else edge.score >= threshold
            )

    def apply_topn(self, n: int, absolute: bool):
        a = abs if absolute else lambda x: x
        for node in self.nodes.values():
            node.in_graph = False

        sorted_edges = sorted(
            list(self.edges.values()), key=lambda edge: a(edge.score), reverse=True
        )
        for edge in sorted_edges[:n]:
            edge.in_graph = True
            edge.parent.in_graph = True
            edge.child.in_graph = True

        for edge in sorted_edges[n:]:
            edge.in_graph = False

    def apply_greedy(self, n_edges, reset=True, absolute: bool = True):
        """
        Greedy approach to pick edges with the largest absolute score until we have n_edges edges activated.
        Uses a single max-heap (negative scores) and a unique ID to avoid Edge < Edge comparison.
        """
        if reset:
            for node in self.nodes.values():
                node.in_graph = False
            for edge in self.edges.values():
                edge.in_graph = False
            self.nodes["logits"].in_graph = True

        def abs_id(s: float):
            return abs(s) if absolute else s

        # Build initial list of candidate edges (child in_graph == True).
        candidate_edges = [e for e in self.edges.values() if e.child.in_graph]

        # Push items as (-score, unique_id, Edge) so we never compare Edge objects directly.
        edges_heap = []
        for e in candidate_edges:
            heapq.heappush(edges_heap, (-abs_id(e.score), id(e), e))

        while n_edges > 0 and edges_heap:
            top_neg_score, _, top_edge = heapq.heappop(edges_heap)
            if top_edge.in_graph:
                # Already activated (could be a duplicate in heap)
                continue
            top_edge.in_graph = True

            parent = top_edge.parent
            if not parent.in_graph:
                parent.in_graph = True
                # Add all of parent's *parent_edges* to the heap
                for parent_edge in parent.parent_edges:
                    if not parent_edge.in_graph:
                        heapq.heappush(
                            edges_heap,
                            (-abs_id(parent_edge.score), id(parent_edge), parent_edge),
                        )

            n_edges -= 1

    def prune_dead_nodes(self, prune_childless=True, prune_parentless=True):
        self.nodes["logits"].in_graph = any(
            parent_edge.in_graph for parent_edge in self.nodes["logits"].parent_edges
        )

        for node in reversed(self.nodes.values()):
            if isinstance(node, LogitNode):
                continue

            if any(child_edge.in_graph for child_edge in node.child_edges):
                node.in_graph = True
            else:
                if prune_childless:
                    node.in_graph = False
                    for parent_edge in node.parent_edges:
                        parent_edge.in_graph = False
                else:
                    if any(child_edge.in_graph for child_edge in node.child_edges):
                        node.in_graph = True
                    else:
                        node.in_graph = False

        if prune_parentless:
            for node in self.nodes.values():
                if (
                    not isinstance(node, InputNode)
                    and node.in_graph
                    and not any(
                        parent_edge.in_graph for parent_edge in node.parent_edges
                    )
                ):
                    node.in_graph = False
                    for child_edge in node.child_edges:
                        child_edge.in_graph = False

    @classmethod
    def from_model(
        cls, model_or_config: Union[HookedTransformer, HookedTransformerConfig, Dict]
    ):
        graph = Graph()
        if isinstance(model_or_config, HookedTransformer):
            cfg = model_or_config.cfg
            graph.cfg = {
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "n_key_value_heads": cfg.n_key_value_heads,  # <-- store n_key_value_heads
                "parallel_attn_mlp": cfg.parallel_attn_mlp,
            }
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            graph.cfg = {
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "n_key_value_heads": cfg.n_key_value_heads,  # <-- store n_key_value_heads
                "parallel_attn_mlp": cfg.parallel_attn_mlp,
            }
        else:
            graph.cfg = model_or_config

        input_node = InputNode()
        graph.nodes[input_node.name] = input_node
        residual_stream = [input_node]

        n_heads = graph.cfg["n_heads"]  # e.g. 32
        n_kv = graph.cfg["n_key_value_heads"]  # e.g. 8

        for layer in range(graph.cfg["n_layers"]):

            # We'll still create 32 attention nodes (one for each "head" 0..31).
            # But we won't connect them to "k" or "v" edges if head > 7.
            attn_nodes = [AttentionNode(layer, head) for head in range(n_heads)]
            mlp_node = MLPNode(layer)

            for attn_node in attn_nodes:
                graph.nodes[attn_node.name] = attn_node
            graph.nodes[mlp_node.name] = mlp_node

            if graph.cfg["parallel_attn_mlp"]:
                for node in residual_stream:
                    for attn_node in attn_nodes:
                        # For Q, connect all heads [0..31].
                        # For K, V, connect only if head < n_kv (i.e. 8).
                        head_idx = attn_node.head
                        # Loop over 'qkv'
                        for letter in "qkv":
                            # Skip hooking if letter in {k, v} but head_idx >= n_kv
                            if letter in ["k", "v"] and head_idx >= n_kv:
                                continue
                            graph.add_edge(node, attn_node, qkv=letter)
                    graph.add_edge(node, mlp_node)

                residual_stream += attn_nodes
                residual_stream.append(mlp_node)

            else:
                for node in residual_stream:
                    for attn_node in attn_nodes:
                        head_idx = attn_node.head
                        for letter in "qkv":
                            # Skip hooking if letter in {k, v} but head_idx >= n_kv
                            if letter in ["k", "v"] and head_idx >= n_kv:
                                continue
                            graph.add_edge(node, attn_node, qkv=letter)
                residual_stream += attn_nodes

                for node in residual_stream:
                    graph.add_edge(node, mlp_node)
                residual_stream.append(mlp_node)

        logit_node = LogitNode(graph.cfg["n_layers"])
        for node in residual_stream:
            graph.add_edge(node, logit_node)

        graph.nodes[logit_node.name] = logit_node

        # Finally set forward/backward size. The logic is now:
        # forward: 1 + n_layers * (n_heads + 1)
        # backward per layer: Q(32) + K(8) + V(8) + MLP(1) = 32 + 8 + 8 + 1 = 49, plus final logits = +1
        layer_stride = n_heads + 2 * n_kv + 1
        graph.n_forward = 1 + graph.cfg["n_layers"] * (n_heads + 1)
        graph.n_backward = graph.cfg["n_layers"] * layer_stride + 1

        return graph

    def to_json(self, filename):
        # non serializable info
        d = {
            "cfg": self.cfg,
            "nodes": {
                str(name): bool(node.in_graph) for name, node in self.nodes.items()
            },
            "edges": {
                str(name): {
                    "score": None if edge.score is None else float(edge.score),
                    "in_graph": bool(edge.in_graph),
                }
                for name, edge in self.edges.items()
            },
        }
        with open(filename, "w") as f:
            json.dump(d, f)

    @classmethod
    def from_json(cls, filename):
        with open(filename, "r") as f:
            d = json.load(f)
        g = Graph.from_model(d["cfg"])
        for name, in_graph in d["nodes"].items():
            g.nodes[name].in_graph = in_graph

        for name, info in d["edges"].items():
            g.edges[name].score = info["score"]
            g.edges[name].in_graph = info["in_graph"]

        return g

    def __eq__(self, other):
        keys_equal = (set(self.nodes.keys()) == set(other.nodes.keys())) and (
            set(self.edges.keys()) == set(other.edges.keys())
        )
        if not keys_equal:
            return False

        for name, node in self.nodes.items():
            if node.in_graph != other.nodes[name].in_graph:
                return False

        for name, edge in self.edges.items():
            if (edge.in_graph != other.edges[name].in_graph) or not np.allclose(
                edge.score, other.edges[name].score
            ):
                return False
        return True

    def to_graphviz(
        self,
        colorscheme: str = "Pastel2",
        minimum_penwidth: float = 0.6,
        maximum_penwidth: float = 5.0,
        layout: str = "dot",
        seed: Optional[int] = None,
    ) -> pgv.AGraph:
        """
        Colorscheme: a cmap colorscheme
        """
        g = pgv.AGraph(
            directed=True,
            bgcolor="white",
            overlap="false",
            splines="true",
            layout=layout,
        )

        if seed is not None:
            np.random.seed(seed)

        colors = {
            node.name: generate_random_color(colorscheme)
            for node in self.nodes.values()
        }

        for node in self.nodes.values():
            if node.in_graph:
                g.add_node(
                    node.name,
                    fillcolor=colors[node.name],
                    color="black",
                    style="filled, rounded",
                    shape="box",
                    fontname="Helvetica",
                )

        scores = self.scores().abs()
        max_score = scores.max().item()
        min_score = scores.min().item()
        for edge in self.edges.values():
            if edge.in_graph:
                score = 0 if edge.score is None else edge.score
                normalized_score = (
                    (abs(score) - min_score) / (max_score - min_score)
                    if max_score != min_score
                    else abs(score)
                )
                penwidth = max(minimum_penwidth, normalized_score * maximum_penwidth)
                g.add_edge(
                    edge.parent.name,
                    edge.child.name,
                    penwidth=str(penwidth),
                    color=edge.get_color(),
                )
        return g

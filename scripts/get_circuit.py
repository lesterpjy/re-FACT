import sys
import os
import time
import pandas as pd
from pathlib import Path
import numpy as np
import tqdm
from functools import partial
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_utils import load_config
from src.circuit_baseline import evaluate_baseline
from src.circuit_eap import attribute
from src.circuit_eval import evaluate_graph
from src.graph import Graph, load_graph_from_json


start = time.time()
cwd = os.getcwd()
logger.info(f"Current working directory: {cwd}")
config = load_config("configs/llama3_config.py", process_data=True)
logger.info("Config loaded.")

# run baseline
# =============================================================================
if "baseline" in config.run:
    baseline = (
        evaluate_baseline(
            config.model,
            config.dataloader,
            partial(config.task_metric, mean=False, loss=False),
        )
        .mean()
        .item()
    )

    corrupted_baseline = (
        evaluate_baseline(
            config.model,
            config.dataloader,
            partial(config.task_metric, mean=False, loss=False),
            run_corrupted=True,
        )
        .mean()
        .item()
    )

    # Create a dictionary to store the baseline values
    data = {"baseline": [baseline], "corrupted_baseline": [corrupted_baseline]}

    # Create a pandas DataFrame from the dictionary
    baseline_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    baseline_df.to_csv(f"{config.data_dir}/baseline_values.csv", index=False)
    logger.info("Baseline values saved to CSV file.")

# =============================================================================
# Instantiate a graph with a model
if "graph" in config.run:
    logger.info("instantiate graph with model (vanilla)")
    g1 = Graph.from_model(config.model)
    # Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
    attribute(
        config.model,
        g1,
        config.dataloader,
        partial(config.task_metric, mean=True, loss=True),
    )

    Path(f"{config.work_dir}/graphs/{config.model_name_noslash}").mkdir(
        exist_ok=True, parents=True
    )
    g1.to_json(
        f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.task}_vanilla.json"
    )
    logger.info("Graph 1 saved to JSON file.")

    # Instantiate a graph with a model
    logger.info("instantiate graph with model (ig)")
    g2 = Graph.from_model(config.model)
    # Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
    attribute(
        config.model,
        g2,
        config.dataloader,
        partial(config.task_metric, mean=True, loss=True),
        integrated_gradients=5,
    )
    Path(f"{config.work_dir}/graphs/{config.model_name_noslash}").mkdir(
        exist_ok=True, parents=True
    )
    g2.to_json(
        f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.task}_task.json"
    )
    logger.info("Graph 2 saved to JSON file.")

# # =============================================================================


if "evaluate" in config.run:
    if "baseline" not in config.run:
        # Read the CSV file to get the baseline values
        try:
            df_read = pd.read_csv(f"{config.data_dir}/baseline_values.csv")
            baseline = df_read["baseline"][0]
            corrupted_baseline = df_read["corrupted_baseline"][0]
            logger.info(f"Baseline (read from file): {baseline}")
            logger.info(f"Corrupted Baseline (read from file): {corrupted_baseline}")
        except FileNotFoundError:
            logger.error(
                f"Error: Baseline values not found in {config.data_dir}/baseline_values.csv"
            )
            raise
    if "graph" not in config.run:
        try:
            file_path_g1 = f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.task}_vanilla.json"
            file_path_g2 = f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.task}_task.json"
            g1 = load_graph_from_json(file_path_g1)
            g2 = load_graph_from_json(file_path_g2)
        except FileNotFoundError:
            logger.error(
                f"Error: Graphs not found in {config.work_dir}/graphs/{config.model_name_noslash}"
            )
            raise
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
        logger.info("Graphs g1 and g2 loaded successfully.")

    logger.info("evaluate graph with metrics")
    gs = [g1, g2]
    n_edges = []
    results = []
    s = 100
    e = 1001
    step = 100
    first_steps = list(range(30, 100, 10))
    later_steps = list(range(s, e, step))
    steps = first_steps + later_steps
    labels = ["EAP", "EAP-IG"]

    logger.info("begin evaluation")
    with tqdm(total=len(gs) * len(steps)) as pbar:
        for i in steps:
            n_edge = []
            result = []
            for graph, label in zip(gs, labels):
                try:
                    pruned_graph_path = f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.task}_{label}_step{i}_pruned.json"
                    graph = load_graph_from_json(pruned_graph_path)
                    logger.info(f"Loaded pruned graph from {pruned_graph_path}")
                except:
                    graph.apply_greedy(i, absolute=True)
                    graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
                    logger.info(f"Applied greedy algorithm with {i} edges")
                    graph.to_json(
                        f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.task}_{label}_step{i}_{n}edges.json"
                    )
                    grapviz_graph = graph.to_graphviz(seed=42)
                    grapviz_graph.write(
                        f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.task}_{label}_step{i}_{n}edges.dot"
                    )
                    logger.info(f"Saved graph to JSON and DOT files")

                n = graph.count_included_edges()
                r = evaluate_graph(
                    config,
                    graph,
                    config.dataloader,
                    partial(config.task_metric, mean=False, loss=False),
                    quiet=False,
                )
                del grapviz_graph
                n_edge.append(n)
                result.append(r.mean().item())
                pbar.update(1)
            n_edges.append(n_edge)
            results.append(result)

    logger.info("done evaluation")
    n_edges = np.array(n_edges)
    results = np.array(results)

    logger.info("saving to csv")
    d = {
        "baseline": [baseline] * len(steps),
        "corrupted_baseline": [corrupted_baseline] * len(steps),
        "edges": steps,
    }

    for i, label in enumerate(labels):
        d[f"edges_{label}"] = n_edges[:, i].tolist()
        d[f"loss_{label}"] = results[:, i].tolist()
    df = pd.DataFrame.from_dict(d)

    Path(f"{config.work_dir}/results/pareto/{config.model_name_noslash}/csv").mkdir(
        exist_ok=True, parents=True
    )
    df.to_csv(
        f"{config.work_dir}/results/pareto/{config.model_name_noslash}/csv/{config.task}.csv",
        index=False,
    )

end = time.time()
logger.info(f"Time taken: {end - start} seconds")
logger.info("done")

import pandas as pd
from pathlib import Path
import numpy as np
import tqdm
from functools import partial
from loguru import logger

from src.config_utils import load_config
from src.circuit_baseline import evaluate_baseline
from src.circuit_eap import attribute
from src.circuit_eval import evaluate_graph
from src.graph import Graph, load_graph_from_json


config = load_config("configs/llama3_config.py")
logger.info("Config loaded.")

# run baseline
# =============================================================================
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
# # Replace with the actual paths to your JSON files
# file_path_g1 = (
#     f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.task}_vanilla.json"
# )
# file_path_g2 = (
#     f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.task}_task.json"
# )

# g1 = load_graph_from_json(file_path_g1)
# g2 = load_graph_from_json(file_path_g2)

# if g1 and g2:
#     print("Graphs g1 and g2 loaded successfully.")
# else:
#     print("Failed to load one or both graphs.")


logger.info("evaluate graph with metrics")
gs = [g1, g2]
n_edges = []
results = []
s = 100
e = 2001
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
            graph.apply_greedy(i, absolute=True)
            graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)

            n = graph.count_included_edges()
            r = evaluate_graph(
                config,
                graph,
                config.dataloader,
                partial(config.task_metric, mean=False, loss=False),
                quiet=False,
            )
            graph.to_json(
                f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.task}_{label}_step{i}_{n}edges.json"
            )
            grapviz_graph = graph.to_graphviz(seed=42)
            grapviz_graph.write(
                f"{config.work_dir}/graphs/{config.model_name_noslash}/{config.task}_{label}_step{i}_{n}edges.dot"
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

# # Read the CSV file back into a DataFrame
# df_read = pd.read_csv(f"{config.data_dir}/baseline_values.csv")
# # Access the baseline and corrupted_baseline values
# baseline = df_read["baseline"][0]
# corrupted_baseline = df_read["corrupted_baseline"][0]
# print(f"Baseline (read from file): {baseline}")
# print(f"Corrupted Baseline (read from file): {corrupted_baseline}")


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

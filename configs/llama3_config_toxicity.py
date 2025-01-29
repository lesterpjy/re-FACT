CONFIG = {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "random_seed": 42,
    "data_dir": "../data",
    "work_dir": "../work",
    "debug": False,
    "labels": ["EAP", "EAP-IG"],
    "task": "toxicity",
    "data_split": 0,
    "metric_name": "logit_diff",
    "batch_size": 2,
    "from_generated_graphs": False,
    "process_data": False,
    "run": ["evaluate"],
}

# You can also set a default device here if desired
device = None  # Device will be loaded dynamically in the load_config function

CONFIG = {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "random_seed": 42,
    "data_dir": "./data",
    "work_dir": "./work",
    "debug": True,
    "labels": ["EAP", "EAP-IG"],
    "task": "gender-bias",
    "task_metric": "logit_diff",
    "batch_size": 2,
}


# You can also set a default device here if desired
device = None  # Device will be loaded dynamically in the load_config function

from typing import Optional
from pathlib import Path
import pandas as pd
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from .config import Config


def prepare_bias_corrupt(config: Config):
    logger.info("Process bias corruption dataset...")
    df = pd.read_csv(
        f"{config.data_dir}/circuit_identification_data/final_toxicity_prompts_{config.data_split}.csv"
    )
    true_idx = config.tokenizer(" true").input_ids[1]
    false_idx = config.tokenizer(" false").input_ids[1]

    df["toxic_label"] = df["toxicity"].apply(
        lambda x: int(x >= 0.5)
    )  # true label of the prompt, bool 0 or 1
    df["clean_answer_idx"] = df["toxic_label"].apply(
        lambda x: true_idx if x == 1 else false_idx
    )
    df["corrupted_answer_idx"] = df["toxic_label"].apply(
        lambda x: false_idx if x == 1 else true_idx
    )

    eapdf = pd.DataFrame(
        {
            "clean": df["prompt_final"],
            "corrupted": df["corrupted_prompt"],
            "clean_answer_idx": df["clean_answer_idx"],
            "corrupted_answer_idx": df["corrupted_answer_idx"],
            "label": df["toxic_label"],
        }
    )
    if config.tiny_sample:
        eapdf = eapdf.sample(30)
    Path(f"{config.data_dir}/circuit_identification_data/{config.task}").mkdir(
        exist_ok=True, parents=True
    )    
    eapdf.to_csv(
        f"{config.data_dir}/circuit_identification_data/{config.task}/corrupt_{config.task}_eap_{config.data_split}.csv",
        index=False,
    )
    logger.info(
        f"Bias corruption dataset save to {config.data_dir}/circuit_identification_data/{config.task}/corrupt_{config.task}_eap_{config.data_split}.csv"
    )


def collate_EAP(xs, task):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    if "toxicity" not in task:
        labels = torch.tensor(labels)
    return clean, corrupted, labels


def model2family(model_name: str):
    if "gpt2" in model_name:
        return "gpt2"
    elif "pythia" in model_name:
        return "pythia"
    elif "Llama" in model_name:
        return "llama"
    else:
        raise ValueError(f"Couldn't find model family for model: {model_name}")


class EAPDataset(Dataset):
    def __init__(self, config: Config):
        logger.info(f"data path {config.datapath}")
        if config.datapath is None:
            self.df = pd.read_csv(
                f"{config.data_dir}/circuit_identification_data/{config.task}/corrupt_{config.task}_eap_{config.data_split}.csv"
            )
            print(
                "loaded dataset from",
                f"{config.data_dir}/circuit_identification_data/{config.task}/corrupt_{config.task}_eap_{config.data_split}.csv",
            )
        else:
            self.df = pd.read_csv(config.datapath)
            print(f"loaded dataset from, {config.datapath}")

        self.task = config.task

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = None
        if self.task == "ioi":
            label = [row["correct_idx"], row["incorrect_idx"]]
        elif "greater-than" in self.task:
            label = row["correct_idx"]
        elif "toxicity" in self.task:
            answer = torch.tensor(eval(row["answers_idx"]))
            corrupted_answer = torch.tensor(eval(row["corrupted_answers_idx"]))
            label = [answer, corrupted_answer]
        elif "fact-retrieval" in self.task:
            label = [row["country_idx"], row["corrupted_country_idx"]]
        elif "bias" in self.task:
            label = [row["clean_answer_idx"], row["corrupted_answer_idx"]]
        elif self.task == "sva":
            label = row["plural"]
        elif self.task == "colored-objects":
            label = [row["correct_idx"], row["incorrect_idx"]]
        elif self.task in {"dummy-easy", "dummy-medium", "dummy-hard"}:
            label = 0
        else:
            raise ValueError(f"Got invalid task: {self.task}")
        return row["clean"], row["corrupted"], label

    def to_dataloader(self, batch_size: int):
        return DataLoader(
            self, batch_size=batch_size, collate_fn=partial(collate_EAP, task=self.task)
        )

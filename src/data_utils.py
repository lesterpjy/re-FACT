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


def prepare_ablate(config: Config):
    logger.info("Process bias corruption dataset...")
    df = pd.read_csv(
        f"{config.data_dir}/circuit_identification_data/final_toxicity_prompts_{config.data_split}.csv"
    )
    true_idx = config.tokenizer(" true").input_ids[1]
    false_idx = config.tokenizer(" false").input_ids[1]

    df["toxic_label"] = df["toxicity"].apply(
        lambda x: int(x >= 0.5)
    )  # true label of the prompt, bool 0 or 1
    df["clean_answer_idx"] = df["toxic_label"].apply(lambda x: true_idx)
    df["corrupted_answer_idx"] = df["toxic_label"].apply(lambda x: false_idx)
    eapdf = pd.DataFrame(
        {
            "clean": df["prompt_final"],
            "corrupted": df["corrupted_prompt"],
            "clean_answer_idx": df["clean_answer_idx"],
            "corrupted_answer_idx": df["corrupted_answer_idx"],
            "label": df["toxic_label"],
        }
    )

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


def prepare_toxicity_corrupt(config: Config):
    logger.info("Process bias corruption dataset...")
    df = pd.read_csv(
        f"{config.data_dir}/circuit_identification_data/final_toxicity_prompts_{config.data_split}.csv"
    )

    df["correct"] = df["toxicity_group"].apply(lambda x: int(x >= 0.5))
    df["answers"] = df["correct"].apply(lambda x: [" true"] if x == 1 else [" false"])
    df["answers_idx"] = df["answers"].apply(
        lambda x: [config.model.to_tokens(i)[0][1].item() for i in x]
    )
    df["corrupted_correct"] = df["corrupted_toxicity"].apply(lambda x: int(x >= 0.5))
    df["corrupted_answers"] = df["corrupted_correct"].apply(
        lambda x: [" true"] if x == 1 else [" false"]
    )
    df["corrupted_answers_idx"] = df["corrupted_answers"].apply(
        lambda x: [config.model.to_tokens(i)[0][1].item() for i in x]
    )
    df = df[
        [
            "prompt_final",
            "correct",
            "answers",
            "answers_idx",
            "corrupted_sentence_prompt",
            "corrupted_correct",
            "corrupted_answers",
            "corrupted_answers_idx",
        ]
    ]
    df = df.rename(
        {"prompt_final": "clean", "corrupted_sentence_prompt": "corrupted"}, axis=1
    )
    Path(f"{config.data_dir}/circuit_identification_data/{config.task}").mkdir(
        exist_ok=True, parents=True
    )
    df.to_csv(
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
            logger.info(
                "loaded dataset from",
                f"{config.data_dir}/circuit_identification_data/{config.task}/corrupt_{config.task}_eap_{config.data_split}.csv",
            )
            # testing with removing last character
            self.df["clean"] = self.df["clean"].str[:-1]
            self.df["corrupted"] = self.df["corrupted"].str[:-1]
            if config.tiny_sample:
                self.df = self.df.sample(config.tiny_sample)
                self.df.to_csv(
                    f"{config.data_dir}/circuit_identification_data/{config.task}/corrupt_{config.task}_eap_{config.data_split}_{config.tiny_sample}samples.csv"
                )
                logger.info(
                    f"saved sampled dataset to {config.data_dir}/circuit_identification_data/{config.task}/corrupt_{config.task}_eap_{config.data_split}_{config.tiny_sample}samples.csv"
                )
                logger.info(f"loaded tiny sample of size {config.tiny_sample}")
        else:
            self.df = pd.read_csv(config.datapath)
            print(f"loaded dataset from, {config.datapath}")

        logger.info(f"Dataset size: {len(self.df)}")
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
            logger.info(f"task: {self.task}")
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

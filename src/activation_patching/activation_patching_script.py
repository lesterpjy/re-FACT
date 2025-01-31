import pandas as pd
from collections import defaultdict
from random import sample
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import random
import pickle
import string
from itertools import product
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import torch

from transformer_lens import HookedTransformer
from transformer_lens import utils, patching

from IPython.display import clear_output
from custom_plotly_utils import imshow, line, scatter
from huggingface_hub import login
import argparse

def get_all_logits(sampled_prompts: list[str], model, run_with_cache: bool = True):
    """
    Function to get logits for a list of prompts
    :param sampled_prompts: list of prompts
    :param model: model
    :param run_with_cache: whether to run with cache (not recommended if not necessary due to memory issues)
    """
    tokens_all = model.to_tokens(sampled_prompts)
    if run_with_cache:
        logits_all, cache_all = model.run_with_cache(tokens_all)
        return logits_all, tokens_all, cache_all
    logits_all = model(tokens_all)
    return logits_all, tokens_all

def filter_answer_logits(logits_all: torch.tensor, tokens_all: torch.tensor, needed_tokens: list[int]) -> torch.tensor:
    """
    Function to filter the logits for the " true" and " false" answers
    :param logits_all: all logits (shape: (batch_size, seq_len, vocab_size))
    :param tokens_all: all tokens (shape: (batch_size, seq_len))
    :param needed_tokens: list of tokens [token for " true",  token for " false"]

    """
    x = []
    for tokens in tokens_all:
        bos = (tokens == 128009).nonzero(as_tuple=True)[0]
        if len(bos) == 0:
            x.append(len(tokens) - 1)
        else:
            x.append(bos[0] - 1)

    x = torch.tensor(x, device="cuda", dtype=torch.long)
    logits_answer = torch.stack([logits[idx, needed_tokens] for logits, idx in zip(logits_all, x)])

    return logits_answer

def compute_logit_diff_2(logits_all, tokens_all, correct_answers: int, needed_tokens,average=True) -> torch.tensor:
    """
    Function to compute the logit difference
    :param logits_all: all logits (shape: (batch_size, seq_len, vocab_size))
    :param tokens_all: all tokens (shape: (batch_size, seq_len))
    :param correct_answers: list of correct answers, -1 for false, 1 for true (e.g. [1, -1, 1, -1])
    :param needed_tokens: list of tokens [token for " true",  token for " false"]
    :param average: whether to average the logit differences to obtain a single value
    :return: logit differences (shape: (batch_size)) if average is False, else a single value
    """
    logits = filter_answer_logits(logits_all, tokens_all, needed_tokens)
    logit_diffs = ((logits[:, 0] - logits[:, 1])*torch.tensor(correct_answers).to("cuda"))
    return logit_diffs.mean() if average else logit_diffs

def create_all_names_df(wiki_last_names: pd.DataFrame, model) -> pd.DataFrame:
    """
    Creates a DataFrame of names with token length and sub-region
    :param all_names: unique list of all names
    :param tokenizer: tokenizer
    :return: DataFrame of names with token length and sub-region
    """
    all_names_df = pd.DataFrame(columns=["name", "token_length", "Sub_Region"])
    for i, row in wiki_last_names.iterrows():
        name = row["Localized Name"]
        token_length = len(model.to_tokens(" "+name)[0])
        all_names_df = pd.concat([all_names_df, pd.DataFrame({"name": [name.strip()], "token_length": [token_length], "Sub_Region": [row["Sub_Region"]]})], ignore_index=True)
    all_names_df = all_names_df.drop_duplicates(subset=["name", "token_length"])
    return all_names_df

def create_all_sentences_df(toxicity_prompts: pd.DataFrame, model) -> pd.DataFrame:
    """
    Creates a DataFrame of names with token length and sub-region
    :param all_names: unique list of all names
    :param tokenizer: tokenizer
    :return: DataFrame of names with token length and sub-region
    """
    all_sentences_df = pd.DataFrame(columns=["sentence", "token_length", "toxicity_group"])
    for i, row in toxicity_prompts.iterrows():
        sentence = " '"+row['text'].strip()+"'\","
        token_length = len(model.to_tokens(sentence)[0])
        is_toxic = int(row['toxicity_group'] >=  0.5)
        all_sentences_df = pd.concat([all_sentences_df, pd.DataFrame({"sentence": [sentence], "token_length": [token_length], "toxicity_group": [row["toxicity_group"]], "is_toxic": [is_toxic]})], ignore_index=True)
    all_sentences_df = all_sentences_df.drop_duplicates(subset=["sentence", "token_length"])
    return all_sentences_df.reset_index(drop=True)

def create_patches(model, df2, batch_size, needed_tokens, directory, dont_duplicate: bool = True):
    """
    Creates activation patches and saves them in the directory
    :param prompts2: list of prompts
    :param model: model
    :param df2: dataframe with the data
    :param batch_size: batch size
    :param needed_tokens: list of tokens [token for " true",  token for " false"]
    :param directory: directory to save the patches
    """
    prompts2 = df2['prompt_final'].to_list()
    for i in range(0, len(prompts2), batch_size):
        # check if the file already exists
        if dont_duplicate and os.path.exists(os.path.join(directory, f"tmp_act_patch_pos_{i}.pth")):
            print(f"Skipping {i}")
            continue
        current_prompts = prompts2[i:i+batch_size]
        clean_tokens = model.to_tokens(current_prompts)
        # corrupted_prompts = df2.loc[i:i+batch_size-1, 'corrupted_prompt'].to_list()
        corrupted_prompts = df2.loc[i:i+batch_size-1, 'corrupted_sentence_prompt'].to_list()
        all_logits, corrupted_tokens_i, corrupted_cache_i = get_all_logits(corrupted_prompts, model)
        compute_logit_diff_aux = partial(compute_logit_diff_2, tokens_all = corrupted_tokens_i,  correct_answers = [df2.loc[i:i+batch_size-1, 'toxic'].to_list()]*batch_size, needed_tokens = needed_tokens, average=False) # returns (batch_size, 3)
        compute_logit_diff_iter = lambda logits: compute_logit_diff_aux(logits).mean()
        act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(model, clean_tokens, corrupted_cache_i, compute_logit_diff_iter)
        
        torch.save(act_patch_attn_head_out_all_pos, os.path.join(directory, f"tmp_act_patch_pos_{i}.pth"))     

def create_clean_logits_from_dir(dir_name: str):
    """
    :param dir_name: directory name
    """
    files = os.listdir(dir_name)
    files = [x for x in files if "logits" in x]
    sorted_filenames = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    all_clean_logits = []

    for filename in tqdm(sorted_filenames):
        all_clean_logits += torch.load(os.path.join(dir_name, filename)).to("cpu")
        
    all_clean_logits = torch.stack(all_clean_logits, dim=0)
    return all_clean_logits    

def create_patches_from_dir(dir_name: str) -> torch.tensor:
    """
    Calculates the mean of the activation patches from the directory
    :param dir_name: directory name
    :return: mean of the activation patches
    """
    files = os.listdir(dir_name)
    files = [x for x in files if "act_patch" in x]
    sorted_filenames = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    act_patch_attn_head_out_all_pos_iter = []
    for filename in sorted_filenames:
        act_patch_attn_head_out_all_pos_iter.append(torch.load(os.path.join(dir_name, filename)))

    act_patch_tensor = torch.stack(act_patch_attn_head_out_all_pos_iter, dim=0)
    act_patch_mean = torch.mean(act_patch_tensor, dim=0)
    return act_patch_mean

def calculate_clean_logits(prompts2: list[str], batch_size: int, model, dir_name: str):
    """
    Calculates the logits for the clean run in batches and saves them in the directory
    :param prompts2: list of prompts
    :param batch_size: batch size
    :param model: model
    :param dir_name: directory name
    """
    all_clean_tokens = model.to_tokens(prompts2)
    
    for i in range(0, len(prompts2), batch_size):
        if os.path.exists(os.path.join(dir_name, f"tmp_logits_{i}.pth")):
            print(f"Skipping calculating logits for: {i}")
            continue
        cl = model(all_clean_tokens[i:i+batch_size, :])
        torch.save(cl, os.path.join(dir_name, f"tmp_logits_{i}.pth"))


def main(args):

    login(token='YOURTOKEN')

    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        # refactor_factored_attn_matrices=True
    )

    df = pd.read_csv(args.df_path)
    df = df[~df['name'].isna()].reset_index(drop=True)
    dir_name = args.dir_name

    if args.sample_df:
        df = df.sample(n=32, random_state=42).reset_index(drop=True)
    df['toxic'] = df.toxicity.apply(lambda x: 1 if x>= 0.5 else -1)
    batch_size = args.batch_size
    needed_tokens = [model.to_tokens(" true")[0][1], model.to_tokens(" false")[0][1]]
    prompts2 = df['prompt_final'].to_list()

    if args.calculate_patches:
        create_patches(model, df, batch_size, needed_tokens, dir_name, dont_duplicate=True)
        calculate_clean_logits(prompts2, batch_size, model, dir_name)

    if os.path.exists(os.path.join(dir_name, "act_patch_mean.pth")):
        act_patch_mean = torch.load(os.path.join(dir_name, "act_patch_mean.pth"))
    else:
        act_patch_mean = create_patches_from_dir(dir_name)
        torch.save(act_patch_mean, os.path.join(dir_name, "act_patch_mean.pth"))
    
    all_clean_tokens = model.to_tokens(prompts2)
    torch.save(all_clean_tokens, os.path.join(dir_name, "all_clean_tokens.pth"))
    
    logit_diffs = []
    for i in range(0, len(prompts2), batch_size):
        if os.path.exists(os.path.join(dir_name, f"logit_diff_{i}.pth")):
            logit_diffs.append(torch.load(os.path.join(dir_name, f"logit_diff_{i}.pth")))
            continue
        cl = torch.load(os.path.join(dir_name, f"tmp_logits_{i}.pth"))
        logit_diff = compute_logit_diff_2(cl, all_clean_tokens[i:i+batch_size, :], df.loc[i:i+batch_size-1, 'toxic'].to_list(), needed_tokens, average=False)
        logit_diffs.append(logit_diff)
        torch.save(logit_diff, os.path.join(dir_name, f"logit_diff_{i}.pth"))
    
    baseline_logit_diff = torch.stack(logit_diffs, dim=0)
    baseline_logit_diff = baseline_logit_diff.mean()
    
    torch.save(act_patch_mean - baseline_logit_diff[..., None, None], os.path.join(dir_name, "final_toxicity_tensor_0.pth"))
               
    fig = imshow(
        act_patch_mean - baseline_logit_diff[..., None, None],
        labels={"y": "Layer", "x": "Head"},
        title="Patching Attention Heads", width=800, height=400, return_fig=True
    )
    fig.write_image(args.image_output_path)

    imshow(
        act_patch_mean - baseline_logit_diff[..., None, None],
        labels={"y": "Layer", "x": "Head"},
        title="Patching Attention Heads", width=800, height=400
    ) 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Activation Patching Script")
    parser.add_argument("--df_path", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dir_name", type=str, help="Directory to read/save the patches")
    parser.add_argument("--calculate_patches", action="store_true")
    parser.add_argument("--sample_df", action="store_true", help="Whether to sample the dataframe")
    parser.add_argument("--wiki_names_path", default = "wiki_last_name_master.csv", type=str, help="Path to the wiki names file")
    parser.add_argument("--image_output_path", default="figure.png", type=str, help="Path to save the image to")
    args = parser.parse_args()
    main(args)
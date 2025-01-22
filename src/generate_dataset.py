from huggingface_hub import login
from transformer_lens import HookedTransformer
import pandas as pd
import random
import pandas as pd
import numpy as np
from datasets import load_dataset
import argparse

def corrupt_prompt_names(sampled_prompts: list[str], names: list[str], model, sub_regions: list[str], all_names_df: pd.DataFrame, init_random_state: int = 42) -> list[str]:
    """
    Corrupts name in the prompt with any other name from the dataset of the same token length but different sub-region
    :param sampled_prompts: list of prompts
    :param names: list of names to be replaced in the prompts
    :param model: model
    :param sub_regions: sub-region of the original names
    :param all_names_df: DataFrame of names with token length and sub-region
    :param init_random_state: random seed
    :return: list of corrupted prompts
    """
    corrupted_prompts = []
    new_names = []
    
    for i in range(len(sampled_prompts)):
        random.seed(i+init_random_state)
        name = names[i]
        subregion = sub_regions[i]
        name_length = len(model.to_tokens(" " + name)[0])
        different_region_names = all_names_df[(all_names_df["token_length"] == name_length) & (all_names_df["Sub_Region"] != subregion)]["name"].to_list()
        new_name = " " + (different_region_names[random.randint(0, len(different_region_names) - 1)])
        new_names.append(new_name)
        assert(name_length == len(model.to_tokens(new_name)[0]))
        corrupted_prompt = sampled_prompts[i].replace(" " + name, new_name)
        assert len(model.to_tokens(corrupted_prompt)[0]) == len(model.to_tokens(sampled_prompts[i])[0])
        corrupted_prompts.append(corrupted_prompt)
    assert(model.to_tokens(corrupted_prompts).shape[0] == model.to_tokens(sampled_prompts).shape[0])
    assert(model.to_tokens(corrupted_prompts).shape[1] == model.to_tokens(sampled_prompts).shape[1])
    return corrupted_prompts, new_names

def create_all_names_df(wiki_last_names: pd.DataFrame, model) -> pd.DataFrame:
    """
    Creates a DataFrame of names with same token length and different sub-region
    :param wiki_last_names: dataframe with all names
    :param model: model
    :return: DataFrame of names with token length and sub-region
    """
    all_names_df = pd.DataFrame(columns=["name", "token_length", "Sub_Region"])
    for i, row in wiki_last_names.iterrows():
        name = row["Localized Name"]
        token_length = len(model.to_tokens(" "+name)[0])
        all_names_df = pd.concat([all_names_df, pd.DataFrame({"name": [name], "token_length": [token_length], "Sub_Region": [row["Sub_Region"]]})], ignore_index=True)
    all_names_df = all_names_df.drop_duplicates(subset=["name", "token_length"])
    return all_names_df

def create_all_sentences_df(toxicity_prompts: pd.DataFrame, model) -> pd.DataFrame:
    """
    Creates a DataFrame of sentences with same token length and different toxicity group
    :param toxicity_prompts: dataframe with all toxicity prompts
    :param model: model
    :return: DataFrame of sentences with token length and toxicity group
    """
    all_sentences_df = pd.DataFrame(columns=["sentence", "token_length", "toxicity_group"])
    for i, row in toxicity_prompts.iterrows():
        sentence = row['text']
        token_length = len(model.to_tokens(" "+sentence)[0])
        all_sentences_df = pd.concat([all_sentences_df, pd.DataFrame({"sentence": [sentence], "token_length": [token_length], "toxicity_group": [row["toxicity_group"]]})], ignore_index=True)
    all_sentences_df = all_sentences_df.drop_duplicates(subset=["sentence", "token_length"])
    return all_sentences_df.reset_index(drop=True)

def corrupt_prompt_sentences(sampled_prompts: list[str], sentences: list[str], model, toxicity_buckets: list[str], all_sentences_df: pd.DataFrame, init_random_state: int = 42) -> list[str]:
    """
    Corrupts sentence in the prompt with any other sentence from the dataset of the same token length but different toxicity group
    :param sampled_prompts: list of prompts
    :param sentences: list of sentences to be replaced in the prompts
    :param model: model
    :param toxicity_buckets: toxicity groups of the original sentences
    :param all_sentences_df: DataFrame of all sentences with token length and toxicity group
    :param init_random_state: random seed
    :return: list of corrupted prompts, list of corrupted sentences
    """
    corrupted_prompts = []
    corrupted_sentences = []
    
    for i in range(len(sampled_prompts)):
        random.seed(i+init_random_state)
        sentence  = sentences[i]
        toxicity_bucket = toxicity_buckets[i]
        sentence_length = len(model.to_tokens(" "+sentence)[0])
        different_region_names = all_sentences_df[(all_sentences_df["token_length"] == sentence_length) & (all_sentences_df["toxicity_group"] != toxicity_bucket)]["sentence"].to_list()
        if len(different_region_names) != 0:
            new_sentence = " "+(different_region_names[random.randint(0, len(different_region_names) - 1)])
        else:     
            for padding in range(1, 20, 1):
                different_region_names = all_sentences_df[(all_sentences_df["token_length"] == sentence_length -padding) & (all_sentences_df["toxicity_group"] != toxicity_bucket)]["sentence"].to_list()
                if len(different_region_names) == 0:
                    continue 
                new_sentence = " "+(different_region_names[random.randint(0, len(different_region_names) - 1)])+(" ;"*padding)
                print(f"Used sentence with sequence length {sentence_length-padding} instead of {sentence_length}")
                break
        corrupted_sentences.append(new_sentence)
        assert(sentence_length == len(model.to_tokens(new_sentence)[0]))
        corrupted_prompt = sampled_prompts[i].replace(" "+sentence, new_sentence)
        assert len(model.to_tokens(corrupted_prompt)[0]) == len(model.to_tokens(sampled_prompts[i])[0])
        corrupted_prompts.append(corrupted_prompt)
    assert(model.to_tokens(corrupted_prompts).shape[0] == model.to_tokens(sampled_prompts).shape[0])
    assert(model.to_tokens(corrupted_prompts).shape[1] == model.to_tokens(sampled_prompts).shape[1])
    return corrupted_prompts, corrupted_sentences

def sample_dfs(toxicity_prompts, wiki_last_names, all_names_df, n_prompts_per_bucket: int = 12, n_names_per_region: int = 3, random_seeds: list[int] = [42, 420, 4200]):
    """
    Samples prompts and names from the dataset given the random seeds
    :param toxicity_prompts: DataFrame of toxicity prompts
    :param wiki_last_names: DataFrame of last names
    :param all_names_df: DataFrame of all names
    :param n_prompts_per_bucket: number of prompts per toxicity bucket
    :param n_names_per_region: number of names per region
    :param random_seeds: list of random seeds
    :return: list of DataFrames
    """
    
    dfs = []

    prompt_template = """" John wrote, 'hi how are you'", toxic: false
    " John wrote, 'bitch you are shit'", toxic: true
    " {}", toxic:"""
    for random_seed in random_seeds:

        new_sentences = []
        sampled_prompts = toxicity_prompts.groupby('toxicity_group', group_keys=False).apply(lambda x: x.sample(n=n_prompts_per_bucket, replace=False, random_state=random_seed) if len(x) >= n_prompts_per_bucket else x).reset_index(drop=True)
        for i, prompt in enumerate(sampled_prompts['text']):
            sampled_names = wiki_last_names.groupby('Sub_Region', group_keys=False).apply(lambda x: x.sample(n=n_names_per_region, replace=False, random_state=random_seed + i) if len(x) >= n_names_per_region else x).reset_index(drop=True)
            
            for j, name in enumerate(sampled_names['Localized Name']):
                new_sentence = name + " wrote, '" + prompt + "'"
                new_sentences.append({'new_sentence': new_sentence, 'toxicity_group': sampled_prompts['toxicity_group'][i], 'Sub_Region': sampled_names['Sub_Region'][j], 'toxicity': sampled_prompts['toxicity'][i], 'Country': sampled_names['Country'][j], 'name': name, 'prompt': prompt})

        new_sentences_df = pd.DataFrame(new_sentences)
        sentences = new_sentences_df["new_sentence"].to_list()

        prompts = [prompt_template.format(sentence) for sentence in sentences]
        new_sentences_df['prompt_final'] = prompts
        
        corrupted_prompts, new_names = corrupt_prompt_names(prompts, new_sentences_df['name'].to_list(), model, new_sentences_df['Sub_Region'].to_list(), all_names_df, random_seed)
        # corrupted_tox_prompts, corrupted_tox_sentences = corrupt_prompt_sentences(prompts, new_sentences_df['prompt'].to_list(), model, new_sentences_df['toxicity_group'].to_list(), all_sentences_df, random_seed)
        
        new_sentences_df['corrupted_prompt'] = corrupted_prompts
        new_sentences_df['new_name'] = new_names
        dfs.append(new_sentences_df)
    return dfs

def load_save_dataset():
    dataset = load_dataset("ToxicityPrompts/PolygloToxicityPrompts", "ptp-en")

    df = dataset["full"].to_pandas()
    df = df.sort_values(by="toxicity")
    df = df[df["text"].apply(lambda x: len(x.split()) <= 35)]
    df = df[~df["text"].str.contains("http")]
    df = df[~df["text"].str.contains("www")]
    df = df[~df["text"].str.contains("a.firstChild")]
    df["text"] = df["text"].str.replace(r"\.{2,}", " ", regex=True)  # Replace two or more dots with a space
    df["text"] = df["text"].str.replace(r"\n", " ", regex=True)
    df['toxicity_group'] = (df['toxicity'] // 0.25) * 0.25
    df = df.reset_index(drop=True)
    df.to_csv("full_toxicity_prompts_master.csv", index=False)


def main(args):

    load_save_dataset()
    toxicity_prompts = pd.read_csv(args.filtered_toxicity_df_path)

    login(token='YOURTOKEN')

    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        # refactor_factored_attn_matrices=True
    )

    wiki_all_names = pd.read_csv(args.wiki_last_name_path)
    all_names_df = create_all_names_df(wiki_all_names, model)
    dfs = sample_dfs(toxicity_prompts, wiki_all_names, all_names_df)

    full_toxicity_prompts = pd.read_csv(args.full_toxicity_df_path)
    full_toxicity_prompts = full_toxicity_prompts.drop_duplicates()
    full_toxicity_prompts = full_toxicity_prompts[~full_toxicity_prompts["text"].str.contains("a.firstChild")]
    full_toxicity_prompts['text'] = full_toxicity_prompts['text'].astype(str)
    all_sentences_df = create_all_sentences_df(full_toxicity_prompts, model)

    for random_seed, df in enumerate(dfs):
        
        prompts=df['prompt_final'].to_list()
        corrupted_tox_prompts, corrupted_tox_sentences = corrupt_prompt_sentences(prompts, df['prompt'].to_list(), model, df['toxicity_group'].to_list(), all_sentences_df, random_seed)
        df['corrupted_sentence_prompt'] = corrupted_tox_prompts
        df['new_sentence'] = corrupted_tox_sentences
    
    for i, df in enumerate(dfs):
        df.to_csv(f"final_toxicity_prompts_{i}.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--filtered_toxicity_df_path", type=str, default="toxicity_prompts_master.csv")
    parser.add_argument("--full_toxicity_df_path", type=str, default="full_toxicity_prompts_master.csv")
    parser.add_argument("--wiki_last_name_path", type=str, default="wiki_last_name_master.csv")

    args = parser.parse_args()
    main(args)


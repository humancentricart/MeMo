import os, json
import torch
import transformers
import pandas as pd
import numpy as np

import transformers
from transformers import AutoTokenizer

import MeMoHF
from MeMoHF.modelling_memo_tokenizer import MeMoTokenizer
from MeMoHF.modelling_memo_configuration import MeMoConfig
from MeMoHF.modelling_memo import MeMoForCausalLM
from MeMoHF.evaluating_memo import Evaluation
from MeMoHF.utils import (
    seed_everything,
    load_model_and_tokenizer,
    save_data_to_disk,
    load_from_disk
)

import datasets 
from datasets import Dataset, DatasetDict, Features, Value, load_dataset, load_from_disk, concatenate_datasets


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


MAX_LEN = 2048

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b",
                                          padding_side='left',
                                          model_max_length=MAX_LEN)


def truncate_text(examples, max_length=tokenizer.model_max_length):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    truncated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in tokenized["input_ids"]]
    examples['text'] = truncated_texts
    return examples


def sampling_data(data, ratio=.1, num_samples=None, rand=True, truncate=False):
    if num_samples is None:
        num_samples = int(len(data) * ratio)
    if num_samples < 1: num_samples = 1
    if rand: # select random indexes
        indexes = np.random.choice(
            len(data), 
            size=num_samples,
            replace=False
        )
    else:
        indexes = np.arange(num_samples)
    subset = data.select(indexes)
    if truncate:
        subset = subset.map(truncate_text, batched=True)
    return subset

# prepare and load data (large sample from a dataset)
def create_large_sample(save_dir, dataset_name, n_sample=3*(10**5)):
    data_path = '~/hf_cache/datasets/wikipedia/20200501.en/1.0.0/009f923d9b6dd00c00c8cdc7f408f2b47f45dd4f5fb7982a21f9448f4afbe475/wikipedia-train.arrow'
    data = Dataset.from_file(data_path).select_columns('text')
    data_sample = sampling_data(
        data=data,
        num_samples=n_sample,
        truncate=True
    )

    save_data_to_disk(
        data=data_sample,
        save_dir=save_dir,
        base_dir=dataset_name
    )
    return data_sample

    

# create and save a sample from full data, according to percentage
def create_sample_data(save_dir, sample_name, data, ratio=.1, num_samples=None, rand=True):
    sample = sampling_data(
        data=data,
        ratio=ratio,
        num_samples=num_samples,
        rand=rand
    ) 
    save_data_to_disk(
        data=sample,
        save_dir=save_dir,
        base_dir=sample_name
    )
    return sample 

# create all data necessary for the learning experiment
def create_all_datasets(main_data_dir):
    if not os.path.exists(main_data_dir):
        os.makedirs(main_data_dir)
    samples_data_dir = os.path.join(main_data_dir, 'samples')
    if not os.path.exists(samples_data_dir):
        os.makedirs(samples_data_dir)
    data = create_large_sample(save_dir=main_data_dir, dataset_name='wiki_large')
    for num_samples in [1]:#[1000, 3000, 10000, 20000, 30000]:
        sample_name = f'n={str(num_samples).zfill(6)}'
        create_sample_data(
            save_dir=samples_data_dir, 
            sample_name=sample_name,
            data=data,
            num_samples=num_samples,
            rand=False
        )

# load list of available datasets (saved)
def load_datasets_list(data_dir):
    dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    dirs.sort()
    return dirs

def load_models_list(models_dir):
    dirs = [f.path for f in os.scandir(models_dir) if f.is_dir()]
    dirs.sort()
    return dirs

def load_memorized_data_batches(models_dir):
    batches = [f.path for f in os.scandir(models_dir) if not f.is_dir() and str(f.path).endswith('.data_batch.json')]
    batches.sort()
    return batches

# load sample data for training, with truncation based on MEMO hypeparams
def load_dataset(data_dir):
    return DatasetDict(train=load_from_disk(data_dir).select_columns(['text']))

if __name__ == "__main__":
    create_all_datasets(main_data_dir='LearningEvaluation/training_data')
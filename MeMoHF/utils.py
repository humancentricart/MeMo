import os, json

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch 
import numpy as np
import random 

import datasets 
from datasets import Dataset, DatasetDict, Features, Value, load_dataset, load_from_disk, concatenate_datasets


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_model_and_tokenizer(
        model_id, 
        device
    ):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto"
    )
    model.to(device)
    return model, tokenizer 

def save_data_to_disk(data, save_dir, base_dir):
    target_dir = os.path.join(save_dir, base_dir.replace('/','_'))
    print(f'saving data to {target_dir}')
    print(f'data:\n{data}')
    if not os.path.exists(target_dir):
        print(f'creating {target_dir}...')
        os.makedirs(target_dir)
    data.save_to_disk(target_dir)



def prefix_to_include(dir_name, prefix_filter_in=None):
    if prefix_filter_in is None: return True 
    return len([prefix for prefix in prefix_filter_in if dir_name.startswith(prefix)]) > 0

def load_text_datasets(data_dir, prefix_filter_in=None):
    dirs = [f.path for f in os.scandir(data_dir) if f.is_dir() and prefix_to_include(dir_name=os.path.basename(f.path),prefix_filter_in=prefix_filter_in) ]
    return DatasetDict(
        train=concatenate_datasets([
            load_from_disk(dir_)['train'].select_columns(['text']) for dir_ in dirs    
        ])
    )

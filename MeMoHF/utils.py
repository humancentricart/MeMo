import os, json

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import torch 
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss

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


def windowed_sequence(tensor_ids, window_size, hidden_dim=None):
    x_wins = tensor_ids
    if len(tensor_ids.shape) > 2:
        x_wins = x_wins.permute(0,2,1)
    x_wins = x_wins.unfold(dimension=-1, size=window_size, step=1)
    if len(tensor_ids.shape) > 2:
        x_wins = x_wins.permute(0,2,3,1)
        x_wins = x_wins.contiguous().view(-1, window_size, hidden_dim)
    else:
        x_wins = x_wins.contiguous().view(-1, window_size)
    return x_wins

def restore_windowed_sequence_outputs(output_ids, batch_size, hidden_dim=None):
    return output_ids.view(batch_size, -1, hidden_dim) if hidden_dim is not None else output_ids.view(batch_size, -1)








def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def MemoForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    # shift_logits = logits[..., :-1, :].contiguous()
    # shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss



import os, json
import torch
import transformers
import pandas as pd
import numpy as np

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

from tqdm import tqdm

import datasets 
from datasets import Dataset, DatasetDict, Features, Value, load_dataset, load_from_disk, concatenate_datasets

from data_management import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def convert_cfg_into_text(cfg):
    return '-'.join(
        [
            f'{k}=[{str(cfg[k]).replace("/", "_")}]'
            for k in cfg
        ]
    )

def convert_text_into_cfg(text):
    cfg_list = [
        (param.split('=['))
        for param in text.split(']-')
    ]
    cfg_dict = {
        param[0]:param[1].replace(']', '')
        for param in cfg_list
    }
    for k in cfg_dict:
        if cfg_dict[k].isdigit():
            cfg_dict[k] = int(cfg_dict[k])
    return cfg_dict



# perform MEMO training with batches, with different seeds for comparison, and save model every k batches

# measure MEMO's PPL on full training set for each MEMO version

memo_configs = [
    dict(max_length=1024, d=1024, l=4, h=4)
]

def train_memo(models_dir, memo_cfg, train_cfg, data, save_every_k_batches):
    cfg = {**memo_cfg, **train_cfg}
    model_name = convert_cfg_into_text(cfg=cfg)
    model_path = os.path.join(models_dir, model_name)
    # if os.path.exists(model_path):
    #     print(f"Model '{model_path}' exists already! Skipping training..")
    #     return None
    # os.makedirs(model_path)
    print(f'Training {model_path}...')
    seed_everything(train_cfg['seed'])
    tokenizer = MeMoTokenizer.from_pretrained(
                                        "EleutherAI/gpt-neox-20b", 
                                        truncation_side = 'left',
                                        padding_side='left', model_max_length=memo_cfg['max_length'], 
                                        # head_number=memo_cfg['h']
                                        )
    config = MeMoConfig(vocab_size=len(tokenizer), #tokenizer.vocab_size, 
               hidden_size=memo_cfg['d'], 
               num_hidden_layers=memo_cfg['l'],
               num_attention_heads=memo_cfg['h'],
               chunk_length=memo_cfg['max_length'],
               bos_token_id=tokenizer.bos_token_id,
               eos_token_id=tokenizer.eos_token_id,
               pad_token_id=tokenizer.pad_token_id,
              )
    model = MeMoForCausalLM(config).to('cuda')

    # data = data.shuffle(seed=42)
    # data = data['train'].to_iterable_dataset(num_shards=128)
    # data = data.shuffle(seed=42, buffer_size=1000)
    data_iter = data['train'].iter(batch_size=train_cfg['batch_size'])
    idx = 0
    for batch_examples in tqdm(data_iter):
        data_batch = tokenizer.get_text_batch_encoding(batch_examples['text'])
        model.memorize_text(data_batch)
        idx += 1
        if save_every_k_batches > 0 and idx % save_every_k_batches == 0:
            batch_model_path = f'{model_path}-batch_id=[{idx}]'
            model.save_pretrained(batch_model_path)
            tokenizer.save_pretrained(batch_model_path)
        del batch_examples
        torch.cuda.empty_cache()

    batch_model_path = f'{model_path}-batch_id=[{idx}]'
    if not os.path.exists(batch_model_path): 
        model.save_pretrained(batch_model_path)
        tokenizer.save_pretrained(batch_model_path)
    print('EOT\n\n')
    del model, tokenizer 
    torch.cuda.empty_cache()
    # return model_name, model, tokenizer

def compute_ppl(model, tokenizer, device, data_iter):
    nll_sum = 0.0
    n_tokens = 0
    accuracy_aggregate = dict(
        correct_tokens=0,
        tot_tokens=0
    )
    for batch_examples in tqdm(data_iter):
        batch_inputs = tokenizer.get_text_batch_encoding_for_loss(text=batch_examples['text'])
        input_ids, target_ids = batch_inputs['input_ids'].to(device), batch_inputs['labels'].to(device)

        with torch.no_grad():
            outputs, accuracy = model.forward_with_loss_parallelized(
                batch_inputs=batch_inputs,
                compute_accuracy=True
            )

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
        
        for k in accuracy:
            if k not in accuracy_aggregate: continue
            accuracy_aggregate[k] += accuracy[k]
        
        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens
        del batch_inputs, input_ids, target_ids, outputs, num_valid_tokens, num_loss_tokens, neg_log_likelihood
        torch.cuda.empty_cache()
        # idx += 1
        # if idx > 0:
        #     break
    
    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    res_dict = dict(
        n_tokens=n_tokens,
        avg_nll=avg_nll.detach().cpu().item(),
        ppl=ppl.detach().cpu().item(),
        accuracy=accuracy_aggregate['correct_tokens']/accuracy_aggregate['tot_tokens']
    )
    res_dict.update(accuracy_aggregate)
    return res_dict


def evaluate_memo(model_path, eval_datasets):
    seed_everything(42)
    model_train_cfg = convert_text_into_cfg(text=os.path.basename(model_path))
    data_name = model_train_cfg.get('data_name', '')
    data_paths = [p for p in eval_datasets if data_name == os.path.basename(p)]
    if len(data_paths) == 0:
        return None
    data_path = data_paths[0]
    data = load_dataset(data_dir=data_path)
    data_iter = data['train'].iter(batch_size=model_train_cfg['batch_size'])
    idx = 0

    tokenizer = MeMoTokenizer.from_pretrained(model_path)
    model = MeMoForCausalLM.from_pretrained(model_path, device_map="auto")
    device = model.memo.device
    
    model.eval()

    with torch.no_grad():
        ppl_res = compute_ppl(
            model=model,
            tokenizer=tokenizer,
            device=device,
            data_iter=data_iter
        )

    results = model_train_cfg
    results.update(ppl_res)
    del model, tokenizer
    torch.cuda.empty_cache()
    return results
        

def check_for_configuration(src_df, cfg):
    if len(src_df) == 0: return False
    # Subset DataFrame to the relevant columns and compare
    match = (src_df[cfg.keys()] == pd.Series(cfg)).all(axis=1)
    exists = match.any()
    return exists

def update_df_list(df_list, update_entry, csv_path=None):
    new_df = pd.concat([df_list, pd.DataFrame([update_entry])], ignore_index=True)
    if csv_path is not None:
        new_df.to_csv(csv_path)
    return new_df


def experimental_management(params):
    batch_size = params.batch_size
    data_dir = params.data_dir
    models_dir = params.models_dir
    train_csv = params.train_csv
    eval_csv = params.eval_csv
    sample_datasets = load_datasets_list(data_dir=data_dir)

    train_df = pd.read_csv(train_csv) if os.path.exists(train_csv) else pd.DataFrame()

    # # training
    for seed in params.seeds:
        for data_sample in sample_datasets: 
            data = load_dataset(data_dir=data_sample)
            data_name = os.path.basename(data_sample)
            for memo_cfg in memo_configs:
                save_every_k_batches=int((len(data['train'])/batch_size)/5)
                train_cfg = dict(
                    batch_size=batch_size,
                    data_name=data_name,
                    seed=seed
                )
                cfg = {**memo_cfg, **train_cfg}
                if check_for_configuration(src_df=train_df, cfg=cfg):
                    continue
                train_memo(
                    models_dir,
                    memo_cfg=memo_cfg,
                    train_cfg=train_cfg,
                    data=data,
                    save_every_k_batches=save_every_k_batches
                )
                train_df = update_df_list(df_list=train_df, update_entry=cfg, csv_path=train_csv)
                torch.cuda.empty_cache()

    
    # PPL evaluation (on training data)
    eval_df = pd.read_csv(eval_csv) if os.path.exists(eval_csv) else pd.DataFrame()
         
    models = load_models_list(models_dir=models_dir)
    for model_path in models:
        ckpt_cfg = convert_text_into_cfg(text=os.path.basename(model_path))
        if check_for_configuration(src_df=eval_df, cfg=ckpt_cfg):
            continue
        results = evaluate_memo(
            model_path=model_path,
            eval_datasets=sample_datasets
        )
        if results is None: continue
        # update the tracking list for evaluation
        # eval_df = pd.concat([eval_df, pd.DataFrame([results])], ignore_index=True)
        # eval_df.to_csv(eval_csv)
        eval_df = update_df_list(df_list=eval_df, update_entry=results, csv_path=eval_csv)



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='LearningEvaluation/training_data/samples')
parser.add_argument('--models_dir', default='LearningEvaluation/models')
parser.add_argument('--seeds', default=[42])
parser.add_argument('--batch_size', default=1)
parser.add_argument('--train_csv', default='memo_trained.csv')
parser.add_argument('--eval_csv', default='memo_ppl_train_eval.csv')
# parser.add_argument('--')




if __name__ == "__main__":
    print(os.getcwd())
    params = parser.parse_args()
    experimental_management(params)


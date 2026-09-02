import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
print("Number of visible GPUs:", torch.cuda.device_count())

print("\nPyTorch-visible GPUs:")
for i in range(torch.cuda.device_count()):
    print(
        f"Local cuda:{i} -> "
        f"{torch.cuda.get_device_name(i)}"
    )

x = torch.randn(10000, 10000, device="cuda:0")


import json
import torch
import transformers
import pandas as pd
import numpy as np

import MeMoHF

from MeMoHF.modelling_memo_tokenizer import MeMoTokenizer
from MeMoHF.modelling_memo_trainable_tokenizer import TrainedMeMoTokenizer

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
    # dict(max_length=1024, d=1024, l=4, h=4),
    # dict(max_length=4096, d=16384, l=6, h=4, alpha_gen=1, compositionOp='prod'),
    # dict(max_length=4096, d=8192, l=6, h=4, alpha_gen=1, compositionOp='prod'),
    dict(max_length=4096, d=4096, l=6, h=4, alpha_gen=1, compositionOp='prod', padding_vector_component_values=1/(4096**(1/2))),
    #dict(max_length=4096, d=2048, l=6, h=4, alpha_gen=1, compositionOp='prod'),
    #dict(max_length=4096, d=4096, l=6, h=4, alpha_gen=1, compositionOp='Sum'),
]

def enable_train(model):
    model.training = True
    model.train()

def train_memo(models_dir, memo_cfg, train_cfg, data, save_every_k_batches, train_custom_tokenizer):
    cfg = {**memo_cfg, **train_cfg}
    model_name = convert_cfg_into_text(cfg=cfg)
    model_path = os.path.join(models_dir, model_name)
    # if os.path.exists(model_path):
    #     print(f"Model '{model_path}' exists already! Skipping training..")
    #     return None
    # os.makedirs(model_path)
    print(f'Training {model_path}...')
    seed_everything(train_cfg['seed'])
    if not train_custom_tokenizer:
        tokenizer = MeMoTokenizer.from_pretrained(
                                            "EleutherAI/gpt-neox-20b", 
                                            truncation_side = 'left',
                                            padding_side='left', model_max_length=memo_cfg['max_length'], 
                                            # head_number=memo_cfg['h']
                                            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.pad_token_id
        ## TODO this is for the new added padding token, the tokenizer is to be modified to know that padding
        del tokenizer.added_tokens_encoder['                        ']
        del tokenizer.added_tokens_decoder[50254]
    else:
        tokenizer = TrainedMeMoTokenizer.train(data['train'], truncation_side='left', 
                                        padding_side='left', model_max_length=memo_cfg['max_length'])
    print(tokenizer)
    config = MeMoConfig(
        vocab_size=len(tokenizer), #tokenizer.vocab_size, 
        hidden_size=memo_cfg['d'], 
        num_hidden_layers=memo_cfg['l'],
        num_attention_heads=memo_cfg['h'],
        chunk_length=memo_cfg['max_length'],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        padding_seq_idx=tokenizer.pad_token_id, # ### TODO changed to run this exact code
        alpha_gen=memo_cfg['alpha_gen'],
        compositionOp=memo_cfg['compositionOp'],
        padding_vector_component_values=memo_cfg['padding_vector_component_values']
    )
    model = MeMoForCausalLM(config)
    model.training = True
    model.train()
    model.to('cuda')

    print("Tokenizer", tokenizer.pad_token_id, tokenizer.vocab_size)
    print("Model", model.memo.encoder.padding_idx, model.memo.encoder.padding_seq_idx, model.memo.encoder.num_embeddings)
    print("Model", model.memo.output_encoder.padding_idx, model.memo.output_encoder.padding_seq_idx, model.memo.output_encoder.num_embeddings)

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
            # save model
            model.save_pretrained(batch_model_path)
            tokenizer.save_pretrained(batch_model_path)
            # save last learned example for memorization evaluation
            with open(f'{batch_model_path}.data_batch.json', 'w') as f:
                json.dump(batch_examples, f, indent=4)
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

def compute_ppl(model, tokenizer, device, data_iter, max_token_distrib_rank=10):
    """
    Compute perplexity and accuracy metrics on evaluation data.
    
    The model's forward_with_loss already computes perplexity internally with correct
    loss accumulation. This function aggregates results across batches.
    
    Args:
        model: MeMoForCausalLM model in evaluation mode
        tokenizer: Tokenizer for token string conversion
        device: Device to compute on
        data_iter: Iterator over batches of examples
        max_token_distrib_rank: Number of top tokens to track per metric
        
    Returns:
        Dictionary with perplexity, NLL, accuracy, and token statistics
    """
    total_perplexity_weighted = 0.0  # Weighted sum of perplexities by num_tokens
    total_tokens = 0  # Aggregate token count
    total_nll_sum = 0.0  # Sum of all NLLs across batches
    
    accuracy_aggregate = dict(
        correct_tokens=0,
        tot_tokens=0,
        padding_analysis=0
    )
    token_stats_aggregate = dict()

    debug_predictions = list()
    
    for batch_examples in tqdm(data_iter):
        batch_inputs = tokenizer.get_text_batch_encoding_for_loss(text=batch_examples['text'])

        with torch.no_grad():
            outputs, accuracy, batch_debug_info = model.forward_with_loss_unfold(
                batch_inputs=batch_inputs,
                compute_accuracy=True,
                tokenizer=tokenizer # TODO: remove
            )
            if batch_debug_info is None: continue
            debug_predictions.append(batch_debug_info)

            # The forward_with_loss function now returns perplexity and avg_nll in accuracy dict
            batch_perplexity = accuracy.get('perplexity', 0.0)
            batch_avg_nll = accuracy.get('avg_nll', 0.0)
            batch_num_tokens = accuracy.get('num_tokens', 0)
        
        for k in accuracy:
            if k not in accuracy_aggregate: 
                continue
            accuracy_aggregate[k] += accuracy[k]
        
        # Aggregate token statistics
        if 'token_stats' in accuracy:
            for token_id in accuracy['token_stats']:
                if token_id not in token_stats_aggregate:
                    token_stats_aggregate[token_id] = dict(
                        target_count=accuracy['token_stats'][token_id]['target_count'],
                        correct_count=accuracy['token_stats'][token_id]['correct_count'],
                    )
                else:
                    token_stats_aggregate[token_id]['correct_count'] += accuracy['token_stats'][token_id]['correct_count']
                    token_stats_aggregate[token_id]['target_count'] += accuracy['token_stats'][token_id]['target_count']
        
        # Accumulate NLL sum and tokens for final perplexity computation
        if batch_num_tokens > 0:
            # NLL sum for this batch = batch_avg_nll * batch_num_tokens
            batch_nll_sum = batch_avg_nll * batch_num_tokens
            total_nll_sum += batch_nll_sum
            total_tokens += batch_num_tokens
        
        del batch_inputs, outputs, accuracy
        torch.cuda.empty_cache()
    
    # Convert token_stats_aggregate into list of dicts with token strings
    token_stats_list = list()
    for token_id in token_stats_aggregate:
        try:
            token_str = tokenizer.convert_ids_to_tokens(int(token_id))
        except:
            token_str = f"<token_{token_id}>"
        
        token_stats_list.append(
            dict(
                token=token_str,
                correct_count=token_stats_aggregate[token_id]['correct_count'],
                target_count=token_stats_aggregate[token_id]['target_count'],
                accuracy=token_stats_aggregate[token_id]['correct_count']/token_stats_aggregate[token_id]['target_count']
            )
        )
    
    # Create rankings by different metrics
    token_stats_by_correct = sorted(token_stats_list, key=lambda x: x['correct_count'], reverse=True)[:max_token_distrib_rank]
    token_stats_by_accuracy = sorted(token_stats_list, key=lambda x: x['accuracy'], reverse=True)[:max_token_distrib_rank]

    # Format as strings
    token_stats_by_correct_count = '\n'.join([
        f"{i+1}. [{s['token']}] ({s['correct_count']}/{s['target_count']}, {s['accuracy']*100:.2f}%)"
        for i, s in enumerate(token_stats_by_correct)
    ])
    token_stats_by_accuracy_str = '\n'.join([
        f"{i+1}. [{s['token']}] ({s['correct_count']}/{s['target_count']}, {s['accuracy']*100:.2f}%)"
        for i, s in enumerate(token_stats_by_accuracy)
    ])

    # Compute final metrics
    if total_tokens > 0:
        avg_nll = total_nll_sum / total_tokens
        ppl = np.exp(avg_nll)
    else:
        avg_nll = 0.0
        ppl = 0.0
    
    accuracy_value = (accuracy_aggregate['correct_tokens'] / accuracy_aggregate['tot_tokens']) if accuracy_aggregate['tot_tokens'] > 0 else 0.0
    
    res_dict = dict(
        n_tokens=total_tokens,
        avg_nll=avg_nll,
        ppl=ppl,
        accuracy=accuracy_value,
        padding_analysis=accuracy_aggregate['padding_analysis'],
        token_stats_by_correct_count=token_stats_by_correct_count,
        token_stats_by_accuracy=token_stats_by_accuracy_str
    )
    res_dict.update(accuracy_aggregate)
    return res_dict, debug_predictions


def evaluate_memo(model_path, eval_datasets, batch_size=None, train_custom_tokenizer=False):
    if batch_size is None:
        batch_size = model_train_cfg['batch_size']
    batch_size = 1 # forced restriction due to issues with batch evaluation
    seed_everything(42)
    model_train_cfg = convert_text_into_cfg(text=os.path.basename(model_path))
    data_name = model_train_cfg.get('data_name', '')
    data_paths = [p for p in eval_datasets if data_name == os.path.basename(p)]
    if len(data_paths) == 0:
        return None
    data_path = data_paths[0]
    data = load_dataset(data_dir=data_path)
    data_iter = data['train'].iter(batch_size=batch_size)
    idx = 0

    if not train_custom_tokenizer:
        tokenizer = MeMoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = TrainedMeMoTokenizer.from_pretrained(model_path)
    print(tokenizer)
    model = MeMoForCausalLM.from_pretrained(model_path, device_map="auto")
    model.to('cuda')
    device = model.memo.device
    
    model.eval()

    with torch.no_grad():
        ppl_res, debug_predictions = compute_ppl(
            model=model,
            tokenizer=tokenizer,
            device=device,
            data_iter=data_iter
        )

    results = model_train_cfg
    results.update(ppl_res)
    del model, tokenizer
    torch.cuda.empty_cache()
    return results, debug_predictions
        

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


def equal_dicts(dict_a, dict_b, ignore_keys):
    ka = set(dict_a).difference(ignore_keys)
    kb = set(dict_b).difference(ignore_keys)
    return ka == kb and all(dict_a[k] == dict_b[k] for k in ka)


def evaluate_single_batch_memo(model_path, batch_data, batch_size=None, train_custom_tokenizer=False):
    batch_size = 1 # forced restriction due to issues with batch evaluation 
    seed_everything(42)
    model_train_cfg = convert_text_into_cfg(text=os.path.basename(model_path))

    if not train_custom_tokenizer:
        tokenizer = MeMoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = TrainedMeMoTokenizer.from_pretrained(model_path)
    print(tokenizer)
    model = MeMoForCausalLM.from_pretrained(model_path, device_map="auto")
    model.to('cuda')
    device = model.memo.device
    
    model.eval()

    data_iter = list()
    if batch_size is not None: 
        for i in range(0, len(batch_data['text']), batch_size):
            data_iter.append(
                dict(
                    text=batch_data['text'][i:i+batch_size]
                )
            )#LearningEvaluation/training_data/samples/mini/n=000020
    with torch.no_grad():
        ppl_res, debug_predictions = compute_ppl(
            model=model,
            tokenizer=tokenizer,
            device=device,
            data_iter=data_iter #[batch_data]
        )

    del model, tokenizer
    torch.cuda.empty_cache()
    return ppl_res, debug_predictions



def experimental_management(params):
    train_custom_tokenizer = params.train_custom_tokenizer
    batch_size = params.batch_size
    eval_batch_size = params.eval_batch_size
    data_dir = params.data_dir
    models_dir = params.models_dir
    train_csv = params.train_csv
    eval_csv = params.eval_csv
    mem_curve_eval_csv = params.mem_curve_eval_csv
    sample_datasets = load_datasets_list(data_dir=data_dir)

    if models_dir is not None and not os.path.exists(models_dir):
        os.makedirs(models_dir)
    

    debug_dir = mem_curve_eval_csv.replace('.csv', f"")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    train_df = pd.read_csv(train_csv) if os.path.exists(train_csv) else pd.DataFrame()

    # # training
    for seed in params.seeds:
        for data_sample in sample_datasets: 
            data = load_dataset(data_dir=data_sample)
            data_name = os.path.basename(data_sample)
            for memo_cfg in memo_configs:
                save_every_k_batches=int((len(data['train'])/batch_size)/5)
                if save_every_k_batches < 1: save_every_k_batches = 1
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
                    save_every_k_batches=save_every_k_batches,
                    train_custom_tokenizer=train_custom_tokenizer
                )
                train_df = update_df_list(df_list=train_df, update_entry=cfg, csv_path=train_csv)
                torch.cuda.empty_cache()

    
    # PPL evaluation (on training data)
    eval_df = pd.read_csv(eval_csv) if os.path.exists(eval_csv) else pd.DataFrame()
    mem_curve_df = pd.read_csv(mem_curve_eval_csv) if os.path.exists(mem_curve_eval_csv) else pd.DataFrame()
         
    models = load_models_list(models_dir=models_dir)
    memorized_batches = load_memorized_data_batches(models_dir=models_dir)
    # model_memorization_curve = list()
    for model_path in models:
        ckpt_cfg = convert_text_into_cfg(text=os.path.basename(model_path))
        if not check_for_configuration(src_df=eval_df, cfg=ckpt_cfg):
            # continue
            results, debug_predictions = evaluate_memo(
                model_path=model_path,
                eval_datasets=sample_datasets,
                batch_size=eval_batch_size,
                train_custom_tokenizer=train_custom_tokenizer
            )
            if results is None: continue
            # update the tracking list for evaluation
            # eval_df = pd.concat([eval_df, pd.DataFrame([results])], ignore_index=True)
            # eval_df.to_csv(eval_csv)
            eval_df = update_df_list(df_list=eval_df, update_entry=results, csv_path=eval_csv)

        for mem_batch_path in memorized_batches:
            batch_cfg = convert_text_into_cfg(text=os.path.basename(mem_batch_path.replace('.data_batch.json', '')))
            if not equal_dicts(dict_a=ckpt_cfg, dict_b=batch_cfg, ignore_keys=['batch_id']): continue
            data_batch_id = batch_cfg['batch_id']
            ckpt_batch_id = ckpt_cfg['batch_id']
            if ckpt_batch_id < data_batch_id: continue # TODO: ignore batches_id not seen by the checkpoints
            _ckpt_cfg = ckpt_cfg.copy()
            _ckpt_cfg['data_batch_id'] = data_batch_id
            _ckpt_cfg['eval_batch_size'] = eval_batch_size if eval_batch_size is not None else 'nil'
            if check_for_configuration(src_df=mem_curve_df, cfg=_ckpt_cfg): continue
            with open(mem_batch_path) as f:
                batch_data = json.load(f)
            results, debug_predictions = evaluate_single_batch_memo(
                model_path=model_path,
                batch_data=batch_data,
                batch_size=eval_batch_size,
                train_custom_tokenizer=train_custom_tokenizer
            )

            print(f"batch accuracy: {results['accuracy']}")
            if results['accuracy'] > .7:
                debug_file = f"{os.path.join(debug_dir, convert_cfg_into_text(_ckpt_cfg))}.json"
                if not os.path.exists(debug_file):
                    with open(debug_file, 'w') as f:
                        json.dump(debug_predictions, f, indent=4)

            _ckpt_cfg.update(results)
            # model_memorization_curve.append(ckpt_cfg)
            mem_curve_df = update_df_list(
                df_list=mem_curve_df, 
                update_entry=_ckpt_cfg, 
                csv_path=mem_curve_eval_csv
            )

    




import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--data_dir', default='original_training_data/new_sample/mini') #samples')
# parser.add_argument('--data_dir', default='training_data/new_sample/scaling') #samples')
# parser.add_argument('--models_dir', default='models_scale_1k')
parser.add_argument('--train_custom_tokenizer', default=False)
parser.add_argument('--models_dir', default='models_scale_pad_testing_6_unfoldv2_correct1d')
parser.add_argument('--seeds', default=[42])
parser.add_argument('--batch_size', default=2)
parser.add_argument('--eval_batch_size', default=2)
# parser.add_argument('--train_csv', default='memo_trained_scale_1k.csv')
# parser.add_argument('--eval_csv', default='memo_ppl_train_eval_scale_1k.csv')
# parser.add_argument('--mem_curve_eval_csv', default='mem_curve_eval_scale_1k.csv')
parser.add_argument('--train_csv', default='memo_trained_scale_pad_testing_6_unfoldv2_correct1d.csv')
parser.add_argument('--eval_csv', default='memo_ppl_train_eval_scale_pad_testing_6_unfoldv2_correct1d.csv')
parser.add_argument('--mem_curve_eval_csv', default='mem_curve_eval_scale_pad_testing_6_unfoldv2_correct1d.csv')
# parser.add_argument('--')




if __name__ == "__main__":
    print(os.getcwd())
    params = parser.parse_args()
    experimental_management(params)


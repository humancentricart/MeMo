from transformers.tokenization_utils import (
    PreTrainedTokenizer, 
    TensorType, 
    BatchEncoding, 
    PaddingStrategy, 
    TruncationStrategy,
    TextInput,
    PreTokenizedInput,
    EncodedInput,
    TextInputPair,
    PreTokenizedInputPair,
    EncodedInputPair
)
from transformers import GPTNeoXTokenizerFast
import transformers
import json
import os

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union



class MeMoTokenizer(GPTNeoXTokenizerFast):
    truncation_side = 'left'
    model_input_names: List[str] = ["input_ids", "token_type_ids"]
    
    def set_max_length(self, max_length):
        self.max_length = max_length + 1
        return self

    def set_head_number(self, head_number):
        self.head_number = head_number
        return self
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        trust_remote_code=False,
        max_length: int = None,
        head_number: int = 4, #New!
        **kwargs,
    ):
        
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs
        ).set_max_length(max_length).set_head_number(head_number)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print(f'Setting pad token and pad token id = {tokenizer.pad_token}, {tokenizer.pad_token_id}')
        return tokenizer
        
    
    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]] = None,
        text_pair: Union[str, List[str], List[List[str]]] = None,
        text_target: Union[str, List[str], List[List[str]]] = None,
        text_pair_target: Union[str, List[str], List[List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = 'max_length',
        truncation: Union[bool, str, TruncationStrategy] = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Union[str, transformers.utils.generic.TensorType] = 'pt',
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = False,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs) -> BatchEncoding:
        
        if max_length is None and (truncation == True or truncation == 'longest_first'):
            max_length = self.max_length
        
        return super().__call__(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs
        )

    # TODO remove
    def encode(self, text: Union[str, List[str], List[List[str]]] = None, 
               padding='max_length', truncation=True, max_length=None):
        batch_input_ids = self.__call__(text, padding=padding, truncation=truncation, max_length=max_length)
        
        input_ids = batch_input_ids['input_ids'][..., :-1]
        labels = batch_input_ids['input_ids'][..., 1:]

        return input_ids, labels


    def get_memo_input(self, batch_input_ids):
        memo_input = {}
        for i in range(self.head_number):
            end = batch_input_ids['input_ids'].shape[-1] - i
            start = max(0, end - self.max_length)
            
            if start > 0 and i == 0:
                print(f"Truncation enabled input = {batch_input_ids['input_ids'].shape} vs {self.max_length}")

            input_ids = batch_input_ids['input_ids'][..., start:end]
            #labels = batch_input_ids['input_ids'][..., start+1:end]
            
            if input_ids.shape[-1] != self.max_length:
                input_ids = self.pad({'input_ids':input_ids}, max_length=self.max_length, padding='max_length')
                input_ids = input_ids['input_ids']
            
            memo_input[i] = {'input_ids': input_ids[..., :-1], 
                             'labels': input_ids[..., 1:]}
        
        return memo_input
    
    def memo_heads_encode(self, text: Union[str, List[str], List[List[str]]] = None, 
               padding='max_length', truncation=True, max_length=None):
        
        batch_input_ids = self.__call__(text, padding=padding, truncation=truncation, max_length=max_length)
        memo_input = self.get_memo_input(batch_input_ids)
        return memo_input


    def get_text_batch_encoding(self, text: Union[str, List[str], List[List[str]]] = None, 
               padding='max_length', truncation=True, max_length=None):
        
        batch_input_ids = self.__call__(text, padding='longest', truncation='do_not_truncate', max_length=None)
        batch_input_ids = self.pad(batch_input_ids, pad_to_multiple_of=self.max_length)

        for k in batch_input_ids:
            n_text = batch_input_ids[k].shape[0]
            new_seq = batch_input_ids[k].shape[1] // self.max_length
            
            batch_input_ids[k] = batch_input_ids[k].reshape(n_text * new_seq, self.max_length)
    
            # Identify rows that are not all zeros (only padding)
            non_zero_mask = batch_input_ids[k].abs().sum(dim=1) != 0
            # Filter rows using the mask
            batch_input_ids[k] = batch_input_ids[k][non_zero_mask]


        memo_input = self.get_memo_input(batch_input_ids)

        return memo_input
    
    
    # TODO remove
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        os.makedirs(f"{save_directory}/encoder/tokenizer", exist_ok=True)
        files = self._tokenizer.model.save(f"{save_directory}/encoder/tokenizer", name=filename_prefix)
        return tuple(files)

    # TODO remove
    def input_token_ids(self, text, return_tensors=None): #TODO overwrite the __call__ method instead
        input_ids = self.__call__(text, return_tensors=return_tensors, return_attention_mask=False)['input_ids']
        return input_ids
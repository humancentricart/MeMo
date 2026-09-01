from transformers import PreTrainedTokenizerBase
import torch

from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union


class MeMoTokenizerBase(PreTrainedTokenizerBase):

    truncation_side = "left"

    def configure(
        self,
        truncation_side="left",
        padding_side="left",
        model_max_length=None,
    ):
        self.truncation_side = truncation_side
        self.padding_side = padding_side

        if model_max_length is not None:
            self.model_max_length = model_max_length + 1

        return self

    def get_memo_input(self, input_ids):
        input_ids = input_ids["input_ids"]

        return {
            "input_ids": input_ids[..., :-1],
            "labels": input_ids[..., 1:],
        }

    def encode(
        self,
        text,
        padding="max_length",
        truncation=True,
        max_length=None,
    ):
        batch_input_ids = self(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors='pt'
        )

        return self.get_memo_input(batch_input_ids)

    def get_text_batch_encoding(
        self,
        text,
        padding="max_length",
        truncation=True,
        max_length=None,
    ):

        batch_input_ids = self(
            text,
            padding="longest",
            truncation="do_not_truncate",
            max_length=None,
            return_tensors='pt'
        )

        batch_input_ids = self.pad(
            batch_input_ids,
            pad_to_multiple_of=self.model_max_length,
        )

        for k in batch_input_ids:

            n_text = batch_input_ids[k].shape[0]

            new_seq = (
                batch_input_ids[k].shape[1]
                // self.model_max_length
            )

            batch_input_ids[k] = batch_input_ids[k].reshape(
                n_text * new_seq,
                self.model_max_length,
            )

            non_zero_mask = (
                batch_input_ids[k].abs().sum(dim=1) != 0
            )

            batch_input_ids[k] = batch_input_ids[k][
                non_zero_mask
            ]

        return self.get_memo_input(batch_input_ids)

    def get_text_batch_encoding_for_loss(self, text: Union[str, List[str], List[List[str]]] = None, max_length=None):
        if max_length is None: max_length = self.model_max_length

        batch_input_ids = self.__call__(text, padding='longest', truncation='do_not_truncate', return_tensors='pt', max_length=None)
        longest_length = batch_input_ids['input_ids'].shape[1]
        batch_input_ids = self.pad(batch_input_ids, pad_to_multiple_of=max_length)

        # Identify rows that are not all zeros (only padding)
        non_zero_mask = None #batch_input_ids['input_ids'].abs().sum(dim=1) != 0

        for k in ['input_ids', 'token_type_ids']:#batch_input_ids:
            n_text = batch_input_ids[k].shape[0]
            new_seq = batch_input_ids[k].shape[1] // max_length
            
            batch_input_ids[k] = batch_input_ids[k].reshape(n_text * new_seq, self.model_max_length)

            if k == 'input_ids':
                non_zero_mask = batch_input_ids['input_ids'].abs().sum(dim=1) != 0
    
            # Filter rows using the mask
            if non_zero_mask is not None:
                batch_input_ids[k] = batch_input_ids[k][non_zero_mask]

        batch_input_ids = self.pad(batch_input_ids, pad_to_multiple_of=max_length+longest_length)

        memo_input = self.get_memo_input(batch_input_ids)
        batch_encoding = memo_input

        # prepare inputs for computing the loss of the full-lenght sentence, which consists into the following steps:
        # 1. extend inputs' length of [max_length] padding tokens, to obtain a [max_length]+max_batch_length tensor
        # 2. extend labels' length to match the length of the inputs', with -100 as IDs for the padding tokens we added
        input_ids, labels = batch_encoding['input_ids'], batch_encoding['labels']
        pad_masking = (labels == self.pad_token_id).type(torch.int) #torch.ones(labels == tokenizer.pad_token_id, dtype=torch.long, device=self.memo.device)
        pad_masking[:, -1] = 0 # ensuring that EOS is not considered as padding, even if pad_tok_id == eos_tok_id
        inverse_pad_masking = (pad_masking == 0).type(torch.int) #torch.ones(pad_masking == 0, dtype=torch.long, device=self.memo.device)
        labels = pad_masking * -100 + inverse_pad_masking * labels

        return dict(
            input_ids=input_ids,
            labels=labels
        )


class TrainedMeMoTokenizer(
    MeMoTokenizerBase,
    PreTrainedTokenizerFast,
):

    @classmethod
    def train(
        cls,
        data,
        truncation_side="left",
        padding_side="left",
        model_max_length=None,
        vocab_size=10_000,
        batch_size=1_000,
    ):

        # ------------------------------------------
        # Define special tokens
        # ------------------------------------------

        special_tokens = [
            "<UNK>",
            "<PAD>",
            "<BOS>",
            "<SEP>",
            "<MASK>"
        ]

        # ------------------------------------------
        # Define tokenizer
        # ------------------------------------------

        base_tokenizer = Tokenizer(
            BPE(
                unk_token="[UNK]"
            )
        )

        base_tokenizer.pre_tokenizer = Whitespace()

        # ------------------------------------------
        # Training configuration
        # ------------------------------------------

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        # ------------------------------------------
        # Dataset iterator
        # ------------------------------------------

        def batch_iterator():

            for i in range(
                0,
                len(data),
                batch_size,
            ):
                yield data[
                    i:i + batch_size
                ]["text"]

        # ------------------------------------------
        # Train tokenizer
        # ------------------------------------------

        base_tokenizer.train_from_iterator(
            batch_iterator(),
            trainer=trainer,
        )

        # ------------------------------------------
        # Wrap as HF tokenizer
        # ------------------------------------------

        tokenizer = cls(
            tokenizer_object=base_tokenizer,
            unk_token="<UNK>",
            pad_token="<PAD>",
            bos_token="<BOS>",
            sep_token="<SEP>",
            mask_token="<MASK>"
        )

        tokenizer.configure(
            truncation_side=truncation_side,
            padding_side=padding_side,
            model_max_length=model_max_length,
        )

        return tokenizer
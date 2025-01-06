import copy
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union


import torch
from torch import Tensor
from torch.nn import functional as F, init, Module, ModuleList
from torch.nn.parameter import Parameter


from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging
from transformers.generation import GenerationMixin


logger = logging.get_logger(__name__)

from .modelling_memo_embedding import MeMoEmbedding
from .modelling_memo_layer import MeMoLayer
from .modelling_memo_configuration import MeMoConfig
from .modelling_memo_exception import MeMoException

import math

VERBOSE = False
#DEVICE = 'cpu'

from dataclasses import dataclass
from transformers.utils import ModelOutput


@dataclass
class MeMoModelOutputWithPast(ModelOutput):
    #TODO cambia descrizione
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_token_representation: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_tokens: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class MeMoCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_tokens: Optional[Tuple[torch.FloatTensor, ...]] = None



class MeMoLayers(ModuleList):
    def _init_weights(self, module):
        pass
    def reset_parameters(self):
        pass



class MeMoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MeMoConfig
    #load_tf_weights = load_tf_weights_in_gpt_neo
    base_model_prefix = "memo" ## TODO?
    supports_gradient_checkpointing = False
    _no_split_modules = ["MeMoLayer"] ## TODO?
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False # TODO
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = False  # TODO: needs a HybridCache

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights. Recursevely called by post_init on each of the child module"""
        module.reset_parameters() 
    
    def reset_parameters(self):
        pass
        

class MeMo(MeMoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self._build_model(
            inner_dim=config.hidden_size, 
            num_of_heads=config.num_attention_heads,
            num_of_layers=config.num_hidden_layers, 
            chunk_length=config.chunk_length, 
            num_embeddings=config.vocab_size,
            padding_idx=config.pad_token_id,
            init_weights=False, ## disable the initialization of weights from the constructor (done in the post_init)
        )
        
        self.gradient_checkpointing = False
        
        # Initialize weights and apply final processing --> init_weights --> _init_weights
        self.post_init() 

    
    def _build_model(self, inner_dim, num_of_heads, num_of_layers, chunk_length, 
                 num_embeddings, padding_idx=0, init_weights=True): #, device=None):
        #super().__init__()
        
        self.d = inner_dim
        self.h = num_of_heads
        self.l = num_of_layers
        self.max_len = self.h**self.l
        self.chunk_length = chunk_length
        
        if self.chunk_length/self.max_len != self.chunk_length//self.max_len:
            raise MeMoException("Chunk length "+ str(self.chunk_length) + \
                " should be divisible for number of heads power numer of layers ("+str(self.max_len) +")")
        
        self.encoder = MeMoEmbedding(num_embeddings, self.d, padding_idx=padding_idx, init_weights=init_weights)
        self.layers = MeMoLayers([MeMoLayer(self.d, self.h, init_weights=init_weights) for _ in range(num_of_layers)])

        
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
        #attention_mask: Optional[torch.Tensor] = None,
        #token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        #head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_token: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple[torch.Tensor], MeMoModelOutputWithPast]: # TODO cambia in BaseModelOutputWithPastAndCrossAttentions
        
        return self.retrieve(
            input_ids=input_ids,
            past_key_values=past_key_values,
            #attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            position_ids=position_ids,
            #head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_token=output_hidden_token,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )
        

    # The most simple implementation
    # Input sequence has a shape of (self.h**self.l,self.d), that is self.h sequences are proposed as input rows
    
    
    def memorize(self, input_ids, labels_ids):
        input_sequence = self.encoder.encode(input_ids)
        output_symbols = self.encoder.encode(labels_ids)
        #print("input_sequence.shape", input_sequence.shape)

        (batch_size, current_length, d) = input_sequence.shape
        assert (current_length == self.chunk_length), f'check tokenization of input text, expected row of {self.chunk_length} tokens'
        
        last_layer = self.layers[self.l-1]
        current_length = self.chunk_length
        
        for layer_level in range(self.l):
            current_length = current_length//self.h
            input_sequence = input_sequence.reshape((batch_size, current_length, self.h, self.d))
        
            #print(f"per layer {layer_level} input_sequence.shape", input_sequence.shape)
            #print(input_sequence[0])    
            output_symbols = output_symbols[:, [(x+1)*self.h-1 for x in range(0,current_length)]] ## the output symbol is always the same tokem?

            ## update the input sequence for the next layer
            input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].memorize(input_sequence, 
                                                                                                output_symbols, 
                                                                                                is_last=(layer_level == self.l-1))
            last_layer.directly_memorize(seq_encoding_for_the_last_layer)

    
    def memorize_text(self, memo_input):
        for i in range(0, self.h):
            self.memorize(memo_input[i]['input_ids'].to(self.device), 
                          memo_input[i]['labels'].to(self.device))
        
    
    def forget(self, input_ids, labels_ids, completely=False):
        input_sequence =  self.encoder.encode(input_ids)
        output_symbols = self.encoder.encode(labels_ids)

        (batch_size, current_length, d) = input_sequence.shape
        assert (current_length == self.chunk_length), f'check tokenization of input text, expected row of {self.chunk_length} tokens'
        
        last_layer = self.layers[self.l-1]
        current_length = self.chunk_length
        
        for layer_level in range(self.l):
            current_length = current_length//self.h
            input_sequence = input_sequence.reshape((batch_size, current_length, self.h, self.d))
        
            #print(f"per layer {layer_level} input_sequence.shape", input_sequence.shape)
            #print(input_sequence[0])    
            output_symbols = output_symbols[:, [(x+1)*self.h-1 for x in range(0, current_length)]]
    
            ## update the input sequence for the next layer
            input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].forget(input_sequence, 
                                                                                              output_symbols, 
                                                                                              completely=completely,
                                                                                              is_last=(layer_level == self.l-1))
            
            last_layer.directly_forget(seq_encoding_for_the_last_layer)
        
        
    
    def forget_text(self, memo_input, completely=False):
        for i in range(0,self.h):
            self.forget(memo_input[i]['input_ids'].to(self.device),
                        memo_input[i]['labels'].to(self.device), 
                        completely=completely)

    
    def retrieve(self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
        #attention_mask: Optional[torch.Tensor] = None,
        #token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        #head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_token: Optional[bool] = None, #output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple[torch.Tensor], MeMoModelOutputWithPast]:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.encoder(input_ids)
        
        seq_length = inputs_embeds.shape[0]

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device)
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        
        sequence_representation = inputs_embeds

        (batch_size, current_length, d) = sequence_representation.shape
        assert (current_length == self.chunk_length), f'check tokenization of input text, expected row of {self.chunk_length} tokens'
        
        encoding_for_the_last_layer = torch.zeros((batch_size, self.d)).to(self.device)
        current_length = self.chunk_length #min(self.chunk_length, self.max_len)

        # moved outside the logic for tokenization, here only assertiion above
        #if len(input_sequence) > current_length:
        #    input_sequence = input_sequence[len(input_sequence)-current_length:len(input_sequence)]

        next_decoder_cache = None
        all_hidden_tokens = () if output_hidden_token else None # token representation
        all_hidden_states = () if output_hidden_states else None # sequence representation

        
        for layer_level in range(self.l):
            current_length = int(current_length/self.h)
            sequence_representation = sequence_representation.reshape((batch_size, current_length, self.h, self.d))

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (sequence_representation,)


            outputs = self.layers[layer_level].retrieve(
                sequence_representation,
                layer_past=past_key_values,
                #attention_mask=causal_mask,
                #head_mask=head_mask[i],
                use_cache=use_cache,
                output_hidden_token=output_hidden_token,
                cache_position=cache_position
            )
            
            sequence_representation, seq_encoding_for_the_last_layer = outputs['sequence_encoding'], outputs['token_encoding']
            
            encoding_for_the_last_layer += seq_encoding_for_the_last_layer


            if use_cache:
                next_decoder_cache = outputs['cache']
            if output_hidden_token:
                all_hidden_tokens = all_hidden_tokens + seq_encoding_for_the_last_layer
                
            # TODO check and add back
            #if VERBOSE:
            #    retreived_output_symbol_vector, score_max = self.encoder.decode(last_layer.directly_retrieve(seq_encoding_for_the_last_layer).unsqueeze(0))
            #    print(f"NORM OF THE VECTOR:", torch.linalg.norm(seq_encoding_for_the_last_layer))
            #    print((retreived_output_symbol_vector, score_max))


        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (sequence_representation,)

        next_cache = next_decoder_cache if use_cache else None

        last_layer = self.layers[self.l-1]
        last_token_representation = last_layer.directly_retrieve(encoding_for_the_last_layer)
        
        ## the old decode step should be in the ForCausalLM pass only (and here one perform the retri)
        #retreived_output_symbol_vector, score_max = self.encoder.decode(last_token_representation)
        #return retreived_output_symbol_vector, score_max

        
        if not return_dict:
            return tuple(
                v for v in [last_token_representation, next_cache, all_hidden_states, all_hidden_tokens] if v is not None
            )
        
        return MeMoModelOutputWithPast(
            last_token_representation=last_token_representation,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            hidden_tokens=all_hidden_tokens,
        )

        

class MeMoForCausalLM(MeMoPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.memo = MeMo(config)
        self.lm_head = self.memo.encoder # same embedding and un-embedding matrix
        
        # Initialize weights and apply final processing
        self.post_init()

        

    def forget_text(self, memo_input, completely=True):
        return self.memo.forget_text(
            memo_input=memo_input,
            completely=completely
        )
    
    def forget(self, input_ids, labels_ids, completely=True):
        return self.memo.forget(
            input_ids=input_ids,    
            labels_ids=labels_ids,
            completely=completely
        )

        
    def memorize_text(self, memo_input):
        return self.memo.memorize_text(
            memo_input=memo_input
        )

            
    def memorize(self, input_ids, labels_ids):
        return self.memo.memorize_text(
            input_ids=input_ids,
            labels_ids=labels_ids
        )

        
    def retrieve(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
        #attention_mask: Optional[torch.Tensor] = None,
        #token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        #head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_token: Optional[bool] = None, #output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.memo.retrieve(
            input_ids=input_ids,
            past_key_values=past_key_values,
            #attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            position_ids=position_ids,
            #head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_token=output_hidden_token,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )

        last_token_representation = outputs['last_token_representation']
        
        #the greedy decode step
        #retrieved_output_symbol_vector, score_max = self.lm_head.decode(last_token_representation)
        #return retrieved_output_symbol_vector, score_max
    
        lm_logits = self.lm_head.lm_logits(last_token_representation)
        loss = None
        
        if not return_dict:
            outputs = (lm_logits,) + outputs[1:]
            return ((loss,) + outputs) if loss is not None else outputs
            

        return MeMoCausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            hidden_tokens=outputs.hidden_tokens,
        )

    # TODO remove, should not be here, added just for checks and complaiance for the original implementation
    def greedy_retrieve(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
        #attention_mask: Optional[torch.Tensor] = None,
        #token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        #head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_token: Optional[bool] = None, #output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        output = self.retrieve(
            input_ids=input_ids,
            past_key_values=past_key_values,
            #attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            position_ids=position_ids,
            #head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_token=output_hidden_token,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )


        pred = torch.max(output.logits, dim=-1) # prediction on the last dimension (from pred in vocab_size to max)

        # max similarity index and value of max similarity
        return pred.indices, pred.values

    

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
        #attention_mask: Optional[torch.Tensor] = None,
        #token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        #head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_token: Optional[bool] = None, #output_attentions
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Optional[Union[Tuple[torch.Tensor], MeMoCausalLMOutputWithPast]]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if labels is not None:
            if not self.training:
                logger.warning_once(
                    "`using forward method with labels but model is in eval mode. Setting model.train() and calling model.memorize"
                )
                self.train()
            
            return self.memorize(
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels
            )
        
        return self.retrieve(
            input_ids=input_ids,
            past_key_values=past_key_values,
            #attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            position_ids=position_ids,
            #head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_token=output_hidden_token,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )


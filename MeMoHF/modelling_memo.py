import copy
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import functional as F, init, Module, ModuleList
from torch.nn.parameter import Parameter


from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging
from transformers.generation import GenerationMixin
# from .gen_utils import GenerationMixin


logger = logging.get_logger(__name__)

from .modelling_memo_embedding import MeMoEmbedding
from .modelling_memo_layer import MeMoLayer, CompositionOp
from .modelling_memo_configuration import MeMoConfig
from .modelling_memo_exception import MeMoException
from .utils import (
    windowed_sequence, 
    restore_windowed_sequence_outputs,
    MemoForCausalLMLoss
)

import math

VERBOSE = False
#DEVICE = 'cpu'
DEBUGGING = False

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
    def _initialize_weights(self, module):
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


    def _initialize_weights(self, module):
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
            padding_seq_idx=config.padding_seq_idx, 
            init_weights=False, ## disable the initialization of weights from the constructor (done in the post_init)
            
            alpha_gen=config.alpha_gen,
            compositionOp=CompositionOp.Prod if config.compositionOp=='prod' else CompositionOp.JLT, #CompositionOp.Prod
            padding_vector_component_values=config.padding_vector_component_values if config.padding_vector_component_values is not None else 0
        )
        
        self.gradient_checkpointing = False
        
        # Initialize weights and apply final processing --> init_weights --> _init_weights
        self.post_init() 

    
    def _build_model(self, 
                inner_dim, 
                num_of_heads, 
                num_of_layers, 
                chunk_length, 
                num_embeddings, 
                padding_idx=0,
                padding_seq_idx=-1,  
                init_weights=True,

                alpha_gen=1,
                layerized_CMM_OUT = True,
                compositionOp=CompositionOp.Prod,
                padding_vector_component_values=0,
                lambda_val=0.9,
        ): #, device=None):
        #super().__init__()
        
        self.d = inner_dim
        self.h = num_of_heads
        self.l = num_of_layers
        self.max_len = self.h**self.l
        self.chunk_length = chunk_length
        self.layerized_CMM_OUT = layerized_CMM_OUT
        self.lambda_val = lambda_val

        self.padding_vector_component_values = padding_vector_component_values
        
        # if self.chunk_length/self.max_len != self.chunk_length//self.max_len:
        #     raise MeMoException("Chunk length "+ str(self.chunk_length) + \
        #         " should be divisible for number of heads power numer of layers ("+str(self.max_len) +")")
        
        #self.encoder = MeMoEmbedding(num_embeddings, self.d, padding_idx=padding_idx, init_weights=init_weights)
        #### FMZ 2026-07-01 - Trying with different encodings for input and for unencoding 
        self.encoder = MeMoEmbedding(num_embeddings, self.d, padding_idx=padding_idx, padding_seq_idx=padding_seq_idx, 
                                     init_weights=init_weights, padding_vector_component_values=self.padding_vector_component_values)        #### FMZ 2026-07-01 - Encoding in
        self.output_encoder = MeMoEmbedding(num_embeddings, self.d, padding_idx=padding_idx, padding_seq_idx=padding_seq_idx, 
                                            init_weights=init_weights, padding_vector_component_values=0) #### FMZ 2026-07-01 - Encoding out ## if !=0 accuracy drop
        self.layers = MeMoLayers(
            [
                MeMoLayer(self.d, self.h, init_weights=init_weights, alpha=alpha_gen, compositionOp=compositionOp, layerized_CMM_OUT=self.layerized_CMM_OUT, is_last=(i+1==num_of_layers)) 
                for i in range(num_of_layers)
            ]
        )

    def generate_sequences(self, input_seqs, layer: int):
        h = self.h
        batch_size, seq_len, hidden_dim = input_seqs.shape
        device = input_seqs.device

        # Distance between selected positions
        step = h ** layer

        # Construct:
        #
        # layer=0, h=4
        #
        # [-1, -1, -1,  0]
        # [-1, -1,  0,  1]
        # [-1,  0,  1,  2]
        # [ 0,  1,  2,  3]
        # layer=1, h=4
        # [-1, -1, -1,  0],
        # [-1, -1, -1,  1],
        # [-1, -1, -1,  2],
        # [-1, -1, -1,  3],
        # [-1, -1,  0,  4],
        # [-1, -1,  1,  5],
        # [-1, -1,  2,  6],
        # [-1, -1,  3,  7],
        # [-1,  0,  4,  8],
        # ...
        positions = torch.arange(seq_len, device=device)

        offsets = (
            torch.arange(h, device=device) - (h - 1)
        ) * step

        whento = positions[:, None] + offsets[None, :]

        # Everything before the sequence becomes PAD = -1
        whento = whento.clamp(min=-1)
        if DEBUGGING:
            print("whento (selected indexes for layer ", layer, "):", whento)

        # Keep track of padding positions
        valid = whento != -1

        # Replace -1 with 0 temporarily for indexing
        gather_indices = whento.clamp(min=0)

        # [batch, seq_len, h, hidden_dim]
        new_seqs = input_seqs[:, gather_indices]

        # Padding embedding
        padding = self.encoder.weight[
            self.encoder.padding_seq_idx
        ].view(1, 1, 1, hidden_dim)

        # Replace invalid gathered values with padding
        new_seqs = torch.where(
            valid[None, :, :, None],
            new_seqs,
            padding
        )

        return new_seqs
        
    
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
        # print("=== EMBEDDING DEBUG ===")
        # print("input shape:", input_ids.shape)
        # print("input dtype:", input_ids.dtype)
        # print("input device:", input_ids.device)
        
        # print("embedding shape:", self.encoder.weight.shape)
        # print("embedding device:", self.encoder.weight.device)
        
        # print("input min:", input_ids.min().item())
        # print("input max:", input_ids.max().item())
        
        # print("padding_seq_idx:", self.encoder.padding_seq_idx)
        
        # assert input_ids.dtype == torch.long
        # assert input_ids.min().item() >= 0
        # assert input_ids.max().item() < self.encoder.weight.shape[0]
        
        input_sequence = self.encoder.encode(input_ids)
        # output_symbols = self.encoder.encode(labels_ids)
        output_symbols = self.output_encoder.encode(labels_ids) #### FMZ 2026-07-01
        

        (batch_size, current_length, d) = input_sequence.shape

        # TODO check what are the implication with the new rule for selection
        # if current_length > self.chunk_length: # truncate the sequence considering only the last [chunk_length] tokens
        #     input_sequence = input_sequence[:, -self.chunk_length:, :]
        # (batch_size, current_length, d) = input_sequence.shape
        # assert (current_length == self.chunk_length), f'check tokenization of input text, expected row of {self.chunk_length} tokens'
        
        last_layer = self.layers[self.l-1]
        #current_length = self.chunk_length
        
        #for layer_level in range(self.l):
        #    current_length = current_length//self.h
        #    input_sequence = input_sequence.reshape((batch_size, current_length, self.h, self.d))
        #    #print(f"per layer {layer_level} input_sequence.shape", input_sequence.shape)
        #    #print(input_sequence[0])    
        #    output_symbols = output_symbols[:, [(x+1)*self.h-1 for x in range(0,current_length)]] ## the output symbol is always the same tokem?

        #    ## update the input sequence for the next layer
        #    input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].memorize(input_sequence, 
        #                                                                                        output_symbols, 
        #                                                                                        is_last=(layer_level == self.l-1))
        #    last_layer.directly_memorize(seq_encoding_for_the_last_layer)

        for layer_level in range(self.l):
            input_sequence = self.generate_sequences(input_seqs=input_sequence, layer=layer_level)
            
            if DEBUGGING:
                if layer_level == 0:
                    print("Decoding input_sequence at layer 0")
                    print(self.encoder.decode(input_sequence))
                retreived_output_symbol_vector, max_value = self.output_encoder.decode(output_symbols) ### FMZ 2026-07-01
                print(f"Layer {layer_level} - Output Sequence: {retreived_output_symbol_vector}")
                

            ## update the input sequence for the next layer
            input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].memorize(input_sequence, 
                                                                                                output_symbols, 
                                                                                                is_last=(layer_level == self.l-1))
            last_layer.directly_memorize(seq_encoding_for_the_last_layer)
        
    
    def memorize_text(self, memo_input):
        #for i in range(0, self.h):
        self.memorize(memo_input['input_ids'].to(self.device), 
                      memo_input['labels'].to(self.device))
        
    
    def forget(self, input_sequence_ids, labels_ids, completely=True):
        input_sequence = self.encoder.encode(input_sequence_ids)
        output_symbols = self.output_encoder.encode(labels_ids) #### FMZ 2026-07-01

        (batch_size, current_length, d) = input_sequence.shape
        #assert (current_length == self.chunk_length), f'check tokenization of input text, expected row of {self.chunk_length} tokens'
        
        last_layer = self.layers[self.l-1]
        
        
        for layer_level in range(self.l):
            input_sequence = self.generate_sequences(input_seqs=input_sequence, layer=layer_level)
            ## TODO how to debug now? 
            input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].forget(input_sequence, 
                                                                                                output_symbols, 
                                                                                                completely=completely,
                                                                                                is_last= (layer_level == self.l-1) )
            
            last_layer.directly_forget(seq_encoding_for_the_last_layer)
            
        
        
    
    def forget_text(self, memo_input, completely=True):
        #for i in range(0,self.h):
        self.forget(memo_input['input_ids'].to(self.device),
                    memo_input['labels'].to(self.device), 
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
            ### This is the specific point where padding tokens are added
            inputs_embeds = self.encoder.encode(input_ids)
            #### 
        
        seq_length = inputs_embeds.shape[0]

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device)
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        
        sequence_representation = inputs_embeds

        (batch_size, current_length, d) = sequence_representation.shape
        # if current_length > self.chunk_length: # truncate the sequence considering only the last [chunk_length] tokens
        #     sequence_representation = sequence_representation[:, -self.chunk_length:, :]
        # (batch_size, current_length, d) = sequence_representation.shape
        # assert (current_length == self.chunk_length), f'check tokenization of input text, expected row of {self.chunk_length} tokens'
        
        encoding_for_the_last_layer = torch.zeros((batch_size, self.d)).to(self.device)
        #current_length = self.chunk_length #min(self.chunk_length, self.max_len)
        
        
        if self.layerized_CMM_OUT: 
            residual_stream = torch.zeros((batch_size, self.d)).to(self.device)

        
        # moved outside the logic for tokenization, here only assertiion above
        #if len(input_sequence) > current_length:
        #    input_sequence = input_sequence[len(input_sequence)-current_length:len(input_sequence)]

        next_decoder_cache = None
        all_hidden_tokens = () if output_hidden_token else None # token representation
        all_hidden_states = () if output_hidden_states else None # sequence representation

        
        #for layer_level in range(self.l):
        #    current_length = int(current_length/self.h)
        #    sequence_representation = sequence_representation.reshape((batch_size, current_length, self.h, self.d))

        #    if output_hidden_states:
        #        all_hidden_states = all_hidden_states + (sequence_representation,)


        #    outputs = self.layers[layer_level].retrieve(
        #        sequence_representation,
        #        layer_past=past_key_values,
        #        #attention_mask=causal_mask,
        #        #head_mask=head_mask[i],
        #        use_cache=use_cache,
        #        output_hidden_token=output_hidden_token,
        #        cache_position=cache_position
        #    )
            
        #    sequence_representation, seq_encoding_for_the_last_layer = outputs['sequence_encoding'], outputs['token_encoding']
            
        #    encoding_for_the_last_layer += seq_encoding_for_the_last_layer
        

        #    if use_cache:
        #        next_decoder_cache = outputs['cache']
        #    if output_hidden_token:
        #        all_hidden_tokens = all_hidden_tokens + seq_encoding_for_the_last_layer
        #        
        #    # TODO check and add back
        #    #if VERBOSE:
        #    #    retreived_output_symbol_vector, score_max = self.encoder.decode(last_layer.directly_retrieve(seq_encoding_for_the_last_layer).unsqueeze(0))
        #    #    print(f"NORM OF THE VECTOR:", torch.linalg.norm(seq_encoding_for_the_last_layer))
        #    #    print((retreived_output_symbol_vector, score_max))

        for layer_level in range(self.l):
            sequence_representation = self.generate_sequences(input_seqs=sequence_representation, layer=layer_level)

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
            
            # This is to capture the layer by layer extraction of the next token: the output of each layer is normalized in order to 
            # penalize short sequences 
            # if self.layerized_CMM_OUT: 
            #     residual_stream += outputs['layered_out_token']
            if self.layerized_CMM_OUT: 
                #residual_stream += outputs['layered_out_token']
                # residual_stream = torch.linalg.norm(outputs['layered_out_token'] + self.lambda_val * residual_stream, dim=0, keepdim=True)
                residual_stream = F.normalize(
                    outputs['layered_out_token'] + self.lambda_val * residual_stream,
                    p=2,
                    dim=1
                )
            


        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (sequence_representation,)

        next_cache = next_decoder_cache if use_cache else None

        last_layer = self.layers[self.l-1]
        if self.layerized_CMM_OUT: 
            last_token_representation = residual_stream
        else:
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

# from .loss_utils import ForCausalLMLoss

class MeMoForCausalLM(MeMoPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.memo = MeMo(config)
        #self.lm_head = self.memo.encoder # same embedding and un-embedding matrix
        self.lm_head = self.memo.output_encoder # FMZ 2026-07-01 different embedding and un-embedding matrix
        # self.loss_function = ForCausalLMLoss
        # Initialize weights and apply final processing
        self.post_init()
        self.loss_function = MemoForCausalLMLoss 

        

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
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_token: Optional[bool] = None, #output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        compute_loss: Optional[bool] = False):

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
        loss = None # TODO: compute the loss function
        # if labels is not None:
        #     # default loss: transformers.loss.loss_utility.ForCausalLMLoss
        #     loss = self.loss_function(logits=lm_logits, labels=labels[:, -1:], vocab_size=self.config.vocab_size)#, **kwargs)

            
        
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
        compute_loss: Optional[bool] = False
    ) -> Optional[Union[Tuple[torch.Tensor], MeMoCausalLMOutputWithPast]]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if labels is not None and not compute_loss:
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
            labels=labels,
            use_cache=use_cache,
            output_hidden_token=output_hidden_token,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )
    
    # def forward_with_loss(
    #     self,
    #     batch_inputs,
    #     return_dict: Optional[bool] = None,
    #     # tokenizer = None
    #     compute_accuracy=False,
    # ) -> Optional[Union[Tuple[torch.Tensor], MeMoCausalLMOutputWithPast]]:

    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     # batch_encoding = tokenizer.get_text_batch_encoding_for_loss(text=text_batch)
    #     input_ids, labels = batch_inputs['input_ids'].to(self.memo.device), batch_inputs['labels'].to(self.memo.device)

    #     logits_list = list()
    #     outputs = None 
    #     lm_logits = None
    #     for i in range(self.memo.chunk_length, labels.shape[1]):
    #         if outputs is not None:
    #             del outputs
    #             torch.cuda.empty_cache()
    #         current_batch = dict(
    #             input_ids=input_ids[:, i-self.memo.chunk_length:i],
    #             labels=labels[:, i-self.memo.chunk_length:i]
    #         )
    #         outputs = self.forward(
    #             input_ids=current_batch['input_ids'],
    #             labels=current_batch['labels'],
    #             return_dict=return_dict,
    #             compute_loss=True
    #         )
    #         logits = outputs.logits 
    #         if lm_logits is None:
    #             lm_logits = logits
    #         else: 
    #             lm_logits = torch.cat([lm_logits, logits], dim=1)
    #         del logits
    #         del current_batch
    #         # logits_list.append(logits)
        
    #     # lm_logits = torch.cat(logits_list, dim=1)
    #     _labels = labels[:, -lm_logits.shape[1]:].contiguous().to(self.memo.device)
    #     loss = self.loss_function(logits=lm_logits, labels=_labels, vocab_size=self.config.vocab_size, shift_labels=_labels)
    #     argmax = torch.argmax(lm_logits, dim=-1)

    #     return MeMoCausalLMOutputWithPast(
    #         loss=loss,
    #         logits=lm_logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         hidden_tokens=outputs.hidden_tokens,
    #     )


    def forward_with_loss(
        self,
        batch_inputs,
        return_dict: Optional[bool] = None,
        # tokenizer = None
        compute_accuracy=False,
        starting_point=9, #2
        tokenizer=None
    ) -> Optional[Union[Tuple[torch.Tensor], MeMoCausalLMOutputWithPast]]:

        starting_point = max(2, starting_point) 
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # batch_encoding = tokenizer.get_text_batch_encoding_for_loss(text=text_batch)
        input_ids, labels = batch_inputs['input_ids'].to(self.memo.device), batch_inputs['labels'].to(self.memo.device)

        logits_list = list()
        outputs = None 
        lm_logits = None

        tot_correct_tokens = None 
        total_tokens = None

        total_loss = None
        total_nll_sum = 0.0  # Sum of negative log likelihoods (for perplexity)
        num_tokens_predicted = 0  # Total number of tokens predicted

        # Per-token statistics: {token_id: {'target_count': int, 'correct_count': int}}
        token_stats = {}
        
        # Track padding tokens for masking analysis
        pad_token_id = getattr(self.config, 'pad_token_id', 0)
        padding_token_analysis = {
            'padding_tokens_masked': 0,      # Padding tokens with label == -100
            'padding_tokens_not_masked': 0,  # Padding tokens with label != -100
            'padding_tokens_correct': 0      # Padding tokens that were correctly predicted (if not masked)
        }

        debug_predictions = list()

        if self.memo.chunk_length+starting_point >= labels.shape[1]:
            return None, None, None 

        for i in range(self.memo.chunk_length+starting_point, labels.shape[1]):
            if outputs is not None:
                del outputs
                torch.cuda.empty_cache()
            current_batch = dict(
                input_ids=input_ids[:, i-self.memo.chunk_length:i],
                # labels=labels[:, i-self.memo.chunk_length:i]
                labels=labels[:, i-1:i]
            )
            outputs = self.forward(
                input_ids=current_batch['input_ids'],
                labels=current_batch['labels'],
                return_dict=return_dict,
                compute_loss=True
            )
            logits = outputs.logits 
            # if lm_logits is None:
            #     lm_logits = logits
            # else: 
            #     lm_logits = torch.cat([lm_logits, logits], dim=1)
            # del logits
            # del current_batch
            # logits_list.append(logits)

            # lm_logits = (logits + 10) * 1000 # scale up logits to make them more confident when applying the softmax
            lm_logits = logits.detach()  # Detach logits to prevent gradients from flowing back through them during loss computation

            # mask out all the logits whose scores are outside the 50 best tokens (top-k filtering), in order to consider only the top-k tokens for the loss computation using the softmax 
            top_k = torch.topk(lm_logits, k=50, dim=-1)
            top_k_indices = top_k.indices
            top_k_values = top_k.values
            mask = torch.full_like(lm_logits, -12.0) #float('-inf'))
            mask.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
            lm_logits = mask * 10 #lm_logits + mask

        
            # lm_logits = torch.cat(logits_list, dim=1)
            # _labels = labels[:, -lm_logits.shape[1]:].contiguous().to(self.memo.device)
            # loss = self.loss_function(logits=lm_logits, labels=_labels, vocab_size=self.config.vocab_size, shift_labels=_labels)
            # argmax = torch.argmax(lm_logits, dim=-1)
            _labels = current_batch['labels'].contiguous().to(self.memo.device)
            loss = self.loss_function(logits=lm_logits, labels=_labels, vocab_size=self.config.vocab_size)#, shift_labels=_labels)
            
            ## For debugging and analysis: compute argmax and softmax values for the current batch
            argmax = torch.argmax(lm_logits, dim=-1)
            argmax_value = torch.max(lm_logits, dim=-1)
            top_k = torch.topk(lm_logits, k=5, dim=-1)
            
            _lm_logits_softmax = torch.nn.functional.softmax(lm_logits, dim=-1)
            argmax_soft = torch.argmax(_lm_logits_softmax, dim=-1)
            argmax_value_soft = torch.max(_lm_logits_softmax, dim=-1)
            top_k_softmax = torch.topk(_lm_logits_softmax, k=5, dim=-1)

            # get the probability of the expected label declared in _labels from _lm_logit_softmax
            expected_label_prob = torch.gather(_lm_logits_softmax, dim=-1, index=_labels.unsqueeze(-1)).squeeze(-1)
            expected_label_score = torch.gather(lm_logits, dim=-1, index=_labels.unsqueeze(-1)).squeeze(-1)
            # batch_debug_info = dict()
            if tokenizer is not None:
                # decode the predicted token ids and expected token ids for the current batch
                pred_tokens = tokenizer.batch_decode(argmax)
                expected_tokens = tokenizer.batch_decode(_labels)
                top_k_tokens_list = top_k.indices.reshape(top_k.indices.shape[0], -1).cpu().numpy().tolist()
                top_k_predicted_tokens = [
                    tokenizer.convert_ids_to_tokens(_top_k_tokens_list)
                    for _top_k_tokens_list in top_k_tokens_list
                ]
                top_k_scores = top_k.values.reshape(top_k.values.shape[0], -1)
                top_k_softmax_scores = top_k_softmax.values.reshape(top_k_softmax.values.shape[0], -1)
                batch_debug_info = {
                    'sequence_index': i,
                    'input_sequence': tokenizer.batch_decode(current_batch['input_ids'], skip_special_tokens=True),
                    'pred_tokens': pred_tokens,
                    'pred_score': argmax_value.values.cpu().numpy().tolist(),
                    'pred_score_softmax': argmax_value_soft.values.cpu().numpy().tolist(),
                    'expected_tokens': expected_tokens,
                    'expected_label_prob': expected_label_prob.cpu().numpy().tolist(),
                    'expected_label_score': expected_label_score.cpu().numpy().tolist(),
                    'top_predicted_tokens': top_k_predicted_tokens,
                    'top_predicted_token_scores': top_k_scores.cpu().numpy().tolist(),
                    'top_predicted_tokens_softmax': top_k_softmax_scores.cpu().numpy().tolist(),

                    'vocab_distribution_score_sum': lm_logits.sum(dim=-1).cpu().numpy().tolist(),
                    'vocab_distribution_score_mean': lm_logits.mean(dim=-1).cpu().numpy().tolist(),
                    'vocab_distribution_score_sum_softmax': _lm_logits_softmax.sum(dim=-1).cpu().numpy().tolist(),
                    'vocab_distribution_score_mean_softmax': _lm_logits_softmax.mean(dim=-1).cpu().numpy().tolist(),
                    'vocab_distrib_min_score': lm_logits.min(dim=-1).values.cpu().numpy().tolist(),
                    'vocab_distrib_min_score_softmax': _lm_logits_softmax.min(dim=-1).values.cpu().numpy().tolist(),

                    'exp_vocab_distribution_score_sum': torch.exp(lm_logits).sum(dim=-1).cpu().numpy().tolist(),
                    'exp_vocab_distribution_score_mean': torch.exp(lm_logits).mean(dim=-1).cpu().numpy().tolist(),
                    'exp_vocab_distrib_min_score': torch.exp(lm_logits).min(dim=-1).values.cpu().numpy().tolist(),
                    'exp_vocab_distrib_max_score': torch.exp(lm_logits).max(dim=-1).values.cpu().numpy().tolist(),

                    'loss': loss.detach().cpu().numpy().tolist(),
                }
                # check all the fields of the batch_debug_info and convert inf or -inf to a string for better readability in the logs
                # for key, value in batch_debug_info.items():
                #     if isinstance(value, list):
                #         batch_debug_info[key] = [v if not (isinstance(v, float) and (v == float('inf') or v == float('-inf'))) else str(v) for v in value]
                #     elif isinstance(value, float) and (value == float('inf') or value == float('-inf')):
                #         batch_debug_info[key] = str(value)
                for key, value in batch_debug_info.items():
                    if isinstance(value, list):
                        if isinstance(value[0], list):
                            batch_debug_info[key] = [[str(v) if str(v) in ['inf', '-inf'] else v for v in sublist] for sublist in value]
                        else:
                            batch_debug_info[key] = [str(v) if str(v) in ['inf', '-inf'] else v for v in value]
                    elif isinstance(value, float) and (str(value) in ['inf', '-inf']):
                        batch_debug_info[key] = str(value)
                debug_predictions.append(batch_debug_info)
                # if the loss is infinte for the current batch, print the debug information
                # if str(loss.detach().cpu().numpy().item()) in ['inf', '-inf']:
                #     print(f"Batch index: {i}")
                #     print(batch_debug_info)

                # print(f"Predicted tokens: {pred_tokens}")
                # print(f"Expected tokens: {expected_tokens}")
                # print(f"Expected token probabilities: {expected_label_prob}")
                # print(f"Top 5 predicted tokens: {tokenizer.batch_decode(top_k.indices)}")
                # print(f"Top 5 predicted token probabilities: {top_k.values}")


            batch_size = _labels.shape[0]
            num_valid_tokens_in_batch = (_labels != -100).sum().item()
            
            # Accumulate loss: assuming loss_function returns mean loss across valid tokens
            # Convert mean to sum by multiplying by number of valid tokens
            if total_loss is None:
                total_loss = loss * num_valid_tokens_in_batch
            else:
                total_loss += loss * num_valid_tokens_in_batch
            
            # Accumulate for perplexity: loss is already mean NLL per token
            # Multiply by number of valid tokens to get sum of NLL for this batch
            total_nll_sum += loss.detach() * num_valid_tokens_in_batch
            num_tokens_predicted += num_valid_tokens_in_batch

            if compute_accuracy:
                # argmax for selecting most probable labels
                pred = torch.max(lm_logits, dim=-1)
                p_indices, p_values = pred.indices, pred.values
                
                # create bitmask for correctly predicted labels
                correct_tokens = (p_indices == _labels).type(torch.int)
                
                # Analyze per-token statistics (before masking -100 tokens)
                valid_mask = _labels != -100  # Tokens that are not masked
                
                # Flatten tensors for per-token analysis
                flat_labels = _labels.flatten()
                flat_correct = correct_tokens.flatten()
                flat_valid = valid_mask.flatten()
                
                for token_id in torch.unique(flat_labels):
                    token_id = token_id.item()
                    if token_id == -100:
                        continue
                    
                    # Find all occurrences of this token
                    token_mask = (flat_labels == token_id)
                    
                    if token_id not in token_stats:
                        token_stats[token_id] = {'target_count': 0, 'correct_count': 0}
                    
                    # Count how many times this token appears as target
                    token_count = torch.sum(token_mask).item()
                    token_stats[token_id]['target_count'] += token_count
                    
                    # Count how many times it was correctly predicted
                    correct_for_token = torch.sum(flat_correct[token_mask]).item()
                    token_stats[token_id]['correct_count'] += correct_for_token
                
                # Analyze padding tokens
                if pad_token_id is not None:
                    padding_mask = (flat_labels == pad_token_id)
                    masked_padding = torch.sum((flat_labels == pad_token_id) & (_labels.flatten() == -100)).item()
                    not_masked_padding = torch.sum((flat_labels == pad_token_id) & (_labels.flatten() != -100)).item()
                    
                    padding_token_analysis['padding_tokens_masked'] += masked_padding
                    padding_token_analysis['padding_tokens_not_masked'] += not_masked_padding
                    
                    # Check if any unmasked padding tokens were correctly predicted
                    if not_masked_padding > 0:
                        unmasked_padding_correct = torch.sum(
                            flat_correct[padding_mask & flat_valid]
                        ).item()
                        padding_token_analysis['padding_tokens_correct'] += unmasked_padding_correct
                
                # set bitmask entries to 0 for -100 tokens
                correct_tokens[_labels == -100] = 0
                correct_tokens = torch.sum(correct_tokens)
                if tot_correct_tokens is None:
                    tot_correct_tokens = correct_tokens
                else:
                    tot_correct_tokens += correct_tokens

                # count how many tokens != -100 in labels
                tot_tokens = torch.sum((_labels != -100).type(torch.int))
                if total_tokens is None:
                    total_tokens = tot_tokens
                else:
                    total_tokens += tot_tokens
            del current_batch
            del lm_logits 
            del logits 

        # # convert token_stats to list of dicts
        # token_stats = [
        #     {
        #         'token_id': token_id,
        #         'target_count': stats['target_count'],
        #         'correct_count': stats['correct_count'],
        #         #'accuracy': (stats['correct_count'] / stats['target_count']) if stats['target_count'] > 0 else 0.0
        #     }
        #     for token_id, stats in token_stats.items()
        # ]
        # # create two list of dictionaries from token_stats, one sorted by correct_count and one sorted by accuracy, and keep the top max_token_distrib_rank tokens for each list
        # token_stats_by_correct = sorted(token_stats, key=lambda x: x['correct_count'], reverse=True)[:max_token_distrib_rank]
        # token_stats_by_target = sorted(token_stats, key=lambda x: x['correct_count']/x['target_count'], reverse=True)[:max_token_distrib_rank]
        
        # compute accuracy, and return dictionary with these fields
        accuracy_results = dict(
            accuracy=(tot_correct_tokens/total_tokens).detach().cpu().item(),
            correct_tokens=tot_correct_tokens.detach().cpu().item(),
            tot_tokens=total_tokens.detach().cpu().item(),
            token_stats=token_stats,
            padding_analysis=padding_token_analysis['padding_tokens_correct'],
        ) if compute_accuracy else None
        
        # Compute perplexity: exp(average NLL)
        # average NLL = total_nll_sum / num_tokens_predicted
        if num_tokens_predicted > 0:
            avg_nll = total_nll_sum / num_tokens_predicted
            perplexity_value = torch.exp(avg_nll).detach().cpu().item()
        else:
            perplexity_value = 0.0
        
        # Add perplexity to the results
        if accuracy_results is not None:
            accuracy_results['perplexity'] = perplexity_value
            accuracy_results['avg_nll'] = (total_nll_sum / num_tokens_predicted).detach().cpu().item() if num_tokens_predicted > 0 else 0.0
            accuracy_results['num_tokens'] = num_tokens_predicted
        else:
            accuracy_results = dict(
                perplexity=perplexity_value,
                avg_nll=(total_nll_sum / num_tokens_predicted).detach().cpu().item() if num_tokens_predicted > 0 else 0.0,
                num_tokens=num_tokens_predicted
            )
        
        batch_debug = dict(
            ppl=accuracy_results['perplexity'],
            avg_nll=accuracy_results['avg_nll'],
            num_tokens=accuracy_results['num_tokens'],
            accuracy=accuracy_results['accuracy'] if compute_accuracy else None,
            correct_tokens=accuracy_results['correct_tokens'] if compute_accuracy else None,
            total_tokens=accuracy_results['tot_tokens'] if compute_accuracy else None,
            predictions=debug_predictions
        )

        return MeMoCausalLMOutputWithPast(
            loss=total_loss,
            logits=None, #lm_logits,
            past_key_values=None, #outputs.past_key_values,
            hidden_states=None, #outputs.hidden_states,
            hidden_tokens=None, #outputs.hidden_tokens,
        ), accuracy_results, batch_debug
    
    


    def forward_with_loss_unfold_v1( ####### ESR implmented and tested
            self,
            batch_inputs,
            return_dict: Optional[bool] = None,
            compute_accuracy=False,
            starting_point=9,
            tokenizer=None,
            window_batch_size: int = 16,  # NEW: number of sliding-window positions batched into a single forward() call
        ) -> Optional[Union[Tuple[torch.Tensor], "MeMoCausalLMOutputWithPast"]]:
        """
        Parallelized version of forward_with_loss.
    
        The original implementation called self.forward() once per target token
        position i, each time re-running the full chunk_length-token window through
        the model. That's an embarrassingly parallel workload (each window is an
        independent input -- `outputs` is deleted every iteration and nothing like
        past_key_values or memory state is threaded between calls), so instead of
        looping token-by-token, this groups `window_batch_size` windows together
        along the batch dimension and does one forward() call per group.
    
        IMPORTANT: this is only equivalent to the original loop if self.forward()
        (and whatever self.memo does internally) is a pure function of
        (input_ids, labels) for a given window -- i.e. it does not mutate any
        persistent/recurrent state that depends on windows being visited in strict
        left-to-right order. Please double check that assumption against your
        `self.memo` implementation before trusting these results in place of the
        original loop.
    
        This does NOT reduce total FLOPs (each window still reprocesses
        chunk_length tokens from scratch, exactly like before) -- it reduces the
        number of separate forward() calls, trading many tiny serialized calls for
        fewer, larger batched ones. That's usually where the real wall-clock win
        comes from (kernel-launch overhead, GPU underutilization from batch size 1,
        per-call autograd graph setup). Tune `window_batch_size` for your GPU
        memory budget: larger = fewer calls = faster, until you run out of memory.
        """
    
        starting_point = max(2, starting_point)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        input_ids, labels = batch_inputs['input_ids'].to(self.memo.device), batch_inputs['labels'].to(self.memo.device)
    
        chunk_length = self.memo.chunk_length
        batch_size, seq_len = input_ids.shape
    
        start = chunk_length + starting_point
        if start >= seq_len:
            return None, None, None
    
        # ---- Build every sliding window up front -----------------------------------
        # windows_all[:, j, :] == input_ids[:, j : j + chunk_length]
        windows_all = input_ids.unfold(dimension=1, size=chunk_length, step=1)  # (B, W_total, chunk_length)
    
        w_lo = starting_point           # j such that i = j + chunk_length == start
        w_hi = seq_len - chunk_length   # j such that i = j + chunk_length == seq_len (exclusive upper bound)
        windows = windows_all[:, w_lo:w_hi, :]                 # (B, num_windows, chunk_length)
    
        label_lo = start - 1
        label_hi = seq_len - 1
        window_labels = labels[:, label_lo:label_hi]           # (B, num_windows), same as labels[:, i-1] for each i
    
        num_windows = window_labels.shape[1]
        assert num_windows == windows.shape[1]
    
        # original loop variable i, per window index j (0-based within `windows`)
        window_i_values = list(range(start, seq_len))
    
        tot_correct_tokens = None
        total_tokens = None
    
        total_loss = None
        total_nll_sum = 0.0
        num_tokens_predicted = 0
    
        # Per-token statistics: {token_id: {'target_count': int, 'correct_count': int}}
        token_stats = {}
    
        pad_token_id = getattr(self.config, 'pad_token_id', 0)
        padding_token_analysis = {
            'padding_tokens_masked': 0,
            'padding_tokens_not_masked': 0,
            'padding_tokens_correct': 0
        }
    
        debug_predictions = list()
    
        for w_start in range(0, num_windows, window_batch_size):
            w_end = min(w_start + window_batch_size, num_windows)
            w = w_end - w_start
    
            cur_windows = windows[:, w_start:w_end, :]     # (B, w, chunk_length)
            cur_labels = window_labels[:, w_start:w_end]    # (B, w)
    
            # fold (batch, window) into one big batch dimension for a single forward() call
            # NOTE: reshape order is row-major, so flat index = b * w + j -- kept consistent
            # below whenever we need to regroup back to (batch, window).
            flat_input_ids = cur_windows.reshape(batch_size * w, chunk_length)
            flat_labels = cur_labels.reshape(batch_size * w, 1)
    
            outputs = self.forward(
                input_ids=flat_input_ids,
                labels=flat_labels,
                return_dict=return_dict,
                compute_loss=True
            )
            logits = outputs.logits
            lm_logits = logits.detach()  # detach to prevent gradients flowing back through this copy
    
            # top-k filtering (same as original): keep only the top-50 logits per position
            top_k = torch.topk(lm_logits, k=50, dim=-1)
            mask = torch.full_like(lm_logits, -12.0)
            mask.scatter_(dim=-1, index=top_k.indices, src=top_k.values)
            lm_logits = mask * 10
    
            _labels = flat_labels.contiguous().to(self.memo.device)
            loss = self.loss_function(logits=lm_logits, labels=_labels, vocab_size=self.config.vocab_size)
    
            argmax = torch.argmax(lm_logits, dim=-1)
            argmax_value = torch.max(lm_logits, dim=-1)
            top_k5 = torch.topk(lm_logits, k=5, dim=-1)
    
            _lm_logits_softmax = torch.nn.functional.softmax(lm_logits, dim=-1)
            argmax_value_soft = torch.max(_lm_logits_softmax, dim=-1)
            top_k_softmax = torch.topk(_lm_logits_softmax, k=5, dim=-1)
    
            expected_label_prob = torch.gather(_lm_logits_softmax, dim=-1, index=_labels.unsqueeze(-1)).squeeze(-1)
            expected_label_score = torch.gather(lm_logits, dim=-1, index=_labels.unsqueeze(-1)).squeeze(-1)
    
            if tokenizer is not None:
                # Reshape everything back to (B, w, ...) so we can emit one debug dict
                # per original position i, in the same shape the un-parallelized loop produced.
                pred_tokens_flat = tokenizer.batch_decode(argmax)
                expected_tokens_flat = tokenizer.batch_decode(_labels)
                input_seq_flat = tokenizer.batch_decode(flat_input_ids, skip_special_tokens=True)
    
                def _regroup(flat_list):
                    return [flat_list[b * w:(b + 1) * w] for b in range(batch_size)]
    
                pred_tokens_grouped = _regroup(pred_tokens_flat)
                expected_tokens_grouped = _regroup(expected_tokens_flat)
                input_seq_grouped = _regroup(input_seq_flat)
    
                pred_score = argmax_value.values.reshape(batch_size, w).cpu().numpy().tolist()
                pred_score_softmax = argmax_value_soft.values.reshape(batch_size, w).cpu().numpy().tolist()
                expected_label_prob_g = expected_label_prob.reshape(batch_size, w).cpu().numpy().tolist()
                expected_label_score_g = expected_label_score.reshape(batch_size, w).cpu().numpy().tolist()
                loss_scalar = loss.detach().cpu().numpy().tolist()  # mean loss over this whole micro-batch
    
                top_k_tokens_list = top_k5.indices.reshape(batch_size, w, -1).cpu().numpy().tolist()
                top_k_scores = top_k5.values.reshape(batch_size, w, -1).cpu().numpy().tolist()
                top_k_softmax_scores = top_k_softmax.values.reshape(batch_size, w, -1).cpu().numpy().tolist()
    
                vocab_sum = lm_logits.reshape(batch_size, w, -1).sum(dim=-1).cpu().numpy().tolist()
                vocab_mean = lm_logits.reshape(batch_size, w, -1).mean(dim=-1).cpu().numpy().tolist()
                vocab_sum_soft = _lm_logits_softmax.reshape(batch_size, w, -1).sum(dim=-1).cpu().numpy().tolist()
                vocab_mean_soft = _lm_logits_softmax.reshape(batch_size, w, -1).mean(dim=-1).cpu().numpy().tolist()
                vocab_min = lm_logits.reshape(batch_size, w, -1).min(dim=-1).values.cpu().numpy().tolist()
                vocab_min_soft = _lm_logits_softmax.reshape(batch_size, w, -1).min(dim=-1).values.cpu().numpy().tolist()
    
                exp_lm_logits = torch.exp(lm_logits).reshape(batch_size, w, -1)
                exp_sum = exp_lm_logits.sum(dim=-1).cpu().numpy().tolist()
                exp_mean = exp_lm_logits.mean(dim=-1).cpu().numpy().tolist()
                exp_min = exp_lm_logits.min(dim=-1).values.cpu().numpy().tolist()
                exp_max = exp_lm_logits.max(dim=-1).values.cpu().numpy().tolist()
    
                for local_j in range(w):
                    i_value = window_i_values[w_start + local_j]
                    batch_debug_info = {
                        'sequence_index': i_value,
                        'input_sequence': [input_seq_grouped[b][local_j] for b in range(batch_size)],
                        'pred_tokens': [pred_tokens_grouped[b][local_j] for b in range(batch_size)],
                        'pred_score': [pred_score[b][local_j] for b in range(batch_size)],
                        'pred_score_softmax': [pred_score_softmax[b][local_j] for b in range(batch_size)],
                        'expected_tokens': [expected_tokens_grouped[b][local_j] for b in range(batch_size)],
                        'expected_label_prob': [expected_label_prob_g[b][local_j] for b in range(batch_size)],
                        'expected_label_score': [expected_label_score_g[b][local_j] for b in range(batch_size)],
                        'top_predicted_tokens': [
                            tokenizer.convert_ids_to_tokens(top_k_tokens_list[b][local_j])
                            for b in range(batch_size)
                        ],
                        'top_predicted_token_scores': [top_k_scores[b][local_j] for b in range(batch_size)],
                        'top_predicted_tokens_softmax': [top_k_softmax_scores[b][local_j] for b in range(batch_size)],
    
                        'vocab_distribution_score_sum': [vocab_sum[b][local_j] for b in range(batch_size)],
                        'vocab_distribution_score_mean': [vocab_mean[b][local_j] for b in range(batch_size)],
                        'vocab_distribution_score_sum_softmax': [vocab_sum_soft[b][local_j] for b in range(batch_size)],
                        'vocab_distribution_score_mean_softmax': [vocab_mean_soft[b][local_j] for b in range(batch_size)],
                        'vocab_distrib_min_score': [vocab_min[b][local_j] for b in range(batch_size)],
                        'vocab_distrib_min_score_softmax': [vocab_min_soft[b][local_j] for b in range(batch_size)],
    
                        'exp_vocab_distribution_score_sum': [exp_sum[b][local_j] for b in range(batch_size)],
                        'exp_vocab_distribution_score_mean': [exp_mean[b][local_j] for b in range(batch_size)],
                        'exp_vocab_distrib_min_score': [exp_min[b][local_j] for b in range(batch_size)],
                        'exp_vocab_distrib_max_score': [exp_max[b][local_j] for b in range(batch_size)],
    
                        # NOTE: unlike the original (which had a true per-position loss),
                        # this is the mean loss over the whole window_batch_size micro-batch,
                        # since loss is now computed once per group rather than once per position.
                        'loss': loss_scalar,
                    }
                    for key, value in batch_debug_info.items():
                        if isinstance(value, list):
                            if len(value) > 0 and isinstance(value[0], list):
                                batch_debug_info[key] = [[str(v) if str(v) in ['inf', '-inf'] else v for v in sublist] for sublist in value]
                            else:
                                batch_debug_info[key] = [str(v) if str(v) in ['inf', '-inf'] else v for v in value]
                        elif isinstance(value, float) and (str(value) in ['inf', '-inf']):
                            batch_debug_info[key] = str(value)
                    debug_predictions.append(batch_debug_info)
    
            num_valid_tokens_in_batch = (_labels != -100).sum().item()
    
            if total_loss is None:
                total_loss = loss * num_valid_tokens_in_batch
            else:
                total_loss += loss * num_valid_tokens_in_batch
    
            total_nll_sum += loss.detach() * num_valid_tokens_in_batch
            num_tokens_predicted += num_valid_tokens_in_batch
    
            if compute_accuracy:
                pred = torch.max(lm_logits, dim=-1)
                p_indices = pred.indices
    
                correct_tokens = (p_indices == _labels).type(torch.int)
                valid_mask = _labels != -100
    
                flat_labels_t = _labels.flatten()
                flat_correct = correct_tokens.flatten()
                flat_valid = valid_mask.flatten()
    
                for token_id in torch.unique(flat_labels_t):
                    token_id = token_id.item()
                    if token_id == -100:
                        continue
                    token_mask = (flat_labels_t == token_id)
                    if token_id not in token_stats:
                        token_stats[token_id] = {'target_count': 0, 'correct_count': 0}
                    token_count = torch.sum(token_mask).item()
                    token_stats[token_id]['target_count'] += token_count
                    correct_for_token = torch.sum(flat_correct[token_mask]).item()
                    token_stats[token_id]['correct_count'] += correct_for_token
    
                if pad_token_id is not None:
                    padding_mask = (flat_labels_t == pad_token_id)
                    masked_padding = torch.sum((flat_labels_t == pad_token_id) & (_labels.flatten() == -100)).item()
                    not_masked_padding = torch.sum((flat_labels_t == pad_token_id) & (_labels.flatten() != -100)).item()
    
                    padding_token_analysis['padding_tokens_masked'] += masked_padding
                    padding_token_analysis['padding_tokens_not_masked'] += not_masked_padding
    
                    if not_masked_padding > 0:
                        unmasked_padding_correct = torch.sum(flat_correct[padding_mask & flat_valid]).item()
                        padding_token_analysis['padding_tokens_correct'] += unmasked_padding_correct
    
                correct_tokens[_labels == -100] = 0
                correct_tokens = torch.sum(correct_tokens)
                tot_correct_tokens = correct_tokens if tot_correct_tokens is None else tot_correct_tokens + correct_tokens
    
                tot_tokens = torch.sum((_labels != -100).type(torch.int))
                total_tokens = tot_tokens if total_tokens is None else total_tokens + tot_tokens
    
            del outputs, logits, lm_logits, flat_input_ids, flat_labels, _labels
            torch.cuda.empty_cache()
    
        accuracy_results = dict(
            accuracy=(tot_correct_tokens / total_tokens).detach().cpu().item(),
            correct_tokens=tot_correct_tokens.detach().cpu().item(),
            tot_tokens=total_tokens.detach().cpu().item(),
            token_stats=token_stats,
            padding_analysis=padding_token_analysis['padding_tokens_correct'],
        ) if compute_accuracy else None
    
        if num_tokens_predicted > 0:
            avg_nll = total_nll_sum / num_tokens_predicted
            perplexity_value = torch.exp(avg_nll).detach().cpu().item()
        else:
            perplexity_value = 0.0
    
        if accuracy_results is not None:
            accuracy_results['perplexity'] = perplexity_value
            accuracy_results['avg_nll'] = (total_nll_sum / num_tokens_predicted).detach().cpu().item() if num_tokens_predicted > 0 else 0.0
            accuracy_results['num_tokens'] = num_tokens_predicted
        else:
            accuracy_results = dict(
                perplexity=perplexity_value,
                avg_nll=(total_nll_sum / num_tokens_predicted).detach().cpu().item() if num_tokens_predicted > 0 else 0.0,
                num_tokens=num_tokens_predicted
            )
    
        batch_debug = dict(
            ppl=accuracy_results['perplexity'],
            avg_nll=accuracy_results['avg_nll'],
            num_tokens=accuracy_results['num_tokens'],
            accuracy=accuracy_results['accuracy'] if compute_accuracy else None,
            correct_tokens=accuracy_results['correct_tokens'] if compute_accuracy else None,
            total_tokens=accuracy_results['tot_tokens'] if compute_accuracy else None,
            predictions=debug_predictions
        )
    
        return MeMoCausalLMOutputWithPast(
            loss=total_loss,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            hidden_tokens=None,
        ), accuracy_results, batch_debug


    def forward_with_loss_unfold(
        self,
        batch_inputs,
        return_dict: Optional[bool] = None,
        compute_accuracy=False,
        starting_point=9,
        tokenizer=None,
        window_batch_size: int = 16,  # number of sliding-window positions batched into a single forward() call
    ) -> Optional[Union[Tuple[torch.Tensor], "MeMoCausalLMOutputWithPast"]]:
        """
        Parallelized + cleaned-up version of forward_with_loss.
     
     
        Stats computation
        ------------------
        The accuracy / token_stats / padding-analysis / perplexity / total_loss
        bookkeeping produces bit-for-bit the same outputs as the original, but:
          - top-1 predictions are computed once per micro-batch (via a single
            torch.max call) instead of twice (once via argmax, once again inside
            the accuracy block).
          - per-token target/correct counts are computed with torch.bincount
            instead of a Python loop over torch.unique(...) + per-token boolean
            masking, which is the dominant cost when many distinct tokens appear
            per micro-batch.
          - softmax / top-5 / expected-label tensors are only computed when a
            tokenizer is supplied (they're exclusively used for the debug dict),
            instead of unconditionally every iteration.
          - avg_nll is computed once instead of twice, and no tensor is mutated
            in place after another view of it has already been used (the
            original mutated `correct_tokens` in place after `flat_correct`,
            a view of the same storage, had already been read elsewhere).
     
        This does NOT reduce total FLOPs of the forward passes themselves (each
        window still reprocesses chunk_length tokens from scratch) -- it reduces
        the number of separate forward() calls and removes redundant/unused
        tensor math around them. Tune `window_batch_size` for your GPU memory
        budget: larger = fewer calls = faster, until you run out of memory.
        """
     
        starting_point = max(2, starting_point)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
     
        input_ids, labels = batch_inputs['input_ids'].to(self.memo.device), batch_inputs['labels'].to(self.memo.device)
     
        chunk_length = self.memo.chunk_length
        vocab_size = self.config.vocab_size
        batch_size, seq_len = input_ids.shape
     
        start = chunk_length + starting_point
        if start >= seq_len:
            return None, None, None
     
        # ---- Build every sliding window up front -----------------------------------
        # windows_all[:, j, :] == input_ids[:, j : j + chunk_length]
        windows_all = input_ids.unfold(dimension=1, size=chunk_length, step=1)  # (B, W_total, chunk_length)
     
        w_lo = starting_point           # j such that i = j + chunk_length == start
        w_hi = seq_len - chunk_length   # j such that i = j + chunk_length == seq_len (exclusive upper bound)
        windows = windows_all[:, w_lo:w_hi, :]                 # (B, num_windows, chunk_length)
     
        label_lo = start - 1
        label_hi = seq_len - 1
        window_labels = labels[:, label_lo:label_hi]           # (B, num_windows), same as labels[:, i-1] for each i
     
        num_windows = window_labels.shape[1]
        assert num_windows == windows.shape[1]
     
        # original loop variable i, per window index j (0-based within `windows`)
        window_i_values = list(range(start, seq_len))
     
        total_loss = None
        total_nll_sum = 0.0
        num_tokens_predicted = 0
     
        tot_correct_tokens = None
        total_tokens = None
     
        pad_token_id = getattr(self.config, 'pad_token_id', 0)
        padding_token_analysis = {
            'padding_tokens_masked': 0,
            'padding_tokens_not_masked': 0,
            'padding_tokens_correct': 0
        }
     
        # Vectorized per-token accumulators (replaces the Python-level token_stats dict
        # that was updated with a torch.unique(...) loop every iteration).
        if compute_accuracy:
            target_counts_total = torch.zeros(vocab_size, dtype=torch.long, device=self.memo.device)
            correct_counts_total = torch.zeros(vocab_size, dtype=torch.long, device=self.memo.device)
     
        debug_predictions = list()
        need_top1 = compute_accuracy or (tokenizer is not None)
     
        for w_start in range(0, num_windows, window_batch_size):
            w_end = min(w_start + window_batch_size, num_windows)
            w = w_end - w_start
     
            cur_windows = windows[:, w_start:w_end, :]     # (B, w, chunk_length)
            cur_labels = window_labels[:, w_start:w_end]    # (B, w)
     
            # fold (batch, window) into one big batch dimension for a single forward() call
            # NOTE: reshape order is row-major, so flat index = b * w + j -- kept consistent
            # below whenever we need to regroup back to (batch, window).
            flat_input_ids = cur_windows.reshape(batch_size * w, chunk_length)
            flat_labels = cur_labels.reshape(batch_size * w, 1)
     
            outputs = self.forward(
                input_ids=flat_input_ids,
                labels=flat_labels,
                return_dict=return_dict,
                compute_loss=True
            )
            logits = outputs.logits
            lm_logits = logits.detach()  # detach to prevent gradients flowing back through this copy
     
            # top-k filtering (same as original): keep only the top-50 logits per position
            top_k = torch.topk(lm_logits, k=50, dim=-1)
            mask = torch.full_like(lm_logits, -12.0)
            mask.scatter_(dim=-1, index=top_k.indices, src=top_k.values)
            lm_logits = mask * 10
     
            _labels = flat_labels.contiguous().to(self.memo.device)
            loss = self.loss_function(logits=lm_logits, labels=_labels, vocab_size=vocab_size)
     
            valid_mask = _labels != -100
            flat_labels_t = _labels.flatten()
            flat_valid = valid_mask.flatten()
     
            # single top-1 reduction, shared by the accuracy block and the debug block
            # (previously computed twice: once via torch.argmax, once via torch.max)
            if need_top1:
                top1 = torch.max(lm_logits, dim=-1)
                argmax = top1.indices
                argmax_value = top1.values
            else:
                argmax = None
                argmax_value = None
     
            if tokenizer is not None:
                # Everything below is exclusively used to build the debug dict, so it's
                # only computed when a tokenizer is actually supplied.
                top_k5 = torch.topk(lm_logits, k=5, dim=-1)
                _lm_logits_softmax = torch.nn.functional.softmax(lm_logits, dim=-1)
                argmax_value_soft = torch.max(_lm_logits_softmax, dim=-1)
                top_k_softmax = torch.topk(_lm_logits_softmax, k=5, dim=-1)
     
                expected_label_prob = torch.gather(_lm_logits_softmax, dim=-1, index=_labels.unsqueeze(-1)).squeeze(-1)
                expected_label_score = torch.gather(lm_logits, dim=-1, index=_labels.unsqueeze(-1)).squeeze(-1)
     
                # Reshape everything back to (B, w, ...) so we can emit one debug dict
                # per original position i, in the same shape the un-parallelized loop produced.
                pred_tokens_flat = tokenizer.batch_decode(argmax)
                expected_tokens_flat = tokenizer.batch_decode(_labels)
                input_seq_flat = tokenizer.batch_decode(flat_input_ids, skip_special_tokens=True)
     
                def _regroup(flat_list):
                    return [flat_list[b * w:(b + 1) * w] for b in range(batch_size)]
     
                pred_tokens_grouped = _regroup(pred_tokens_flat)
                expected_tokens_grouped = _regroup(expected_tokens_flat)
                input_seq_grouped = _regroup(input_seq_flat)
     
                pred_score = argmax_value.reshape(batch_size, w).cpu().numpy().tolist()
                pred_score_softmax = argmax_value_soft.values.reshape(batch_size, w).cpu().numpy().tolist()
                expected_label_prob_g = expected_label_prob.reshape(batch_size, w).cpu().numpy().tolist()
                expected_label_score_g = expected_label_score.reshape(batch_size, w).cpu().numpy().tolist()
                loss_scalar = loss.detach().cpu().numpy().tolist()  # mean loss over this whole micro-batch
     
                top_k_tokens_list = top_k5.indices.reshape(batch_size, w, -1).cpu().numpy().tolist()
                top_k_scores = top_k5.values.reshape(batch_size, w, -1).cpu().numpy().tolist()
                top_k_softmax_scores = top_k_softmax.values.reshape(batch_size, w, -1).cpu().numpy().tolist()
     
                lm_logits_r = lm_logits.reshape(batch_size, w, -1)
                softmax_r = _lm_logits_softmax.reshape(batch_size, w, -1)
                exp_r = torch.exp(lm_logits_r)
     
                vocab_sum = lm_logits_r.sum(dim=-1).cpu().numpy().tolist()
                vocab_mean = lm_logits_r.mean(dim=-1).cpu().numpy().tolist()
                vocab_sum_soft = softmax_r.sum(dim=-1).cpu().numpy().tolist()
                vocab_mean_soft = softmax_r.mean(dim=-1).cpu().numpy().tolist()
                vocab_min = lm_logits_r.min(dim=-1).values.cpu().numpy().tolist()
                vocab_min_soft = softmax_r.min(dim=-1).values.cpu().numpy().tolist()
     
                exp_sum = exp_r.sum(dim=-1).cpu().numpy().tolist()
                exp_mean = exp_r.mean(dim=-1).cpu().numpy().tolist()
                exp_min = exp_r.min(dim=-1).values.cpu().numpy().tolist()
                exp_max = exp_r.max(dim=-1).values.cpu().numpy().tolist()
     
                for local_j in range(w):
                    i_value = window_i_values[w_start + local_j]
                    batch_debug_info = {
                        'sequence_index': i_value,
                        'input_sequence': [input_seq_grouped[b][local_j] for b in range(batch_size)],
                        'pred_tokens': [pred_tokens_grouped[b][local_j] for b in range(batch_size)],
                        'pred_score': [pred_score[b][local_j] for b in range(batch_size)],
                        'pred_score_softmax': [pred_score_softmax[b][local_j] for b in range(batch_size)],
                        'expected_tokens': [expected_tokens_grouped[b][local_j] for b in range(batch_size)],
                        'expected_label_prob': [expected_label_prob_g[b][local_j] for b in range(batch_size)],
                        'expected_label_score': [expected_label_score_g[b][local_j] for b in range(batch_size)],
                        'top_predicted_tokens': [
                            tokenizer.convert_ids_to_tokens(top_k_tokens_list[b][local_j])
                            for b in range(batch_size)
                        ],
                        'top_predicted_token_scores': [top_k_scores[b][local_j] for b in range(batch_size)],
                        'top_predicted_tokens_softmax': [top_k_softmax_scores[b][local_j] for b in range(batch_size)],
     
                        'vocab_distribution_score_sum': [vocab_sum[b][local_j] for b in range(batch_size)],
                        'vocab_distribution_score_mean': [vocab_mean[b][local_j] for b in range(batch_size)],
                        'vocab_distribution_score_sum_softmax': [vocab_sum_soft[b][local_j] for b in range(batch_size)],
                        'vocab_distribution_score_mean_softmax': [vocab_mean_soft[b][local_j] for b in range(batch_size)],
                        'vocab_distrib_min_score': [vocab_min[b][local_j] for b in range(batch_size)],
                        'vocab_distrib_min_score_softmax': [vocab_min_soft[b][local_j] for b in range(batch_size)],
     
                        'exp_vocab_distribution_score_sum': [exp_sum[b][local_j] for b in range(batch_size)],
                        'exp_vocab_distribution_score_mean': [exp_mean[b][local_j] for b in range(batch_size)],
                        'exp_vocab_distrib_min_score': [exp_min[b][local_j] for b in range(batch_size)],
                        'exp_vocab_distrib_max_score': [exp_max[b][local_j] for b in range(batch_size)],
     
                        # NOTE: unlike the original (which had a true per-position loss),
                        # this is the mean loss over the whole window_batch_size micro-batch,
                        # since loss is now computed once per group rather than once per position.
                        'loss': loss_scalar,
                    }
                    for key, value in batch_debug_info.items():
                        if isinstance(value, list):
                            if len(value) > 0 and isinstance(value[0], list):
                                batch_debug_info[key] = [[str(v) if str(v) in ['inf', '-inf'] else v for v in sublist] for sublist in value]
                            else:
                                batch_debug_info[key] = [str(v) if str(v) in ['inf', '-inf'] else v for v in value]
                        elif isinstance(value, float) and (str(value) in ['inf', '-inf']):
                            batch_debug_info[key] = str(value)
                    debug_predictions.append(batch_debug_info)
     
            num_valid_tokens_in_batch = flat_valid.sum().item()
     
            if total_loss is None:
                total_loss = loss * num_valid_tokens_in_batch
            else:
                total_loss += loss * num_valid_tokens_in_batch
     
            total_nll_sum += loss.detach() * num_valid_tokens_in_batch
            num_tokens_predicted += num_valid_tokens_in_batch
     
            if compute_accuracy:
                flat_correct = (argmax == _labels).flatten()  # bool, same shape as flat_labels_t
     
                valid_labels = flat_labels_t[flat_valid]
                valid_correct = flat_correct[flat_valid]
     
                # per-token target/correct counts, vectorized (replaces the
                # torch.unique(...) + per-token masking Python loop)
                target_counts_total += torch.bincount(valid_labels, minlength=vocab_size)
                correct_counts_total += torch.bincount(valid_labels[valid_correct], minlength=vocab_size)
     
                # padding analysis
                is_pad = flat_labels_t == pad_token_id
                masked_padding = torch.sum(is_pad & ~flat_valid).item()
                not_masked_padding = torch.sum(is_pad & flat_valid).item()
                padding_token_analysis['padding_tokens_masked'] += masked_padding
                padding_token_analysis['padding_tokens_not_masked'] += not_masked_padding
                if not_masked_padding > 0:
                    padding_token_analysis['padding_tokens_correct'] += torch.sum(flat_correct & is_pad & flat_valid).item()
     
                correct_incr = valid_correct.sum()
                tot_correct_tokens = correct_incr if tot_correct_tokens is None else tot_correct_tokens + correct_incr
     
                tokens_incr = flat_valid.sum()
                total_tokens = tokens_incr if total_tokens is None else total_tokens + tokens_incr
     
            del outputs, logits, lm_logits, flat_input_ids, flat_labels, _labels
            torch.cuda.empty_cache()
     
        if compute_accuracy:
            nonzero_ids = torch.nonzero(target_counts_total, as_tuple=True)[0].tolist()
            token_stats = {
                tid: {
                    'target_count': target_counts_total[tid].item(),
                    'correct_count': correct_counts_total[tid].item(),
                }
                for tid in nonzero_ids
            }
            accuracy_results = dict(
                accuracy=(tot_correct_tokens / total_tokens).detach().cpu().item(),
                correct_tokens=tot_correct_tokens.detach().cpu().item(),
                tot_tokens=total_tokens.detach().cpu().item(),
                token_stats=token_stats,
                padding_analysis=padding_token_analysis['padding_tokens_correct'],
            )
        else:
            accuracy_results = None
     
        if num_tokens_predicted > 0:
            avg_nll_tensor = total_nll_sum / num_tokens_predicted
            avg_nll_value = avg_nll_tensor.detach().cpu().item()
            perplexity_value = torch.exp(avg_nll_tensor).detach().cpu().item()
        else:
            avg_nll_value = 0.0
            perplexity_value = 0.0
     
        if accuracy_results is None:
            accuracy_results = dict()
        accuracy_results['perplexity'] = perplexity_value
        accuracy_results['avg_nll'] = avg_nll_value
        accuracy_results['num_tokens'] = num_tokens_predicted
     
        batch_debug = dict(
            ppl=accuracy_results['perplexity'],
            avg_nll=accuracy_results['avg_nll'],
            num_tokens=accuracy_results['num_tokens'],
            accuracy=accuracy_results['accuracy'] if compute_accuracy else None,
            correct_tokens=accuracy_results['correct_tokens'] if compute_accuracy else None,
            total_tokens=accuracy_results['tot_tokens'] if compute_accuracy else None,
            predictions=debug_predictions
        )
     
        return MeMoCausalLMOutputWithPast(
            loss=total_loss,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            hidden_tokens=None,
        ), accuracy_results, batch_debug

    
    def forward_with_loss_parallelized(
        self,
        batch_inputs,
        return_dict: Optional[bool] = None,
        compute_accuracy=False,
        # tokenizer = None
        # pad_token_id=0
    ) -> Optional[Union[Tuple[torch.Tensor], MeMoCausalLMOutputWithPast]]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # batch_encoding = tokenizer.get_text_batch_encoding_for_loss(text=text_batch)
        input_ids, labels = batch_inputs['input_ids'].to(self.memo.device), batch_inputs['labels'].to(self.memo.device)

        logits_list = list()
        outputs = None 
        lm_logits = None
        # for i in range(self.memo.chunk_length, labels.shape[1]):
        
        source_ids = windowed_sequence(
            tensor_ids=input_ids,
            window_size=self.memo.chunk_length,
            # hidden_dim=self.memo.config.hi
        )
        # target_ids = windowed_sequence(
        #     tensor_ids=labels[:, -self.memo.chunk_length:],
        #     window_size=1
        # )

        outputs = self.forward(
            input_ids=source_ids,
            # labels=target_ids,
            return_dict=return_dict,
            compute_loss=True
        )

        logits = outputs.logits 
        lm_logits = restore_windowed_sequence_outputs(
            output_ids=logits,
            batch_size=input_ids.shape[0],
            hidden_dim=self.config.vocab_size
        )
            # logits_list.append(logits)
        
        # lm_logits = torch.cat(logits_list, dim=1)
        _labels = labels[:, -lm_logits.shape[1]:].contiguous().to(self.memo.device)#labels
        # _labels[_labels == pad_token_id] = -100 # TODO: manage situations in which EOS is 0; replace 0 with tokenizer.pad_token_id
        loss = self.loss_function(logits=lm_logits, labels=_labels, vocab_size=self.config.vocab_size, shift_labels=_labels)
        # pred = torch.max(lm_logits, dim=-1)
        # p_indices, p_values = pred.indices, pred.values
        accuracy_results = None
        if compute_accuracy:
            # argmax for selecting most probable labels
            pred = torch.max(lm_logits, dim=-1)
            p_indices, p_values = pred.indices, pred.values
            # create bitmask for correctly predicted labels
            correct_tokens = (p_indices == _labels).type(torch.int)
            # set bitmask entries to 0 for -100 tokens
            correct_tokens[_labels == -100] = 0
            correct_tokens = torch.sum(correct_tokens)
            # count how many tokens != -100 in labels
            tot_tokens = torch.sum((_labels != -100).type(torch.int))
            # compute accuracy, and return dictionary with these fields
            accuracy_results = dict(
                accuracy=(correct_tokens/tot_tokens).detach().cpu().item(),
                correct_tokens=correct_tokens.detach().cpu().item(),
                tot_tokens=tot_tokens.detach().cpu().item()
            )




        return MeMoCausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            hidden_tokens=outputs.hidden_tokens,
        ), accuracy_results


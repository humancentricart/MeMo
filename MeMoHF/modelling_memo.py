import copy
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

from tqdm import tqdm

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
DEBUGGING = False
DEBUGGING_PROJECTION = False

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
    #ESR 2026-09-16
    residual_stream_unpacked: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_tokens: Optional[Tuple[torch.FloatTensor, ...]] = None
    raw_recall_unpacked: Optional[torch.FloatTensor] = None


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
    lm_logits_unpacked: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_tokens: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_probs: Optional[Dict[str, torch.Tensor]] = None      # NEW name -> (B, T, V) probabilities
    decoder_log_probs: Optional[Dict[str, torch.Tensor]] = None  # NEW same, log-probabilities (optional)
    decoder_info: Optional[Dict[str, torch.Tensor]] = None       # NEW per-layer weights, load, temperature


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
        
 
# Decoders compared at the model level (see the analysis document, section 3).
# Each one gives, for every layer, logits over the vocabulary from the RAW recall r = M k;
# the per-layer logits are then pooled with the same layer weights as the legacy decoder.
#   legacy         : normalized recall, plain dot product, top-100 zero-masking (the previous behaviour)
#   dot_T1         : u_c . r                                         (raw recall, no mask, T = 1)
#   bayes_T        : beta_th * (u_c . r - 0.5*gamma*|u_c|^2)         theoretical temperature
#   bayes_T_prior  : bayes_T + log pi_c                              prior from the layer's token counts
#   fit_T          : same as bayes_T with the layer's fitted beta_scale (plain, no prior)
#   fit_T_prior    : same as bayes_T_prior with the layer's fitted beta_scale (plain, prior)
# fit_* coincide with bayes_* until MeMoLayer.set_beta_scale(...) is called with the fitted ratio.
DECODER_NAMES = ("dot_T1", "bayes_T", "bayes_T_prior", "fit_T", "fit_T_prior")
 
 
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
            compositionOp = CompositionOp[config.compositionOp.upper()],
            padding_vector_component_values=config.padding_vector_component_values if config.padding_vector_component_values is not None else 0,
            prior_smoothing=getattr(config, "prior_smoothing", 1.0),
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
                layerized_CMM_OUT = True,
                compositionOp=CompositionOp.PROD,
                padding_vector_component_values=0,
                lambda_val=0.9,
                prior_smoothing=1.0,
        ): 
        
        self.d = inner_dim
        self.h = num_of_heads
        self.l = num_of_layers
        self.max_len = self.h**self.l
        self.chunk_length = chunk_length
        self.lambda_val = lambda_val
        self.compositionOp = compositionOp
        self.padding_vector_component_values = padding_vector_component_values
        self.layerized_CMM_OUT = True        
        
        #### FMZ 2026-07-01 - Trying with different encodings for input and for unencoding 
        self.encoder = MeMoEmbedding(num_embeddings, self.d, padding_idx=padding_idx, padding_seq_idx=padding_seq_idx, 
                                     init_weights=init_weights, padding_vector_component_values=self.padding_vector_component_values)        #### FMZ 2026-07-01 - Encoding in
        self.output_encoder = MeMoEmbedding(num_embeddings, self.d, padding_idx=padding_idx, padding_seq_idx=padding_seq_idx, 
                                            init_weights=init_weights, padding_vector_component_values=0) #### FMZ 2026-07-01 - Encoding out ## if !=0 accuracy drop
 
        # every layer keeps the statistics of what it stores (prior = token counts over the
        # num_embeddings possible values, load, key/value norms) used by the Bayes decoders
        def make_layer(op):
            return MeMoLayer(self.d, self.h, init_weights=init_weights, compositionOp=op,
                             num_embeddings=num_embeddings, prior_smoothing=prior_smoothing)
 
        if (self.compositionOp != CompositionOp.PROD and self.compositionOp != CompositionOp.PRODWITHSHUFFLING) or self.l<= 3:
            self.layers = MeMoLayers(
                [
                    make_layer(self.compositionOp) 
                    for i in range(self.l)
                ]
            )
        else:
            ## ESR hardcoded num of layers (3) for now, from exp results
             self.layers = MeMoLayers(
                [
                    make_layer(self.compositionOp) 
                    for i in range(2)
                ] + 
                [
                    make_layer(CompositionOp.SUMTHENPRODWITHSHUFFLING) 
                    for i in range(2, self.l)
                ]
            )
 
    def generate_sequences(self, input_seqs, layer: int):
 
        h = self.h
        batch_size, seq_len, hidden_dim = input_seqs.shape
        device = input_seqs.device
 
 
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
 
        #    [[j * self.h ** (layer) + i for j in range(0, self.h)] for i in range(num_output_symbols)]
        #    [[j * step + i for j in range(0, self.h)] for i in range(num_output_symbols)]
 
        # Distance between selected positions
        step = h ** layer
        positions = torch.arange(seq_len, device=device)
 
        offsets = (
            torch.arange(h, device=device) - (h - 1)
        ) * step
 
        whento = positions[:, None] + offsets[None, :]
 
        # Everything before the sequence becomes PAD = -1
        whento = whento.clamp(min=-1)
        #if DEBUGGING:
        #    print("whento (selected indexes for layer ", layer, "):", whento)
 
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
        # print("Padding:", padding)
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
 
    @torch.no_grad()
    def reset_memory(self):
        """Empty the memory of every layer together with the statistics (prior, load, norms) describing it."""
        for layer in self.layers:
            layer.reset_memory()
        
    
    def memorize(self, input_ids, labels_ids):
        input_sequence = self.encoder.encode(input_ids)
        output_symbols = self.output_encoder.encode(labels_ids) #### FMZ 2026-07-01
        
        for layer_level in range(self.l):
            if DEBUGGING_PROJECTION:
                print("*"*80)
                print(f"Layer {layer_level}")
            input_sequence = self.generate_sequences(input_seqs=input_sequence, layer=layer_level)
            
            if DEBUGGING:
                if layer_level == 0:
                    print("Decoding input_sequence at layer 0")
                    print(self.encoder.decode(input_sequence))
                retreived_output_symbol_vector, max_value = self.output_encoder.decode(output_symbols) ### FMZ 2026-07-01
                print(f"Layer {layer_level} - Output Sequence: {retreived_output_symbol_vector}")
                
 
            ## update the input sequence for the next layer
            start = self.h**layer_level if layer_level != 0 else 0 #### WORKING
            input_sequence, update_at_layer = self.layers[layer_level].memorize(input_sequence, 
                                                                                output_symbols, 
                                                                                to_save_sequences=range(start, input_sequence.shape[1]), ### FMZ 2026-09-24
                                                                                is_last=(layer_level == self.l-1),
                                                                                output_ids=labels_ids) # token id of each output symbol: updates the layer's prior
        
    
    def memorize_text(self, memo_input):
        self.memorize(memo_input['input_ids'].to(self.device), 
                      memo_input['labels'].to(self.device))
 
    
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
 
        (batch_size, _, _) = sequence_representation.shape
        encoding_for_the_last_layer = torch.zeros((batch_size, self.d)).to(self.device)
        
        
        if self.layerized_CMM_OUT: 
            residual_stream = torch.zeros((batch_size, self.chunk_length, self.d)).to(self.device)
            residual_stream_unpacked = torch.zeros((self.l, batch_size, self.chunk_length, self.d)).to(self.device)
            # raw (unnormalized) recall M k of every layer: what the Bayes decoders are calibrated for
            raw_recall_unpacked = torch.zeros((self.l, batch_size, self.chunk_length, self.d)).to(self.device)
        
 
        next_decoder_cache = None
        all_hidden_tokens = () if output_hidden_token else None # token representation
        all_hidden_states = () if output_hidden_states else None # sequence representation
 
 
        for layer_level in range(self.l):
            if DEBUGGING_PROJECTION:
                print("*"*80)
                print(f"Layer {layer_level}")
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
            
            if seq_encoding_for_the_last_layer is not None:
                encoding_for_the_last_layer += seq_encoding_for_the_last_layer
            
            
            if self.layerized_CMM_OUT: 
                residual_stream = F.normalize(
                    outputs['layered_out_token'] + self.lambda_val * residual_stream,
                    p=2,
                    dim=1
                )
                residual_stream_unpacked[layer_level] = outputs['layered_out_token']
                raw_recall_unpacked[layer_level] = outputs['layered_out_raw']
 
 
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (sequence_representation,)
 
        next_cache = next_decoder_cache if use_cache else None
 
        if self.layerized_CMM_OUT: 
            last_token_representation = residual_stream
        
        if not return_dict:
            return tuple(
                v for v in [last_token_representation, next_cache, all_hidden_states, all_hidden_tokens] if v is not None
            )
        
        return MeMoModelOutputWithPast(
            last_token_representation=last_token_representation,
            residual_stream_unpacked=residual_stream_unpacked,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            hidden_tokens=all_hidden_tokens,
            raw_recall_unpacked=raw_recall_unpacked,
        )
 
 
class MeMoForCausalLM(MeMoPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
 
    def __init__(self, config):
        super().__init__(config)
        self.memo = MeMo(config)
        self.lm_head = self.memo.output_encoder # FMZ 2026-07-01 different embedding and un-embedding matrix
        # Initialize weights and apply final processing
        self.post_init()
        self.loss_function = MemoForCausalLMLoss 
 
        
    def memorize_text(self, memo_input):
        return self.memo.memorize_text(
            memo_input=memo_input
        )
 
            
    def memorize(self, input_ids, labels_ids):
        return self.memo.memorize_text(
            input_ids=input_ids,
            labels_ids=labels_ids
        )
 
    @torch.no_grad()
    def reset_memory(self):
        self.memo.reset_memory()
 
    # ------------------------------------------------------------------------------------------
    # Bayes decoders
    # ------------------------------------------------------------------------------------------
    def _candidate_matrix(self):
        """u_c: the vectors the layers store as values for token c. Assumed to be the rows of the
        output embedding (the same table used by lm_head). Override if encode() transforms them."""
        return self.lm_head.weight
 
    def _layer_decoder_logits(self, layer, recall, E, E_sq, names):
        """Logits over the vocabulary of one layer, for each requested decoder.
        recall: (B, T, d) raw recall M k of this layer;  E: (V, d);  E_sq: (V,) = |u_c|^2."""
        dot = torch.matmul(recall, E.t())                       # u_c . r              (B, T, V)
        gamma = layer.mean_key_sq_norm()                        # mean |k|^2 of the stored keys
        S = dot - 0.5 * gamma * E_sq                            # u_c . r - 0.5 gamma |u_c|^2
        log_prior = layer.log_prior() if any(n.endswith("_prior") for n in names) else None
 
        out = {}
        for n in names:
            if n == "dot_T1":
                out[n] = dot
            elif n == "bayes_T":
                out[n] = layer.inverse_temperature(with_prior=False, apply_scale=False) * S
            elif n == "bayes_T_prior":
                out[n] = layer.inverse_temperature(with_prior=True, apply_scale=False) * S + log_prior
            elif n == "fit_T":
                out[n] = layer.inverse_temperature(with_prior=False, apply_scale=True) * S
            elif n == "fit_T_prior":
                out[n] = layer.inverse_temperature(with_prior=True, apply_scale=True) * S + log_prior
        return out
        
        
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
        compute_loss: Optional[bool] = False,
        decoders: Optional[Sequence[str]] = None,
        return_log_probs: Optional[bool] = False,
    ):
        """decoders: which of DECODER_NAMES to compute besides "legacy" (default: all).
        The probability distributions of all decoders are returned in `decoder_probs`
        (name -> (B, T, V)); `logits` is still the legacy decoder's logits."""
 
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
 
        names = tuple(DECODER_NAMES if decoders is None else decoders)
        unknown = [n for n in names if n not in DECODER_NAMES]
        if unknown:
            raise ValueError(f"unknown decoders {unknown}; available: {DECODER_NAMES}")
        
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
 
        # ---------------- legacy decoder (unchanged behaviour) ----------------
        last_token_representation = outputs['residual_stream_unpacked']   # ['last_token_representation'] # 
        lm_logits = self.lm_head.lm_logits(last_token_representation)
        lm_logits_unpacked = lm_logits
 
        num_layers = self.memo.l
        decay = self.memo.lambda_val
 
        # Keep only the top 100 logits for each layer
        k = 100
        topk_values, topk_indices = torch.topk(lm_logits, k=k, dim=-1)
 
        # Create a mask
        mask = torch.zeros_like(lm_logits, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
 
        # Zero out all logits outside the top k
        lm_logits = lm_logits.masked_fill(~mask, 0.0)
 
        # Layer-wise weighted sum
        weights = decay ** torch.arange(
            num_layers - 1, -1, -1,
            device=lm_logits.device
        )
        layer_weights = weights / weights.sum()          # (l,) normalized, reused by the Bayes decoders
        ## ESR 2026-09-16
        # [num_layers] -> [num_layers, 1, 1, 1]
        weights = weights.view(num_layers, 1, 1, 1)
        
        
        lm_logits = (lm_logits * weights).sum(dim=0) / weights.sum()
 
        # ---------------- Bayes decoders: same layer-weighted sum, on calibrated logits ----------------
        # Every layer turns its raw recall into logits (temperature follows that layer's load, the
        # prior is that layer's token-count prior); logits are then pooled with the same normalized
        # layer weights as above. The weighted sum of logits equals the weighted sum of log-probabilities
        # up to a constant per position, so the final softmax is a weighted log-linear pooling.
        decoder_probs = {"legacy": torch.softmax(lm_logits, dim=-1)}
        decoder_log_probs = {"legacy": torch.log_softmax(lm_logits, dim=-1)} if return_log_probs else None
 
        if names:
            E = self._candidate_matrix()
            E_sq = E.pow(2).sum(-1)
            raw_recall = outputs['raw_recall_unpacked']                       # (l, B, T, d)
            pooled = {n: None for n in names}
            for layer_level in range(num_layers):
                per_layer = self._layer_decoder_logits(
                    self.memo.layers[layer_level], raw_recall[layer_level], E, E_sq, names)
                for n in names:
                    contrib = layer_weights[layer_level] * per_layer[n]
                    pooled[n] = contrib if pooled[n] is None else pooled[n] + contrib
                del per_layer
            for n in names:
                logp = torch.log_softmax(pooled[n], dim=-1)
                decoder_probs[n] = logp.exp()
                if return_log_probs:
                    decoder_log_probs[n] = logp
 
        layers = self.memo.layers
        decoder_info = dict(
            layer_weights=layer_weights,                                                    # (l,)
            n_pairs=torch.stack([layers[i].n_pairs for i in range(num_layers)]),            # (l,) stored pairs P
            gamma=torch.stack([layers[i].mean_key_sq_norm() for i in range(num_layers)]),   # (l,) mean |k|^2
            inverse_temperature=torch.stack(                                                 # (l,) theoretical beta (plain, no scale)
                [layers[i].inverse_temperature(apply_scale=False) for i in range(num_layers)]),
            beta_scale=torch.stack([layers[i].beta_scale for i in range(num_layers)]),       # (l, 4) fitted corrections
        )
        
        loss = None # TODO: compute the loss function
            
        
        if not return_dict:
            outputs = (lm_logits,) + outputs[1:]
            return ((loss,) + outputs) if loss is not None else outputs
            
 
        return MeMoCausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            hidden_tokens=outputs.hidden_tokens,
            lm_logits_unpacked=lm_logits_unpacked,
            decoder_probs=decoder_probs,
            decoder_log_probs=decoder_log_probs,
            decoder_info=decoder_info,
        )
 
 
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
        compute_loss: Optional[bool] = False,
        decoders: Optional[Sequence[str]] = None,
        return_log_probs: Optional[bool] = False,
    ) -> Optional[Union[Tuple[torch.Tensor], MeMoCausalLMOutputWithPast]]:
 
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if labels is not None and not compute_loss:
            if not self.training:
                logger.warning_once(
                    "`using forward method with labels but model is in eval mode. Setting model.train() and calling model.memorize"
                )
                self.train()
            
            return self.memorize(input_ids, labels_ids=labels)
        
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
            cache_position=cache_position,
            decoders=decoders,
            return_log_probs=return_log_probs,
        )
    

    ### ESR 2026-09-08
    @torch.no_grad()
    def _score_decoder(self, log_probs, probs, safe_labels, valid_mask, topk, vocab_size,
                       compute_accuracy=False):
        """Metrics of ONE decoder from its own distribution over the vocabulary.
 
        log_probs, probs : (B, K, V)   log-probabilities / probabilities of the decoder
        safe_labels      : (B, K)      labels with the masked positions replaced by a valid id
        valid_mask       : (B, K)      False where the label is -100
 
        No top-k filtering and no rescaling is applied here: calibration (NLL, entropy) is exactly
        what separates the decoders. Ranks count ties AGAINST the decoder (a token tied with the
        true one is ranked above it), so a decoder that cannot tell tokens apart is not credited.
        """
        n_valid = int(valid_mask.sum().item())
        denom = max(n_valid, 1)
 
        logp_true = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, K)
 
        def masked_mean(x):
            return (torch.where(valid_mask, x, torch.zeros_like(x)).sum() / denom).item()
 
        avg_nll = masked_mean(-logp_true)
 
        # rank of the true token: 0 = it is the arg-max
        ref = logp_true.unsqueeze(-1)
        rank0 = (log_probs > ref).sum(dim=-1) + (log_probs == ref).sum(dim=-1) - 1                # (B, K)
 
        entropy = -torch.special.xlogy(probs, probs).sum(dim=-1)                                  # (B, K)
        confidence = probs.max(dim=-1).values                                                     # (B, K)
 
        result = dict(
            avg_nll=avg_nll,
            ppl=float(torch.exp(torch.tensor(avg_nll)).item()) if avg_nll < 80 else float("inf"),
            avg_entropy=masked_mean(entropy),        # ~ avg_nll for a calibrated decoder
            avg_confidence=masked_mean(confidence),  # mean max-probability; ~ top-1 accuracy if calibrated
            topk_acc={int(k): ((rank0 < k) & valid_mask).sum().item() / denom for k in topk},
            accuracy=((rank0 == 0) & valid_mask).sum().item() / denom,
            num_tokens=n_valid,
        )
 
        if compute_accuracy:
            flat_valid = valid_mask.flatten()
            valid_labels = safe_labels.flatten()[flat_valid]
            valid_correct = (rank0 == 0).flatten()[flat_valid]
            target_counts = torch.bincount(valid_labels, minlength=vocab_size)
            correct_counts = torch.bincount(valid_labels[valid_correct], minlength=vocab_size)
            ids = torch.nonzero(target_counts, as_tuple=True)[0].tolist()
            tc, cc = target_counts.tolist(), correct_counts.tolist()
            result['token_stats'] = {tid: {'target_count': tc[tid], 'correct_count': cc[tid]} for tid in ids}
        return result
 
 
    def forward_with_loss_simple(
        self,
        batch_inputs,
        return_dict: Optional[bool] = None,
        compute_accuracy=False,
        starting_point=9,
        tokenizer=None,
        decoders: Optional[Sequence[str]] = None,
        topk: Sequence[int] = (1, 5, 10, 100),
    ) -> Optional[Union[Tuple[torch.Tensor], "MeMoCausalLMOutputWithPast"]]:
        """
        Simplified forward_with_loss for models whose forward() already scores an
        entire sequence in one call (see the original docstring: logits[:, p, :] is aligned
        with labels[:, p]; -100 in labels excludes a position from every statistic).
 
        Decoders. The model returns, besides the legacy `logits`, one probability distribution
        per decoder (outputs.decoder_probs / decoder_log_probs: name -> (B, seq_len, V)).
        `decoders` selects which of the Bayes decoders to compute (None = all; "legacy" is always
        there). Results:
 
          * The top-level fields of accuracy_results / batch_debug (perplexity, avg_nll, accuracy,
            loss, token_stats, predictions, ...) are the LEGACY decoder, computed exactly as before
            (top-50 filtering, fill value and x10 included), so existing callers are unaffected.
          * accuracy_results['decoders'][name] and batch_debug['decoders'][name] give, for EVERY
            decoder (legacy included), metrics computed from its own distribution with no
            filtering or rescaling: avg_nll, ppl, avg_entropy, avg_confidence, accuracy (top-1),
            topk_acc {k: accuracy}, num_tokens (and token_stats when compute_accuracy).
            The legacy entry here is the unfiltered legacy distribution, so it is directly
            comparable with the others; it will differ from the top-level (filtered) legacy numbers.
        """
        starting_point = max(2, starting_point)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
     
        input_ids, labels = batch_inputs['input_ids'].to(self.memo.device), batch_inputs['labels'].to(self.memo.device)
     
        vocab_size = self.config.vocab_size
        batch_size, seq_len = input_ids.shape
     
        if starting_point >= seq_len:
            return None, None, None
     
        # ---- Single forward pass over the whole sequence ----------------------------
        outputs = self.forward(
            input_ids=input_ids,
            labels=labels,
            return_dict=return_dict,
            compute_loss=True,
            decoders=decoders,
            return_log_probs=True,      # NLL and ranks are computed from log-probabilities (no underflow)
        )
        logits = outputs.logits  # (B, seq_len, vocab_size)
        lm_logits = logits.detach()
     
        if starting_point > 0:
            lm_logits = lm_logits[:, starting_point:, :]
            labels = labels[:, starting_point:]
     
        # positions kept, expressed as indices into the *original* sequence -- used
        # only for the 'sequence_index' field in the debug output.
        kept_positions = list(range(starting_point, seq_len))
     
        # top-k filtering (same as original): keep only the top-50 logits per position
        top_k = torch.topk(lm_logits, k=50, dim=-1)
        mask = torch.full_like(lm_logits, -12.0)
        mask.scatter_(dim=-1, index=top_k.indices, src=top_k.values)
        lm_logits = mask * 10
     
        _labels = labels.contiguous().to(self.memo.device)
        loss = self.loss_function(logits=lm_logits, labels=_labels, vocab_size=vocab_size)
     
        valid_mask = _labels != -100
        flat_labels_t = _labels.flatten()
        flat_valid = valid_mask.flatten()
     
        num_valid_tokens = flat_valid.sum().item()
     
        total_loss = loss * num_valid_tokens
        total_nll_sum = loss.detach() * num_valid_tokens
        num_tokens_predicted = num_valid_tokens
     
        need_top1 = compute_accuracy or (tokenizer is not None)
        if need_top1:
            top1 = torch.max(lm_logits, dim=-1)
            argmax = top1.indices          # (B, kept_len)
            argmax_value = top1.values     # (B, kept_len)
        else:
            argmax = None
            argmax_value = None
     
        # ---- Accuracy / token_stats / padding analysis, vectorized, one shot -------
        accuracy_results = None
        if compute_accuracy:
            flat_correct = (argmax == _labels).flatten()  # bool
     
            valid_labels = flat_labels_t[flat_valid]
            valid_correct = flat_correct[flat_valid]
     
            target_counts = torch.bincount(valid_labels, minlength=vocab_size)
            correct_counts = torch.bincount(valid_labels[valid_correct], minlength=vocab_size)
            nonzero_ids = torch.nonzero(target_counts, as_tuple=True)[0].tolist()
            token_stats = {
                tid: {
                    'target_count': target_counts[tid].item(),
                    'correct_count': correct_counts[tid].item(),
                }
                for tid in nonzero_ids
            }
     
            pad_token_id = getattr(self.config, 'pad_token_id', 0)
            is_pad = flat_labels_t == pad_token_id
            padding_tokens_masked = torch.sum(is_pad & ~flat_valid).item()
            padding_tokens_not_masked = torch.sum(is_pad & flat_valid).item()
            padding_tokens_correct = (
                torch.sum(flat_correct & is_pad & flat_valid).item()
                if padding_tokens_not_masked > 0 else 0
            )
     
            tot_correct_tokens = valid_correct.sum().item()
            tot_tokens = int(num_valid_tokens)
     
            accuracy_results = dict(
                accuracy=(tot_correct_tokens / tot_tokens) if tot_tokens > 0 else 0.0,
                correct_tokens=tot_correct_tokens,
                tot_tokens=tot_tokens,
                token_stats=token_stats,
                padding_analysis=padding_tokens_correct,
            )
            # kept for parity with the original dict shape, in case callers read these:
            accuracy_results['padding_tokens_masked'] = padding_tokens_masked
            accuracy_results['padding_tokens_not_masked'] = padding_tokens_not_masked
     
        # ---- Perplexity ---------------------------------------------------------------
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
 
        # ---- Per-decoder metrics (every distribution returned by the model) ---------------
        # _labels contains -100 at masked positions: use a clamped copy for gather/decoding only
        pad_id_for_safe = getattr(self.config, 'pad_token_id', 0) or 0
        safe_labels_all = torch.where(
            _labels == -100, torch.full_like(_labels, pad_id_for_safe), _labels
        )
        decoder_log_probs = {}
        decoder_results = {}
        if outputs.decoder_log_probs is not None:
            for name, lp in outputs.decoder_log_probs.items():
                lp = lp.detach()[:, starting_point:, :]
                pr = outputs.decoder_probs[name].detach()[:, starting_point:, :]
                decoder_log_probs[name] = lp
                decoder_results[name] = self._score_decoder(
                    lp, pr, safe_labels_all, valid_mask, topk, vocab_size, compute_accuracy
                )
        accuracy_results['decoders'] = decoder_results
     
        # ---- Optional debug info (only built if a tokenizer is supplied) --------------
        debug_predictions = list()
        if tokenizer is not None:
            top_k5 = torch.topk(lm_logits, k=5, dim=-1)
            _lm_logits_softmax = torch.nn.functional.softmax(lm_logits, dim=-1)
            argmax_value_soft = torch.max(_lm_logits_softmax, dim=-1)
            top_k_softmax = torch.topk(_lm_logits_softmax, k=5, dim=-1)
     
            # _labels contains -100 at masked positions, which is not a valid vocab
            # index / token id -- gather() and tokenizer decoding both require real
            # ids. Build a clamped copy for indexing/decoding only; the *_prob /
            # *_score / *_tokens values at masked positions are meaningless
            # placeholders and should be ignored there (valid_mask / flat_valid
            # already tells you which positions are real).
            pad_token_id_for_decode = getattr(self.config, 'pad_token_id', 0) or 0
            safe_labels = torch.where(
                _labels == -100,
                torch.full_like(_labels, pad_token_id_for_decode),
                _labels
            )
     
            expected_label_prob = torch.gather(_lm_logits_softmax, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
            expected_label_score = torch.gather(lm_logits, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
     
            pred_tokens_flat = tokenizer.batch_decode(argmax.reshape(-1, 1))
            expected_tokens_flat = tokenizer.batch_decode(safe_labels.reshape(-1, 1))
     
            kept_len = lm_logits.shape[1]
     
            def _regroup(flat_list):
                return [flat_list[b * kept_len:(b + 1) * kept_len] for b in range(batch_size)]
     
            pred_tokens_grouped = _regroup(pred_tokens_flat)
            expected_tokens_grouped = _regroup(expected_tokens_flat)
     
            pred_score = argmax_value.cpu().numpy().tolist()
            pred_score_softmax = argmax_value_soft.values.cpu().numpy().tolist()
            expected_label_prob_l = expected_label_prob.cpu().numpy().tolist()
            expected_label_score_l = expected_label_score.cpu().numpy().tolist()
     
            top_k_tokens_list = top_k5.indices.cpu().numpy().tolist()
            top_k_scores = top_k5.values.cpu().numpy().tolist()
            top_k_softmax_scores = top_k_softmax.values.cpu().numpy().tolist()
     
            vocab_sum = lm_logits.sum(dim=-1).cpu().numpy().tolist()
            vocab_mean = lm_logits.mean(dim=-1).cpu().numpy().tolist()
            vocab_sum_soft = _lm_logits_softmax.sum(dim=-1).cpu().numpy().tolist()
            vocab_mean_soft = _lm_logits_softmax.mean(dim=-1).cpu().numpy().tolist()
            vocab_min = lm_logits.min(dim=-1).values.cpu().numpy().tolist()
            vocab_min_soft = _lm_logits_softmax.min(dim=-1).values.cpu().numpy().tolist()
     
            exp_lm_logits = torch.exp(lm_logits)
            exp_sum = exp_lm_logits.sum(dim=-1).cpu().numpy().tolist()
            exp_mean = exp_lm_logits.mean(dim=-1).cpu().numpy().tolist()
            exp_min = exp_lm_logits.min(dim=-1).values.cpu().numpy().tolist()
            exp_max = exp_lm_logits.max(dim=-1).values.cpu().numpy().tolist()
     
            loss_scalar = loss.detach().cpu().numpy().tolist()  # mean loss over the whole sequence/batch
 
            # per-decoder debug: top-5 tokens with their probabilities, and the probability of the label
            dec_dbg = {}
            for name, lp in decoder_log_probs.items():
                t5 = torch.topk(lp, k=5, dim=-1)
                true_lp = torch.gather(lp, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
                dec_dbg[name] = dict(
                    top_ids=t5.indices.cpu().numpy().tolist(),            # (B, K, 5)
                    top_probs=t5.values.exp().cpu().numpy().tolist(),      # (B, K, 5)
                    expected_prob=true_lp.exp().cpu().numpy().tolist(),    # (B, K)
                )
     
            for local_p in range(kept_len):
                i_value = kept_positions[local_p]
                batch_debug_info = {
                    'sequence_index': i_value,
                    'pred_tokens': [pred_tokens_grouped[b][local_p] for b in range(batch_size)],
                    'pred_score': [pred_score[b][local_p] for b in range(batch_size)],
                    'pred_score_softmax': [pred_score_softmax[b][local_p] for b in range(batch_size)],
                    'expected_tokens': [expected_tokens_grouped[b][local_p] for b in range(batch_size)],
                    'expected_label_prob': [expected_label_prob_l[b][local_p] for b in range(batch_size)],
                    'expected_label_score': [expected_label_score_l[b][local_p] for b in range(batch_size)],
                    'top_predicted_tokens': [
                        tokenizer.convert_ids_to_tokens(top_k_tokens_list[b][local_p])
                        for b in range(batch_size)
                    ],
                    'top_predicted_token_scores': [top_k_scores[b][local_p] for b in range(batch_size)],
                    'top_predicted_tokens_softmax': [top_k_softmax_scores[b][local_p] for b in range(batch_size)],
     
                    'vocab_distribution_score_sum': [vocab_sum[b][local_p] for b in range(batch_size)],
                    'vocab_distribution_score_mean': [vocab_mean[b][local_p] for b in range(batch_size)],
                    'vocab_distribution_score_sum_softmax': [vocab_sum_soft[b][local_p] for b in range(batch_size)],
                    'vocab_distribution_score_mean_softmax': [vocab_mean_soft[b][local_p] for b in range(batch_size)],
                    'vocab_distrib_min_score': [vocab_min[b][local_p] for b in range(batch_size)],
                    'vocab_distrib_min_score_softmax': [vocab_min_soft[b][local_p] for b in range(batch_size)],
     
                    'exp_vocab_distribution_score_sum': [exp_sum[b][local_p] for b in range(batch_size)],
                    'exp_vocab_distribution_score_mean': [exp_mean[b][local_p] for b in range(batch_size)],
                    'exp_vocab_distrib_min_score': [exp_min[b][local_p] for b in range(batch_size)],
                    'exp_vocab_distrib_max_score': [exp_max[b][local_p] for b in range(batch_size)],
     
                    # NOTE: single scalar mean loss over the whole batch/sequence, not per-position
                    # (same caveat as the windowed version -- there's now only one loss value total).
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
 
                # per-decoder predictions at this position (added after the inf/-inf sanitization,
                # probabilities are finite)
                batch_debug_info['decoders'] = {
                    name: {
                        'pred_tokens': [tokenizer.convert_ids_to_tokens(d['top_ids'][b][local_p][0]) for b in range(batch_size)],
                        'expected_label_prob': [d['expected_prob'][b][local_p] for b in range(batch_size)],
                        'top_predicted_tokens': [tokenizer.convert_ids_to_tokens(d['top_ids'][b][local_p]) for b in range(batch_size)],
                        'top_predicted_tokens_prob': [d['top_probs'][b][local_p] for b in range(batch_size)],
                    }
                    for name, d in dec_dbg.items()
                }
                debug_predictions.append(batch_debug_info)
     
        batch_debug = dict(
            ppl=accuracy_results['perplexity'],
            avg_nll=accuracy_results['avg_nll'],
            num_tokens=accuracy_results['num_tokens'],
            accuracy=accuracy_results['accuracy'] if compute_accuracy else None,
            correct_tokens=accuracy_results['correct_tokens'] if compute_accuracy else None,
            total_tokens=accuracy_results['tot_tokens'] if compute_accuracy else None,
            decoders={
                name: {k: v for k, v in res.items() if k != 'token_stats'}
                for name, res in decoder_results.items()
            },
            predictions=debug_predictions
        )
     
        return MeMoCausalLMOutputWithPast(
            loss=total_loss,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            hidden_tokens=None,
            decoder_info=outputs.decoder_info,
        ), accuracy_results, batch_debug
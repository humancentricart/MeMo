# from https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/gpt_neo/configuration_gpt_neo.py#L29
"""MeMo model configuration"""

from collections import OrderedDict
from typing import Any, Mapping, Optional
from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MeMoConfig(PretrainedConfig):

    model_type = "memo"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50254,
        #max_position_embeddings=2048,
        hidden_size=2048,
        num_hidden_layers=24,
        #attention_types=[[["global", "local"], 12]],
        num_attention_heads=16,
        chunk_length=384,
        #intermediate_size=None,
        #window_size=256,
        #activation_function="gelu_new",
        #resid_dropout=0.0,
        #embed_dropout=0.0,
        #attention_dropout=0.0,
        #classifier_dropout=0.1,
        #layer_norm_epsilon=1e-5,
        #initializer_range=0.02,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        #self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.chunk_length= chunk_length
        #self.intermediate_size = intermediate_size
        #self.window_size = window_size
        #self.activation_function = activation_function
        #self.resid_dropout = resid_dropout
        #self.embed_dropout = embed_dropout
        #self.attention_dropout = attention_dropout
        #self.classifier_dropout = classifier_dropout
        #self.layer_norm_epsilon = layer_norm_epsilon
        #self.initializer_range = initializer_range
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        #self.attention_types = attention_types
        #self.attention_layers = self.expand_attention_types_params(attention_types)

        super().__init__(bos_token_id=bos_token_id, 
                         eos_token_id=eos_token_id, 
                         pad_token_id=pad_token_id, **kwargs)
    



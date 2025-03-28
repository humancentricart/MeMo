import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.module import Module
from torch.nn.modules.linear import Linear
import math

from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging


from .modelling_memo_exception import MeMoException
# self.Prj = torch.normal(0, 1/math.sqrt(self.d*self.h), size=(self.d,self.d*self.h))
# self.Prj = torch.transpose(self.Prj, 0, 1)

# as in Linear torch.nn.modules.linear

verbose = False


class ProjectionSequence(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int, # self.d*self.h after transpose
        out_features: int, # self.d after transpose
        bias: bool = False,
        device=None,
        dtype=None,
        init_weights=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mean = 0
        self.std = 1 / math.sqrt(out_features)
        
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        self.register_parameter("bias", None)
        if init_weights:
            self.reset_parameters()

    def _init_weights(self):
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        init.normal_(self.weight, mean=self.mean, std=self.std)
        #proj = self
        #print((proj.weight.T @ proj.weight).diag(), (proj.weight @ proj.weight.T).diag())
        
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.T, self.bias)

    def extra_repr(self) -> str:
        return f"(trasposed wrt saved one) in_features={self.out_features}, out_features={self.in_features}"



#self.W_v_single_head = torch.normal(0, 1/math.sqrt(self.d_k), size=(self.d,self.d_k))

# as in Linear torch.nn.modules.linear
class ProjectionTokens(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int, # d
        out_features: int, #d_k
        bias: bool = True,
        device=None,
        dtype=None,
        init_weights=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mean = 0
        self.std = 1/math.sqrt(out_features)
        
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        self.register_parameter("bias", None)

        if init_weights:
            self.reset_parameters()

    def _init_weights(self):
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        init.normal_(self.weight, mean=self.mean, std=self.std)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"




# CMM : correlation matrix memory for the specific layer
class CorrelationMatrixMemory(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int, # d
        out_features: int, #d_k
        bias: bool = True,
        device=None,
        dtype=None,
        init_weights=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        self.register_parameter("bias", None)
        if init_weights:
            self.reset_parameters()

    def _init_weights(self):
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        #print('CMM inizializzataaa')
        init.zeros_(self.weight)
        #print(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.T)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"

    def memorize(self, update):
        if len(update.shape) ==  3:
            update = torch.sum(update, dim=0) #handle batch update
        
        with torch.no_grad():
            self.weight += update #in place update

    def forget(self, update):
        if len(update.shape) ==  3:
            update = torch.sum(update, dim=0) #handle batch update
            
        with torch.no_grad():
            self.weight -= update #in place update




class MeMoLayer(Module):
    
    def __init__(self, inner_dim, num_of_heads, init_weights=True, **kwargs):
        super().__init__()
        
        self.d = inner_dim
        self.h = num_of_heads
        self.d_k = self.d // self.h
        if self.d / self.h != self.d_k:
            raise MeMoException("Inner dimension " + str(self.d) + " should be divisible for number of heads " + str(self.h))

        self.W_v_single_head = ProjectionTokens(self.d, self.d_k, init_weights=init_weights)
        self.Prj = ProjectionSequence(self.d, self.d*self.h, init_weights=init_weights)
        # CMM : correlation matrix memory for the specific layer
        self.CMM = CorrelationMatrixMemory(self.d, self.d, init_weights=init_weights)
        # CMM OUT : correlation matrix memory for the specific layer
        #self.CMM_OUT = CorrelationMatrixMemory(self.d, self.d, init_weights=init_weights)

    def _init_weights(self):
        self.reset_parameters()
        
    def reset_parameters(self):
        pass
        #self.W_v_single_head.reset_parameters() 
        #self.Prj.reset_parameters()
        #self.CMM.reset_parameters()
    
    def get_projections(self, input_sequence, blocks, h, d):
        #print('input_sequence.shape, layer', input_sequence.shape)
        #print('self.d, self.h', self.d, self.h)
        # shape (blocks,self.d)
        #print(input_sequence.reshape((blocks,self.d * self.h)).shape)
        batch_size = input_sequence.shape[0]
        
        sequence_encoding = self.Prj(input_sequence.reshape((batch_size, blocks, self.d * self.h)))
        # shape (blocks,self.d)
        seq_enc_per_token = self.W_v_single_head(input_sequence) / math.sqrt(self.h)
        seq_enc_per_token = seq_enc_per_token.reshape((batch_size, blocks, self.d))

        #print('sequence_encoding', sequence_encoding.shape)
        #print('seq_enc_per_token', seq_enc_per_token.shape)
        return sequence_encoding, seq_enc_per_token

    
    def penalize(self, sequence_encoding, seq_enc_per_token):
        # Penalizing factors to avoid multiple storage of the same sequence encoding in intermediate CMMs
        # This penalizing factor should merge the repetitions founds in the current sequence and the already stored
        # sequences in the current CMM
        # dimension = (blocks,1)
        batch_size = sequence_encoding.shape[0]
        all_sequences = torch.sum(sequence_encoding, dim=1)
        # OLD : all_sequences = all_sequences.reshape(batch_size, self.d, 1)
        #print(all_sequences.shape) #(batch_size, d, 1)

        all_sequences = torch.sum(all_sequences, dim=0)
        all_sequences = all_sequences.reshape(self.d, 1)
        #print(all_sequences.shape) #(batch_size, d, 1)

       
        stored_sequences_filter = 1 - torch.round(torch.matmul(self.CMM(seq_enc_per_token), all_sequences))
        stored_sequences_filter[stored_sequences_filter < 0] = 0.0 # Sequences may appear more than once in all_sequences
        
        #print(stored_sequences_filter.shape)
        new_sequences = torch.round(torch.matmul(sequence_encoding, all_sequences))
        #print(new_sequences.shape)
        
        #### Penalizing factors : provided that it is correct, elements with 0 are problematic. For the moment,
        # there is a correcting added of 0.001 to avoid 0 elements.
        #print(stored_sequences_filter.shape, new_sequences.shape)
        penalizing_factors = 1/(stored_sequences_filter*new_sequences)
        
        penalizing_factors[penalizing_factors == math.inf] = 0
        penalizing_factors[penalizing_factors == -math.inf] = 0
    
        return sequence_encoding * penalizing_factors
    
    # The most simple implementation
    # Input sequence is has a shape of (self.h,self.d), that is self.h sequences are proposed as input rows
    def memorize(self, input_sequence, output_symbols, is_last = False):
        (batch_size, blocks,h,d) = input_sequence.shape
        sequence_encoding, seq_enc_per_token = self.get_projections(input_sequence, blocks, h, d)
        
        # Updating local CMM
        if not is_last:
            surviving_vectors = self.penalize(sequence_encoding, seq_enc_per_token)
            # TODO
            # batch ready? not for the sum, in general is always thought as 1 seq per input (in blocks)
            #self.CMM = self.CMM + torch.matmul(torch.transpose(seq_enc_per_token, 0, 1), surviving_vectors)
            CMM_update = torch.matmul(torch.transpose(seq_enc_per_token, -2, -1), surviving_vectors)
            self.CMM.memorize(CMM_update)
            
            
        #### To be adjusted for taking into consideration the entire sequence
        seq_enc_plus_out = torch.matmul(torch.transpose(seq_enc_per_token,-2,-1), output_symbols) ## Key (sequenze di h token) x Value ==> matrice??
        return sequence_encoding, seq_enc_plus_out
    
    def directly_memorize(self, input_sequence):
        self.CMM.memorize(input_sequence)

    def forget(self, input_sequence, output_symbols, completely=False, is_last = False):
        (batch_size, blocks,h,d) = input_sequence.shape
        sequence_encoding, seq_enc_per_token = self.get_projections(input_sequence, blocks, h, d)
        
        # Updating local CMM
        if not is_last and completely:
            surviving_vectors = self.penalize(sequence_encoding, seq_enc_per_token)
            CMM_update = torch.matmul(torch.transpose(seq_enc_per_token, -2, -1), surviving_vectors)
            self.CMM.forget(CMM_update)


        #### To be adjusted for taking into consideration the entire sequence
        seq_enc_plus_out = torch.matmul(torch.transpose(seq_enc_per_token,-2,-1), output_symbols) ## Key (sequenze di h token) x Value ==> matrice??
        return sequence_encoding, seq_enc_plus_out

    def directly_forget(self, input_sequence):
        self.CMM.forget(input_sequence)

    def retrieve(self, 
                 input_sequence: Optional[torch.Tensor] = None,
                 layer_past: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
                 use_cache: Optional[bool] = None,
                 output_hidden_token: Optional[bool] = None, #output_attentions: Optional[bool] = None,
                 cache_position: Optional[Union[Cache, torch.Tensor]] = None,
        ):
        
        (batch_size, blocks, _, _) = input_sequence.shape
        
        seq_enc_per_token = self.W_v_single_head(input_sequence).reshape((batch_size, blocks, self.d)) / math.sqrt(self.h)
        # print(seq_enc_per_token.shape) (batch_size, blocks, d)
        retrieved_sequence_encoding = self.CMM(seq_enc_per_token)

        
        #if verbose:
        #    sequence_encoding = self.Prj(input_sequence.reshape((batch_size, blocks, self.d * self.h)))
        #    out = torch.matmul(sequence_encoding, torch.transpose(retrieved_sequence_encoding,-2,-1))
        #    out1 = torch.linalg.norm(seq_enc_per_token, dim=-1)
        #    
        #    if torch.max(out1).item() > 1.5 or torch.max(out).item()> 1.5:
        #        print("Errore")
        # last token seq_enc_per_token[-1].reshape(batch_size,self.d)

        
        return OrderedDict([
            ('sequence_encoding', retrieved_sequence_encoding),
            ('token_encoding', seq_enc_per_token[:, -1].reshape(batch_size,self.d)),
            ('cache', None)
        ])
         

    def directly_retrieve(self,vector):
        return self.CMM(vector)


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
import numpy as np

import torch.nn.functional as F


DEBUGGING = False
DEBUGGING_PROJECTION = False
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


    def _initialize_weights(self):
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        init.normal_(self.weight, mean=self.mean, std=self.std)
        
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.T, self.bias)

    def extra_repr(self) -> str:
        return f"(trasposed wrt saved one) in_features={self.out_features}, out_features={self.in_features}"



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


    def _initialize_weights(self):
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
    

    def _initialize_weights(self):
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





from enum import Enum
class CompositionOp(Enum):
    JLT = 1
    PROD = 2
    SUM = 3
    PRODWITHSHUFFLING = 4
    SUMTHENPRODWITHSHUFFLING = 5
    SCC = 6


class MeMoLayer(Module):
    
    def __init__(self, inner_dim, num_of_heads, init_weights=True, compositionOp=CompositionOp.JLT, **kwargs):
        super().__init__()
        
        self.d = inner_dim
        self.h = num_of_heads
        self.d_k = self.d // self.h
        if self.d / self.h != self.d_k:
            raise MeMoException("Inner dimension " + str(self.d) + " should be divisible for number of heads " + str(self.h))

        
        self.compOp = compositionOp
        
        ### ESR 2026-09-08 init of the projection matrices only if the composition operation is the JLT
        if self.compOp == CompositionOp.JLT:
            self.W_v_single_head = ProjectionTokens(self.d, self.d_k, init_weights=init_weights)
            self.Prj = ProjectionSequence(self.d, self.d*self.h, init_weights=init_weights)


        if self.compOp in (
            CompositionOp.PRODWITHSHUFFLING,
            CompositionOp.SUMTHENPRODWITHSHUFFLING,
            CompositionOp.SCC,
        ):
            
            self.register_buffer(
                "shuff",
                torch.stack([torch.randperm(self.d) for _ in range(self.h)])
            )

        # ESR 2026/10/30 only the layerized predictions are implemented now
        self.layerized_CMM_OUT = True ### FMZ 2026:05:04
        # CMM OUT : correlation matrix memory for the specific layer
        if self.layerized_CMM_OUT: 
            self.CMM_OUT = CorrelationMatrixMemory(self.d, self.d, init_weights=init_weights) ### FMZ 2026:05:04


    def _initialize_weights(self):
        self.reset_parameters()
        
    def reset_parameters(self):
        pass
    

    def inspect_tensor(self, name, x):
        x = x.detach().float()

        print(
            f"{name} | shape={tuple(x.shape)} | "
            f"mean={x.mean().item():.5g} | "
            f"std={x.std().item():.5g} | "
            f"mean_abs={x.abs().mean().item():.5g} | "
            f"rms={x.square().mean().sqrt().item():.5g} | "
            f"min_abs={x.abs().amin().item():.5g} | "
            f"max_abs={x.abs().amax().item():.5g} | "
            f"zero_frac={(x == 0).float().mean().item():.5g}"
        )

        # L2 norm of each [hidden_dim] vector, aggregated over batch and tot_seq
        norms = x.norm(dim=-1)

        print(
            f"{name} | vector L2 norm: "
            f"mean={norms.mean().item():.5g} | "
            f"std={norms.std().item():.5g} | "
            f"min={norms.amin().item():.5g} | "
            f"max={norms.amax().item():.5g}"
        )
        print("-"*80)

    def get_projections(self, input_sequence, blocks, h, d):
        
        batch_size = input_sequence.shape[0]
        if DEBUGGING_PROJECTION:
            self.inspect_tensor('input', input_sequence)

        if self.compOp == CompositionOp.PROD:
            sequence_encoding = torch.prod(input_sequence,2)
            sequence_encoding = F.normalize(sequence_encoding, p=2, dim=2)
            return sequence_encoding, None
        
        elif self.compOp == CompositionOp.PRODWITHSHUFFLING:
            sequence_encoding = self.prod_with_shuf(input_sequence)
            if DEBUGGING_PROJECTION:
                self.inspect_tensor('output BEFOR NORM', sequence_encoding)
            sequence_encoding = F.normalize(sequence_encoding, p=2, dim=2)
            if DEBUGGING_PROJECTION:
                self.inspect_tensor('output', sequence_encoding)
            return sequence_encoding, None

        elif self.compOp == CompositionOp.SCC:
            sequence_encoding = self.shuffled_circular_convolution(input_sequence)
            if DEBUGGING_PROJECTION:
                self.inspect_tensor('output BEFOR NORM', sequence_encoding)
            sequence_encoding = F.normalize(sequence_encoding, p=2, dim=2)
            if DEBUGGING_PROJECTION:
                self.inspect_tensor('output', sequence_encoding)
            return sequence_encoding, None
        
        elif self.compOp == CompositionOp.SUMTHENPRODWITHSHUFFLING:
            sequence_encoding = self.sum_prod_with_shuf(input_sequence)
            if DEBUGGING_PROJECTION:
                self.inspect_tensor('output BEFOR NORM', sequence_encoding)
            sequence_encoding = F.normalize(sequence_encoding, p=2, dim=2)
            if DEBUGGING_PROJECTION:
                self.inspect_tensor('output', sequence_encoding)
            return sequence_encoding, None
        
        ## ESR 2026-09-06 still unclear to me 
        elif self.compOp == CompositionOp.JLT:
            sequence_encoding = self.Prj(input_sequence.reshape((batch_size, blocks, self.d * self.h)))
        
            seq_enc_per_token = self.W_v_single_head(input_sequence) / math.sqrt(self.h)
            seq_enc_per_token = seq_enc_per_token.reshape((batch_size, blocks, self.d))
            
            return sequence_encoding, seq_enc_per_token
        
        elif self.compOp == CompositionOp.SUM:
            sequence_encoding = torch.sum(input_sequence,2)
            sequence_encoding = F.normalize(sequence_encoding, p=2, dim=2)
            return sequence_encoding, None
        
        else: 
            print("ERROR")


    def prod_with_shuf(self, input_sequence):
        B, L, H, D = input_sequence.shape
        shuff = self.shuff.view(1, 1, H, D)
        input_sequence = torch.gather(input_sequence, dim=-1, index=shuff.expand(B, L, H, D))

        return torch.prod(input_sequence,2)

    def sum_prod_with_shuf(self, input_sequence, bag_of_words_until=None, eps=1e-12):
        B, L, H, D = input_sequence.shape
        shuff = self.shuff.view(1, 1, H, D)
        input_sequence = torch.gather(input_sequence, dim=-1, index=shuff.expand(B, L, H, D))

        if bag_of_words_until is None:
            bag_of_words_until = H - 1
        if not 0 <= bag_of_words_until <= H:
            raise ValueError(
                f"bag_of_words_until must be between 0 and {H}"
            )

        bag_of_words = input_sequence[:, :, :bag_of_words_until, :].sum(dim=2).unsqueeze(dim=2)
        bag_of_words = F.normalize(bag_of_words, p=2, dim=-1)
        vectors_to_compose = torch.cat(
            (bag_of_words, input_sequence[:, :, bag_of_words_until:, :]),
            dim=2,
        )
        
        return torch.prod(vectors_to_compose, dim=2)


    def shuffled_circular_convolution(self, input_sequence: torch.Tensor) -> torch.Tensor:
        B, L, H, D = input_sequence.shape
        shuff = self.shuff.view(1, 1, H, D)
        input_sequence = torch.gather(input_sequence, dim=-1, index=shuff.expand(B, L, H, D))

        X = torch.fft.fft(input_sequence, dim=3)
        X_prod = torch.prod(X, dim=2, keepdim=True)

        z = torch.fft.ifft(X_prod, dim=3)

        return torch.real(z).squeeze(dim=2)
    


    def memorize(self, input_sequence, output_symbols, to_save_sequences=None, is_last = False):
        (batch_size, blocks,h,d) = input_sequence.shape
        sequence_encoding, seq_enc_per_token = self.get_projections(input_sequence, blocks, h, d)
        
        # ESR 2026-09-17 
        if to_save_sequences is not None:
            to_save_output_symbols = output_symbols[:, to_save_sequences]
        else:
            to_save_output_symbols = output_symbols
        
            
        if self.compOp == CompositionOp.JLT:
            
            if to_save_sequences is not None:
                to_save_seq_enc_per_token = seq_enc_per_token[:, to_save_sequences]
            else:
                to_save_seq_enc_per_token = seq_enc_per_token

            if DEBUGGING:
                print(f"to_save_seq_enc_per_token : {to_save_seq_enc_per_token.shape} {torch.transpose(to_save_seq_enc_per_token,-2,-1).shape} +  {to_save_output_symbols.shape}")
            
            seq_enc_plus_out = torch.matmul(torch.transpose(to_save_seq_enc_per_token,-2,-1), to_save_output_symbols) 
        else:
            if to_save_sequences is not None:
                to_save_sequence_encoding = sequence_encoding[:, to_save_sequences]
            else:
                to_save_sequence_encoding = sequence_encoding
            
            if DEBUGGING:
                print(f"seq_enc_plus_out : {to_save_sequence_encoding.shape} {torch.transpose(to_save_sequence_encoding,-2,-1).shape} +  {to_save_output_symbols.shape}")
            
            seq_enc_plus_out = torch.matmul(torch.transpose(to_save_sequence_encoding,-2,-1), to_save_output_symbols)
        
        if self.layerized_CMM_OUT: 
            self.CMM_OUT.memorize(seq_enc_plus_out)  ### FMZ 2026:05:04
        
        return sequence_encoding, seq_enc_plus_out

    

    # def forget(self, input_sequence, output_symbols, completely=False, is_last = False):
    #     (batch_size, blocks,h,d) = input_sequence.shape
    #     sequence_encoding, seq_enc_per_token = self.get_projections(input_sequence, blocks, h, d)
        
    #     # Updating local CMM
    #     if not is_last and completely and self.use_local_CMM:
    #         surviving_vectors = self.penalize(sequence_encoding, seq_enc_per_token)
    #         CMM_update = torch.matmul(torch.transpose(seq_enc_per_token, -2, -1), surviving_vectors)
    #         self.CMM.forget(CMM_update)


    #     #### To be adjusted for taking into consideration the entire sequence
    #     seq_enc_plus_out = torch.matmul(torch.transpose(seq_enc_per_token,-2,-1), output_symbols) 
    #     ## Key (sequenze di h token) x Value ==> matrice??
    #     return sequence_encoding, seq_enc_plus_out

    # def directly_forget(self, input_sequence):
    #     self.CMM.forget(input_sequence)

    def retrieve(self, 
                 input_sequence: Optional[torch.Tensor] = None,
                 layer_past: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
                 use_cache: Optional[bool] = None,
                 output_hidden_token: Optional[bool] = None, #output_attentions: Optional[bool] = None,
                 cache_position: Optional[Union[Cache, torch.Tensor]] = None,
        ):
                
        (batch_size, blocks,h,d) = input_sequence.shape
        sequence_encoding, seq_enc_per_token = self.get_projections(input_sequence, blocks, h, d)

        if self.layerized_CMM_OUT:

            if self.compOp == CompositionOp.JLT:
                if DEBUGGING:
                    print("seq_enc_per_token.shape", seq_enc_per_token.shape) #1 16 2048
                    print(self.CMM_OUT)
                layered_out_token = self.CMM_OUT(seq_enc_per_token)#[:,-1,:])
                if DEBUGGING:
                    print("layered_out_token.shape", layered_out_token.shape)
            else:
                if DEBUGGING:
                    print("sequence_encoding.shape", sequence_encoding.shape) #1 16 2048
                    print(self.CMM_OUT)
                layered_out_token = self.CMM_OUT(sequence_encoding)#[:,-1,:])
                if DEBUGGING:
                    print("layered_out_token.shape", layered_out_token.shape)
            
            layered_out_token = F.normalize(layered_out_token, p=2, dim=1)
            
            if DEBUGGING:
                print(f"Adding layered_out_token {layered_out_token.shape}")
        else:
            layered_out_token = None
            
        
        if seq_enc_per_token is not None:
            token_encoding = seq_enc_per_token[:, -1].reshape(batch_size,self.d)
        else:
            token_encoding = None
        
        return dict(
                sequence_encoding=sequence_encoding, 
                token_encoding=token_encoding,#, locally_predicted
                layered_out_token=layered_out_token,
                cache=None 
        )


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
    
    def __init__(self, inner_dim, num_of_heads, init_weights=True, compositionOp=CompositionOp.JLT,
                 num_embeddings=None, prior_smoothing=1.0, **kwargs):
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
 
        # ------------------------------------------------------------------------------------
        # Statistics of what the memory contains, used by the decoder (prior and temperature).
        #
        # The set of possible values is NOT stored here: values are rows of an embedding table with
        # `num_embeddings` entries, owned by the decoder. The layer only keeps
        #   token_counts[c] : how many stored pairs have value c        (the prior, as counts)
        #   n_pairs         : P, number of stored pairs (= token_counts.sum() when ids are tracked)
        #   key_sq_sum      : sum of ||k||^2 over the stored (encoded) keys     -> gamma = mean ||k||^2
        #   value_sq_sum    : sum of ||v||^2 over the stored values              -> mean ||v||^2
        #   beta_scale      : (4,) multiplicative corrections of the theoretical inverse temperature,
        #                     one per decoder family, order (plain, plain+prior, whitened, whitened+prior)
        #                     (1.0 = pure theory; set each to its fitted beta_fit/beta_theory)
        # All are buffers: they follow .to(device), are saved in state_dict and are NOT parameters.
        # Counts are kept instead of log-probabilities because log(pi_c) depends on the total P and
        # on the smoothing, so it is computed on demand (see log_prior).
        # ------------------------------------------------------------------------------------
        self.num_embeddings = num_embeddings
        self.prior_smoothing = prior_smoothing
        if num_embeddings is not None:
            self.register_buffer("token_counts", torch.zeros(num_embeddings, dtype=torch.long))
        self.register_buffer("n_pairs", torch.zeros((), dtype=torch.long))
        self.register_buffer("key_sq_sum", torch.zeros((), dtype=torch.float64))
        self.register_buffer("value_sq_sum", torch.zeros((), dtype=torch.float64))
        self.register_buffer("beta_scale", torch.ones(4, dtype=torch.float64))
 
 
    def _initialize_weights(self):
        self.reset_parameters()
        
    def reset_parameters(self):
        pass
 
    # ---------------------------------------------------------------------------------------
    # Prior and temperature
    # ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def reset_statistics(self):
        """Zero the statistics that describe the memory content. beta_scale is a calibration, not
        a content statistic, so it is kept."""
        if self.num_embeddings is not None:
            self.token_counts.zero_()
        self.n_pairs.zero_()
        self.key_sq_sum.zero_()
        self.value_sq_sum.zero_()
 
    @torch.no_grad()
    def reset_memory(self):
        """Empty the memory AND the statistics describing it. Use this (not CMM_OUT.reset_parameters()
        alone), otherwise the prior and the load would refer to pairs that are no longer stored."""
        self.CMM_OUT.reset_parameters()
        self.reset_statistics()
 
    @torch.no_grad()
    def _update_statistics(self, stored_keys, stored_values, stored_ids):
        """stored_keys (B, L', d), stored_values (B, L', d), stored_ids (B, L') or None:
        exactly the pairs that are written into CMM_OUT in this call."""
        self.n_pairs.add_(stored_values.shape[0] * stored_values.shape[1])
        self.key_sq_sum.add_(stored_keys.detach().double().pow(2).sum())
        self.value_sq_sum.add_(stored_values.detach().double().pow(2).sum())
        if stored_ids is not None:
            ids = stored_ids.reshape(-1).to(device=self.token_counts.device, dtype=torch.long)
            self.token_counts.index_add_(0, ids, torch.ones_like(ids))
 
    def log_prior(self, smoothing=None):
        """log pi_c for every embedding entry, pi_c = (count_c + a) / (P + a*V)   -> shape (V,).
        a = smoothing (default: self.prior_smoothing). With nothing stored and a > 0 the prior is
        uniform. a = 0 restricts the decoder to the values that were actually stored (log pi = -inf
        for the others); with nothing stored and a = 0 a uniform prior is returned."""
        if self.num_embeddings is None:
            raise RuntimeError("log_prior requires num_embeddings to be passed to the layer")
        a = self.prior_smoothing if smoothing is None else smoothing
        counts = self.token_counts.to(torch.float64)
        V = counts.numel()
        if a == 0 and int(self.n_pairs) == 0:
            return torch.full((V,), -math.log(V), dtype=torch.float32, device=counts.device)
        return torch.log((counts + a) / (counts.sum() + a * V)).to(torch.float32)
 
    def mean_key_sq_norm(self):
        """gamma = mean ||k||^2 of the stored encoded keys (~1 for the non-JLT operators)."""
        return (self.key_sq_sum / self.n_pairs.clamp_min(1)).to(torch.float32)
 
    def mean_value_sq_norm(self):
        """mean ||v||^2 of the stored values."""
        return (self.value_sq_sum / self.n_pairs.clamp_min(1)).to(torch.float32)
 
    def load(self):
        """alpha = (P - 1) / d, the load seen by one probe (the P - 1 other pairs make the noise)."""
        return ((self.n_pairs.double() - 1).clamp_min(1.0) / self.d).to(torch.float32)
 
    def inverse_temperature(self, whitened=False, with_prior=False, apply_scale=True):
        """Theoretical inverse temperature beta of the softmax over the raw recall r = M k.
            plain decoder   : logits = beta * (u_c . r - 0.5*gamma*||u_c||^2) + log pi_c
                              beta = gamma * d^2 / ((P-1) * mean||v||^2)      (= gamma / s^2)
            whitened decoder: logits = beta * (u_c^T C^-1 r - 0.5*gamma*u_c^T C^-1 u_c) + log pi_c
                              beta = gamma * d / (P-1)                         (= gamma / alpha)
        The two families live on different scales (they differ by mean||v||^2 / d), so a fitted
        correction must be stored per family AND per with/without prior (the fitted temperature differs
        when the prior is used). The matching entry of `beta_scale` (1.0 = theory) multiplies the
        result unless apply_scale=False. Always computed from the CURRENT content of the memory, so it
        follows the load while memorizing."""
        n = self.n_pairs.double()
        p_other = (n - 1).clamp_min(1.0)
        gamma = self.key_sq_sum / n.clamp_min(1.0)
        if whitened:
            beta = gamma * self.d / p_other
        else:
            mean_v2 = (self.value_sq_sum / n.clamp_min(1.0)).clamp_min(1e-12)
            beta = gamma * self.d ** 2 / (p_other * mean_v2)
        if apply_scale:
            beta = beta * self.beta_scale[self._beta_index(whitened, with_prior)]
        return beta.to(torch.float32)
 
    @staticmethod
    def _beta_index(whitened, with_prior):
        return 2 * int(bool(whitened)) + int(bool(with_prior))
 
    @torch.no_grad()
    def set_beta_scale(self, scale, whitened=False, with_prior=False):
        """Set the correction fitted by the decoder (beta_fit / beta_theory) for one decoder family."""
        self.beta_scale[self._beta_index(whitened, with_prior)] = float(scale)
    
 
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
    
 
 
    def memorize(self, input_sequence, output_symbols, to_save_sequences=None, is_last = False,
                 output_ids=None):
        """output_ids: (B, L) long tensor with the embedding index of each output symbol
        (output_symbols[b, l] is the embedding of output_ids[b, l]). Required when the layer was
        built with num_embeddings, since it is what updates the prior."""
        (batch_size, blocks,h,d) = input_sequence.shape
 
        # validate BEFORE touching the memory, so a failed call leaves memory and statistics consistent
        if self.num_embeddings is not None:
            if output_ids is None:
                raise ValueError("this layer was built with num_embeddings: memorize() needs output_ids "
                                 "(token id of every output symbol) to update the prior")
            if tuple(output_ids.shape) != tuple(output_symbols.shape[:2]):
                raise ValueError(f"output_ids {tuple(output_ids.shape)} must match "
                                 f"output_symbols[:2] {tuple(output_symbols.shape[:2])}")
 
        sequence_encoding, seq_enc_per_token = self.get_projections(input_sequence, blocks, h, d)
        
        # ESR 2026-09-17 
        if to_save_sequences is not None:
            to_save_output_symbols = output_symbols[:, to_save_sequences]
        else:
            to_save_output_symbols = output_symbols
 
        # ids of exactly the pairs that are stored (same selection as the values)
        if self.num_embeddings is not None:
            if to_save_sequences is not None:
                to_save_output_ids = output_ids[:, to_save_sequences]
            else:
                to_save_output_ids = output_ids
        else:
            to_save_output_ids = None
        
            
        if self.compOp == CompositionOp.JLT:
            
            if to_save_sequences is not None:
                to_save_seq_enc_per_token = seq_enc_per_token[:, to_save_sequences]
            else:
                to_save_seq_enc_per_token = seq_enc_per_token
            stored_keys = to_save_seq_enc_per_token
 
            if DEBUGGING:
                print(f"to_save_seq_enc_per_token : {to_save_seq_enc_per_token.shape} {torch.transpose(to_save_seq_enc_per_token,-2,-1).shape} +  {to_save_output_symbols.shape}")
            
            seq_enc_plus_out = torch.matmul(torch.transpose(to_save_seq_enc_per_token,-2,-1), to_save_output_symbols) 
        else:
            if to_save_sequences is not None:
                to_save_sequence_encoding = sequence_encoding[:, to_save_sequences]
            else:
                to_save_sequence_encoding = sequence_encoding
            stored_keys = to_save_sequence_encoding
            
            if DEBUGGING:
                print(f"seq_enc_plus_out : {to_save_sequence_encoding.shape} {torch.transpose(to_save_sequence_encoding,-2,-1).shape} +  {to_save_output_symbols.shape}")
            
            seq_enc_plus_out = torch.matmul(torch.transpose(to_save_sequence_encoding,-2,-1), to_save_output_symbols)
        
        if self.layerized_CMM_OUT: 
            self.CMM_OUT.memorize(seq_enc_plus_out)  ### FMZ 2026:05:04
            # prior (token counts), load P, gamma and mean ||v||^2: updated with the pairs just stored
            self._update_statistics(stored_keys, to_save_output_symbols, to_save_output_ids)
        
        return sequence_encoding, seq_enc_plus_out
 
 
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
            
            # raw recall r = M k: the quantity the Bayes decoder (temperature, prior) is calibrated for
            layered_out_raw = layered_out_token
            layered_out_token = F.normalize(layered_out_token, p=2, dim=-1)
            
            if DEBUGGING:
                print(f"Adding layered_out_token {layered_out_token.shape}")
        else:
            layered_out_token = None
            layered_out_raw = None
            
        
        if seq_enc_per_token is not None:
            token_encoding = seq_enc_per_token[:, -1].reshape(batch_size,self.d)
        else:
            token_encoding = None
        
        return dict(
                sequence_encoding=sequence_encoding, 
                token_encoding=token_encoding,#, locally_predicted
                layered_out_token=layered_out_token,
                layered_out_raw=layered_out_raw,
                cache=None 
        )
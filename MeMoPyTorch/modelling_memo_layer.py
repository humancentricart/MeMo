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
# self.Prj = torch.normal(0, 1/math.sqrt(self.d*self.h), size=(self.d,self.d*self.h))
# self.Prj = torch.transpose(self.Prj, 0, 1)

# as in Linear torch.nn.modules.linear


DEBUGGING = False
verbose = False


if DEBUGGING:
    import matplotlib.pyplot as plt

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
        in_features: int, # d * h
        out_features: int, #d
        bias: bool = True,
        device=None,
        dtype=None,
        init_weights=True
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
        #print(f"Sequence shape : {input.shape} CMM Shape: {self.weight.T.shape}")
        return F.linear(input, self.weight.T)
        #return F.linear(input, self.weight)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"

    def memorize(self, update):
        #print(f"UPDATE SHAPE 1: {update.shape}")
        if len(update.shape) ==  3:
            update = torch.sum(update, dim=0) #handle batch update
        #print(f"UPDATE SHAPE 2: {update.shape}")
        
        with torch.no_grad():
            self.weight += update #in place update

    def forget(self, update):
        if len(update.shape) ==  3:
            update = torch.sum(update, dim=0) #handle batch update
            
        with torch.no_grad():
            self.weight -= update #in place update




class MeMoLayer(Module):
    
    def __init__(self, inner_dim, num_of_heads, init_weights=True, is_last=False, alpha=1, **kwargs):
        super().__init__()

        self.alpha = alpha # Computed vs. memorized sequence encoding (alpha = 1 only computed)
        self.d = inner_dim
        self.h = num_of_heads
        self.d_k = self.d // self.h
        if self.d / self.h != self.d_k:
            raise MeMoException("Inner dimension " + str(self.d) + " should be divisible for number of heads " + str(self.h))

        self.W_v_single_head = ProjectionTokens(self.d, self.d_k, init_weights=init_weights)
        self.use_local_CMM = (alpha < 1)
        
        
        self.Prj = ProjectionSequence(self.d, self.d*self.h, init_weights=init_weights)
        # CMM : correlation matrix memory for the specific layer
        if self.use_local_CMM or is_last:
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
        #print(f"Shape input sequences : {sequence_encoding.shape} - {all_sequences.shape}")
        # OLD : all_sequences = all_sequences.reshape(batch_size, self.d, 1)
        #print(all_sequences.shape) #(batch_size, d, 1)
        all_sequences = torch.sum(all_sequences, dim=0)

        # as in MeMoCMM
        #all_sequences = torch.sum(sequence_encoding, dim=0)

        
        all_sequences = all_sequences.reshape(self.d, 1)
        #print(all_sequences.shape) #(batch_size, d, 1)
       
        stored_sequences_filter = 1 - torch.round(torch.matmul(self.CMM(seq_enc_per_token), all_sequences))
        #stored_sequences_filter = 1 - torch.sum(torch.round(torch.matmul(sequence_encoding, torch.transpose(sequence_encoding, -2 ,-1))),dim=2)
        stored_sequences_filter[stored_sequences_filter < 0] = 0.0 # Sequences may appear more than once in all_sequences
        
        #print(stored_sequences_filter.shape)
        #new_sequences = torch.round(torch.matmul(sequence_encoding, all_sequences))
        new_sequences = torch.sum(torch.round(torch.matmul(sequence_encoding, torch.transpose(sequence_encoding, -2 ,-1))),dim=2)
        (batch,no_elem) = new_sequences.shape
        new_sequences = new_sequences.reshape((batch,no_elem,1))
        #print(new_sequences.shape)
        
        #### Penalizing factors : provided that it is correct, elements with 0 are problematic. For the moment,
        # there is a correcting added of 0.001 to avoid 0 elements.
        #print(stored_sequences_filter.shape, new_sequences.shape)
        penalizing_factors = 1/(stored_sequences_filter*new_sequences)
        
        penalizing_factors[penalizing_factors == math.inf] = 0
        penalizing_factors[penalizing_factors == -math.inf] = 0

        if DEBUGGING:
            print(f"Penalizing Factors Len: {penalizing_factors.shape} : {torch.sum(penalizing_factors,dim=1)}")
            #(a,b,c) = penalizing_factors.shape
            #to_display = torch.sum(torch.round(torch.matmul(sequence_encoding, torch.transpose(sequence_encoding, -2 ,-1))),dim=2)
            #print(f"{sequence_encoding.shape} * {torch.transpose(sequence_encoding, -2 ,-1).shape} = {to_display.shape}" )
            #print(f"new_sequences matrix : {to_display}")
            #print(torch.matmul(sequence_encoding, all_sequences).reshape((a*b*c)))
            #print(stored_sequences_filter.reshape((a*b*c)))
            #print(penalizing_factors.reshape((a*b*c)))
        return sequence_encoding * penalizing_factors
        #return sequence_encoding

    
    
    # The most simple implementation
    # Input sequence is has a shape of (self.h,self.d), that is self.h sequences are proposed as input rows
    def memorize(self, input_sequence, output_symbols, is_last = False):
        (batch_size, blocks,h,d) = input_sequence.shape
        sequence_encoding, seq_enc_per_token = self.get_projections(input_sequence, blocks, h, d)
        penalization = False
        # Updating local CMM
        if not is_last:
            if penalization: 
                surviving_vectors = self.penalize(sequence_encoding, seq_enc_per_token)
            else:
                surviving_vectors = sequence_encoding
            if self.use_local_CMM:
                CMM_update = torch.matmul(torch.transpose(seq_enc_per_token, -2, -1), surviving_vectors)/np.power(1.06,batch_size)
                self.CMM.memorize(CMM_update)

            if DEBUGGING:
                #retrieved_sequence_encoding = self.CMM(seq_enc_per_token)
                L_CMM = torch.sum(CMM_update,dim=0)
                retrieved_sequence_encoding = torch.matmul(seq_enc_per_token,L_CMM)
                print(f"{seq_enc_per_token.shape} * {L_CMM.shape} = {retrieved_sequence_encoding.shape}")
                DEB_OUT = torch.matmul(retrieved_sequence_encoding,torch.transpose(surviving_vectors, -2, -1))
                #DEB_OUT = torch.matmul(seq_enc_per_token,torch.transpose(seq_enc_per_token, -2, -1),)
                #(_,NoSeqs,_) = DEB_OUT.shape  
                diagonals = DEB_OUT.diagonal(dim1=1, dim2=2)
                #print(f"{retrieved_sequence_encoding.shape} - {surviving_vectors.shape} {torch.transpose(surviving_vectors, -2, -1).shape} | What has been stored - SHAPE {DEB_OUT.shape} \n {diagonals}")
                data = diagonals.to('cpu').view(-1).detach().numpy()
                print(f"{DEB_OUT.shape} - - {data.shape}" )

                # Plot the distribution using a histogram
                plt.hist(data, bins=30, density=True, alpha=0.7, color='blue')
                plt.title("Distribution of Tensor Values")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.show()
            
        #### To be adjusted for taking into consideration the entire sequence
        if DEBUGGING: 
            print(f"seq_enc_plus_out : {seq_enc_per_token.shape} {torch.transpose(seq_enc_per_token,-2,-1).shape} +  {output_symbols.shape}")
            
        seq_enc_plus_out = torch.matmul(torch.transpose(seq_enc_per_token,-2,-1), output_symbols) 
        ## Key (sequenze di h token) x Value ==> matrice??
        #self.CMM_OUT.memorize(seq_enc_plus_out)
        
        return sequence_encoding, seq_enc_plus_out
    
    def directly_memorize(self, input_sequence):
        self.CMM.memorize(input_sequence)
        #print(self.CMM.weight)

    def forget(self, input_sequence, output_symbols, completely=False, is_last = False):
        (batch_size, blocks,h,d) = input_sequence.shape
        sequence_encoding, seq_enc_per_token = self.get_projections(input_sequence, blocks, h, d)
        
        # Updating local CMM
        if not is_last and completely:
            surviving_vectors = self.penalize(sequence_encoding, seq_enc_per_token)
            CMM_update = torch.matmul(torch.transpose(seq_enc_per_token, -2, -1), surviving_vectors)
            self.CMM.forget(CMM_update)


        #### To be adjusted for taking into consideration the entire sequence
        seq_enc_plus_out = torch.matmul(torch.transpose(seq_enc_per_token,-2,-1), output_symbols) 
        ## Key (sequenze di h token) x Value ==> matrice??
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
        
        #(batch_size, blocks, _, _) = input_sequence.shape
        
        (batch_size, blocks,h,d) = input_sequence.shape
        sequence_encoding, seq_enc_per_token = self.get_projections(input_sequence, blocks, h, d)

        #OLD VERSION: seq_enc_per_token = self.W_v_single_head(input_sequence).reshape((batch_size, blocks, self.d)) / math.sqrt(self.h)
        # print(seq_enc_per_token.shape) (batch_size, blocks, d)
        if self.use_local_CMM:
            retrieved_sequence_encoding = self.CMM(seq_enc_per_token)

        #TODO check and rewrite
        #if verbose:
        #    sequence_encoding = self.Prj(input_sequence.reshape((batch_size, blocks, self.d * self.h)))
        #    out = torch.matmul(sequence_encoding, torch.transpose(retrieved_sequence_encoding,-2,-1))
        #    out1 = torch.linalg.norm(seq_enc_per_token, dim=-1)
        #    
        #    if torch.max(out1).item() > 1.5 or torch.max(out).item()> 1.5:
        #        print("Errore")
        # last token seq_enc_per_token[-1].reshape(batch_size,self.d)
        #locally_predicted = self.CMM_OUT(seq_enc_per_token[-1].reshape(1,self.d))

        #return retrieved_sequence_encoding, seq_enc_per_token[:, -1].reshape(batch_size,self.d)#, locally_predicted
        #return sequence_encoding, seq_enc_per_token[:, -1].reshape(batch_size,self.d)#, locally_predicted
        if self.use_local_CMM:
            return self.alpha*sequence_encoding+(1-self.alpha)*retrieved_sequence_encoding, seq_enc_per_token[:, -1].reshape(batch_size,self.d)#, locally_predicted
        else:
             return sequence_encoding, seq_enc_per_token[:, -1].reshape(batch_size,self.d)#, locally_predicted

    def directly_retrieve(self,vector):
        return self.CMM(vector)


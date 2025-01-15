from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F, init, Module, ModuleList
from torch.nn.parameter import Parameter

from .modelling_memo_embedding import MeMoEmbedding
from .modelling_memo_layer import MeMoLayer
from .modelling_memo_exception import MeMoException

import math


DEBUGGING = False

VERBOSE = False
DEVICE = 'cpu'

class MeMo(Module):
    def __init__(self, inner_dim, num_of_heads, num_of_layers, chunk_length, 
                 num_embeddings, padding_idx=0, device=None):
        super().__init__()
        
        self.d = inner_dim
        self.h = num_of_heads
        self.l = num_of_layers
        self.max_len = self.h**self.l
        self.chunk_length = chunk_length
        
        if self.chunk_length/self.max_len != self.chunk_length//self.max_len:
            raise MeMoException("Chunk length "+ str(self.chunk_length) + \
                " should be divisible for number of heads power numer of layers ("+str(self.max_len) +")")

        self.device = device if device is not None else DEVICE
        
        self.encoder = MeMoEmbedding(num_embeddings, self.d, padding_idx=padding_idx, device=self.device)
        self.layers = ModuleList([MeMoLayer(self.d, self.h) for _ in range(num_of_layers)])

        self.to(self.device)
    


    # The most simple implementation
    # Input sequence has a shape of (self.h**self.l,self.d), that is self.h sequences are proposed as input rows
    def memorize(self, input_sequence_ids, labels_ids):
        input_sequence = self.encoder.encode(input_sequence_ids)
        output_symbols = self.encoder.encode(labels_ids)
        ##print("input_sequence.shape", input_sequence.shape)

        (batch_size, current_length, d) = input_sequence.shape
        assert (current_length == self.chunk_length), f'check tokenization of input text, expected row of {self.chunk_length} tokens'
        
        last_layer = self.layers[self.l-1]
        current_length = self.chunk_length
        
        #for layer_level in range(self.l):
        #    current_length = current_length//self.h
        #    input_sequence = input_sequence.reshape((batch_size, current_length, self.h, self.d))
        #
        #    ##print(f"per layer {layer_level} input_sequence.shape", input_sequence.shape)
        #    ##print(input_sequence[0])    
        #    output_symbols = output_symbols[:, [(x+1)*self.h-1 for x in range(0,current_length)]] ## the output symbol is always the same tokem?

        for layer_level in range(self.l):
            if self.h ** (layer_level + 1) < current_length + 1:
                ## update the input sequence for the next layer
                layer_output_idxs = [i - self.h ** ((layer_level + 1) - 1) for i in range(self.h ** (layer_level + 1), current_length + 1)]
                output_symbols = output_symbols[:, layer_output_idxs]
                #print(output_symbols.shape)
                
                input_index = [[j for j in range(i - self.h ** (layer_level + 1), i, self.h ** ((layer_level + 1) - 1))] 
                               for i in range(self.h ** (layer_level + 1), current_length + 1)]
                input_sequence = input_sequence[:, input_index]
                
                if DEBUGGING:
                    retreived_output_symbol_vector, max_value = self.encoder.decode(output_symbols)
                    #print(retreived_output_symbol_vector)
                
                ## update the input sequence for the next layer
                input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].memorize(input_sequence, 
                                                                                                    output_symbols, 
                                                                                                    is_last=(layer_level == self.l-1))
                last_layer.directly_memorize(seq_encoding_for_the_last_layer)
            else:
                break

    def memorize_text(self, memo_input):
        #for i in range(0, self.h):
        self.memorize(memo_input['input_ids'].to(self.device), 
                      memo_input['labels'].to(self.device))
        
    
    def forget(self, input_sequence_ids, labels_ids, completely=True):
        input_sequence =  self.encoder.encode(input_sequence_ids)
        output_symbols = self.encoder.encode(labels_ids)

        (batch_size, current_length, d) = input_sequence.shape
        assert (current_length == self.chunk_length), f'check tokenization of input text, expected row of {self.chunk_length} tokens'
        
        last_layer = self.layers[self.l-1]
        current_length = self.chunk_length
        
        for layer_level in range(self.l):
            #current_length = current_length//self.h
            #input_sequence = input_sequence.reshape((batch_size, current_length, self.h, self.d))
        
            ##print(f"per layer {layer_level} input_sequence.shape", input_sequence.shape)
            ##print(input_sequence[0])    
            #output_symbols = output_symbols[:, [(x+1)*self.h-1 for x in range(0, current_length)]] ## the output symbol is always the same tokem?

            
            if self.h ** (layer_level + 1) < current_length + 1:
                ## update the input sequence for the next layer
                layer_output_idxs = [i - self.h ** ((layer_level + 1) - 1) for i in range(self.h ** (layer_level + 1), current_length + 1)]
                output_symbols = output_symbols[:, layer_output_idxs]
                #print(output_symbols.shape)
                
                input_index = [[j for j in range(i - self.h ** (layer_level + 1), i, self.h ** ((layer_level + 1) - 1))] 
                               for i in range(self.h ** (layer_level + 1), current_length + 1)]
                input_sequence = input_sequence[:, input_index]
                                
                if DEBUGGING:
                    retreived_output_symbol_vector, max_value = self.encoder.decode(output_symbols)
                    #print(retreived_output_symbol_vector)
                
                input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].forget(input_sequence, 
                                                                                                  output_symbols, 
                                                                                                  completely=completely,
                                                                                                  is_last= (layer_level == self.l-1) )
                
                last_layer.directly_forget(seq_encoding_for_the_last_layer)
            else:
                break
        
        
    
    def forget_text(self, memo_input, completely=True):
        #for i in range(0,self.h):
        self.forget(memo_input['input_ids'].to(self.device),
                    memo_input['labels'].to(self.device), 
                    completely=completely)

    def retrieve(self, input_sequence_ids):
        input_sequence = self.encoder(input_sequence_ids)
        last_layer = self.layers[self.l-1]

        (batch_size, current_length, d) = input_sequence.shape
        assert (current_length == self.chunk_length), f'check tokenization of input text, expected row of {self.chunk_length} tokens'
        
        encoding_for_the_last_layer = torch.zeros((batch_size, self.d)).to(self.device)
        current_length = self.chunk_length #min(self.chunk_length, self.max_len)

        # TODO moved outside the logic for tokenization, here only assertiion above
        #if len(input_sequence) > current_length:
        #    input_sequence = input_sequence[len(input_sequence)-current_length:len(input_sequence)]
        
        for layer_level in range(self.l):
            current_length = int(current_length/self.h)
            input_sequence = input_sequence.reshape((batch_size, current_length, self.h, self.d))
            
            input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].retrieve(input_sequence)
            encoding_for_the_last_layer += seq_encoding_for_the_last_layer


            # TODO ckec
            #### DEBUGGING
            #if debugging:
            #    retreived_output_symbol_vector_APPO, max_APPO = self.encoder.decode(last_layer.directly_retrieve(seq_encoding_for_the_last_layer))
            #    #print(f"OUT APPO: {retreived_output_symbol_vector_APPO} level: {layer_level}")
            #    retreived_output_symbol_vector_APPO, max_APPO = self.encoder.decode(locally_predicted)
            #    #print(f"LOCALLY APPO: {retreived_output_symbol_vector_APPO} level: {layer_level}")

            # TODO check
            #if VERBOSE:
            #    retreived_output_symbol_vector, score_max = self.encoder.decode(last_layer.directly_retrieve(seq_encoding_for_the_last_layer).unsqueeze(0))
            #    #print(f"NORM OF THE VECTOR:", torch.linalg.norm(seq_encoding_for_the_last_layer))
            #    #print((retreived_output_symbol_vector, score_max))

        retreived_output_symbol_vector, score_max = self.encoder.decode(last_layer.directly_retrieve(encoding_for_the_last_layer))
        return retreived_output_symbol_vector, score_max

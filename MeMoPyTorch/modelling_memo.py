from json import encoder
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F, init, Module, ModuleList
from torch.nn.parameter import Parameter

from .modelling_memo_embedding import MeMoEmbedding
from .modelling_memo_layer import MeMoLayer
from .modelling_memo_layer import CompositionOp
from .modelling_memo_exception import MeMoException

import math



DEBUGGING = False

VERBOSE = False
DEVICE = 'cpu'

class MeMo(Module):
    def __init__(self, inner_dim, num_of_heads, num_of_layers, chunk_length, 
                 num_embeddings, padding_idx=0, padding_seq_idx=-1, device=None, alpha_gen = 1, compositionOp = CompositionOp.Prod):  # num_embedding must be equal to number of tokens + 2
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
        
        self.encoder = MeMoEmbedding(num_embeddings, self.d, padding_idx=padding_idx, padding_seq_idx=padding_seq_idx, device=self.device)
        self.layers = ModuleList([MeMoLayer(self.d, self.h, alpha=alpha_gen, compositionOp = compositionOp, is_last=(i +1 == num_of_layers)) for i in range(num_of_layers)])

        self.to(self.device)
    
    
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
    
    def memorize(self, input_sequence_ids, labels_ids):
        input_sequence = self.encoder.encode(input_sequence_ids)
        output_symbols = self.encoder.encode(labels_ids)

        (batch_size, current_length, d) = input_sequence.shape
        #assert (current_length == self.chunk_length), f'check tokenization of input text, expected row of {self.chunk_length} tokens'
        
        last_layer = self.layers[self.l-1]
        
        
        for layer_level in range(self.l):
            
            input_sequence = self.generate_sequences(input_seqs=input_sequence, layer=layer_level)
            
            if DEBUGGING:
                if layer_level == 0:
                    print("Decoding input_sequence at layer 0")
                    print(self.encoder.decode(input_sequence))
                retreived_output_symbol_vector, max_value = self.encoder.decode(output_symbols)
                print(f"Layer {layer_level} - Output Sequence: {retreived_output_symbol_vector}")
            
            ## update the input sequence for the next layer
            input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].memorize(input_sequence, 
                                                                                                output_symbols, 
                                                                                                is_last=(layer_level == self.l-1))
            last_layer.directly_memorize(seq_encoding_for_the_last_layer)


    def memorize_text(self, memo_input):
        self.memorize(memo_input['input_ids'].to(self.device), 
                      memo_input['labels'].to(self.device))
        
    
    def forget(self, input_sequence_ids, labels_ids, completely=True):
        input_sequence = self.encoder.encode(input_sequence_ids)
        output_symbols = self.encoder.encode(labels_ids)

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

    def retrieve(self, input_sequence_ids):
        input_sequence = self.encoder.encode(input_sequence_ids)

        (batch_size, current_length, d) = input_sequence.shape
        #assert (current_length == self.chunk_length), f'check tokenization of input text, expected row of {self.chunk_length} tokens'
        

        encoding_for_the_last_layer = torch.zeros((batch_size, self.d)).to(self.device)
        #current_length = self.max_len #min(self.chunk_length, self.max_len)

        last_layer = self.layers[self.l-1]
        
        
        for layer_level in range(self.l):
            input_sequence = self.generate_sequences(input_seqs=input_sequence, layer=layer_level)

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
        if DEBUGGING:
            if min(score_max) < 0.1:
                print(f"Symbol: {retreived_output_symbol_vector} {score_max}")
        return retreived_output_symbol_vector, score_max

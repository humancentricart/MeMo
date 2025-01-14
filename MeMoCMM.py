import torch
import math
import torch.nn.functional as F


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

verbose = False
debugging = False


empty_token = "<N>"

def main():
    # the first attempt
    d,h,l = 1024, 2, 6
    torch.manual_seed(1)
    #random.seed(1)
    with open("testo_di_prova3.txt", encoding="utf8") as my_first_text_f:
        my_first_text_tokenized = my_first_text_f.read().split()

    memo = MeMo(d,h,l,800,list(set(my_first_text_tokenized)))

    memo.memorize_text(my_first_text_tokenized)

    '''
    seq2 = ['1', '2', '3', '4']
    #seq2 = ['7', '8', '1', '2']
    print(f"IN :{seq2}")
    max_value, out = memo.retrieve(seq2)
    print(f"OUT :{max_value} - {out}")

    '''

    # EVALUATE
    e = Evaluation()
    out = e.check_memorization(memo,my_first_text_tokenized)
    print("Degree of memorization: %f ", out)

    #memo.forget_text(my_first_text_tokenized,completely=False)
    #out = e.check_memorization(memo,my_first_text_tokenized)
    #print("Degree of memorization: %f ", out)


    '''
    memo.memorize(my_first_text_tokenized[10:18])
    #memo.memorize(my_first_text_tokenized[11:20])
    print(my_first_text_tokenized[9:20])
    max_value , out = memo.retrieve(my_first_text_tokenized[9:17])
    print("Emitted  : " + out + " " + str(max_value))
    print("Expected : " + my_first_text_tokenized[17])
    seq2 = ['27', '27', '27', '27', '27', '27', '27', '27']
    print(seq2)
    max_value, out = memo.retrieve(seq2)
    print("Emitted  : " + out + " " + str(max_value))
    print("Expected : " + my_first_text_tokenized[17])
    '''



class MeMoException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__("MeMo Error: " + self.message)

class MeMoEncoder:
    def __init__(self, inner_dim, h, dictionary=None, max_lenght = 10):
        self.d = inner_dim
        self.h = h
        # Encoding vectors for tokens - the first row is a zero row for padding
        #self.dict_enc = torch.normal(0, 1 / math.sqrt(self.d), size=(len(dictionary) + 1, self.d))
        dictionary = [empty_token] + dictionary
        self.dict_enc = torch.normal(0, 1 / math.sqrt(self.d), size=(len(dictionary), self.d))
        self.dict_enc[0] = torch.zeros(self.d)
        #self.dict_enc[0] = self.dict_enc[0]*0.01
        self.dictionary = dictionary
        self.max_len = max_lenght
        if debugging: print(dictionary)


    def encode(self,sequence,only_input=False):
        if only_input:
            l  = len(sequence)
            input_ = self.dict_enc[[0 for _ in range(0,self.max_len - l)] +\
                                   [self.dictionary.index(x) for x in sequence[max(0,l-self.max_len):l]]]
            #print("Representation of the input:\n" +
            #      str([0 for _ in range(0,self.max_len - l)] +\
            #                       [x for x in sequence[max(0,l-self.max_len):l]]))
            output_ = None
        else:
            l  = len(sequence)
            input_s = [0 for _ in range(0,self.max_len - l+1 )] +\
                                   [self.dictionary.index(x) for x in sequence[max(0,l-self.max_len-1):l-1]]
            input_ = self.dict_enc[input_s]
            if debugging: print(f"Representation of the input:\n{str(input_s)}")
            l = l + 1
            output_s = [0 for _ in range(0,self.max_len - l+1 )] +\
                                   [self.dictionary.index(x) for x in sequence[max(0,l-self.max_len-1):l]]
            output_ = self.dict_enc[output_s]
            if debugging: print(f"Representation of the input:\n{str(output_s)}")
        if l > self.max_len + self.h :
            print(f"Text exceeding chunk length {l} vs {self.max_len + self.h - 1}")

        return input_,output_

    def decode_multi(self,vector):
        out = torch.matmul(self.dict_enc,torch.transpose(vector,0,1))
        #for i in range(0,len(self.dictionary)):
        #    print(f"{self.dictionary[i]}:{out[i+1].item():.4f}", end=",")
        #print()
#        out = F.softmax(out*10,dim=0)
#        for i in range(0,len(self.dictionary)):
#            print(f"{self.dictionary[i]}:{out[i+1].item():.4f}", end=",")
#        print()
        #print(str(torch.argmax(out[1:])))
        #return self.dictionary[torch.argmax(out) - 1], max(out)

        #return self.dictionary[torch.argmax(out[1:])], max(out)
        massimi = [x.item() for x in torch.argmax(out, dim=0)]
        return [self.dictionary[m] for m in massimi], 0.0 #, max(out)

    def decode(self,vector):
        out = torch.matmul(self.dict_enc,torch.transpose(vector,0,1))
        if debugging:
            for (w,v) in sorted(zip(self.dictionary,out), key=lambda x:x[1], reverse=True)[:3]:
                print(f"{w}:{v.item():.4f}", end=",")
            #for i in range(0,len(self.dictionary)):
            #    print(f"{self.dictionary[i]}:{out[i].item():.4f}", end=",")
            print()
#        out = F.softmax(out*10,dim=0)
#        for i in range(0,len(self.dictionary)):
#            print(f"{self.dictionary[i]}:{out[i+1].item():.4f}", end=",")
#        print()
        #print(str(torch.argmax(out[1:])))
        #return self.dictionary[torch.argmax(out) - 1], max(out)

        return self.dictionary[torch.argmax(out[1:])+1], max(out)
        #massimi = [x.item() for x in torch.argmax(out, dim=0)]
        #return [self.dictionary[m] for m in massimi], 0.0 #, max(out)


class MeMoLayer:
    def __init__(self,inner_dim, num_of_heads):
        self.d = inner_dim
        self.h = num_of_heads
        self.d_k = self.d // self.h
        if self.d / self.h != self.d_k:
            raise MeMoException("Inner dimension " + str(self.d) + " should be divisible for number of heads " + str(self.h))
        # W_v for a single head - at the very end, there should be num_of_heads W_v_single_head matrices or a W_v inner_dim x inner_dim
        self.W_v_single_head = torch.normal(0, 1/math.sqrt(self.d_k), size=(self.d,self.d_k))

        # Prj : projection matrix that determines the representation of a sequence
        self.Prj = torch.normal(0, 1/math.sqrt(self.d*self.h), size=(self.d,self.d*self.h))
        self.Prj = torch.transpose(self.Prj, 0, 1)
        # CMM : correlation matrix memory for the specific layer
        self.CMM = torch.zeros(size=(self.d,self.d), requires_grad=False)
        # CMM OUT : correlation matrix memory for the specific layer
        self.CMM_OUT = torch.zeros(size=(self.d,self.d), requires_grad=False)

    # The most simple implementation
    # Input sequence is has a shape of (self.h,self.d), that is self.h sequences are proposed as input rows
    def memorize(self, input_sequence,output_symbols, is_last = False):

        (blocks,h,d) = input_sequence.shape
        # shape (blocks,self.d)
        # WITH PROJECTION MATRIX
        sequence_encoding = torch.matmul(input_sequence.reshape((blocks,self.d * self.h)),self.Prj)

        # WITH AN APPROXIMATION OF SHUFFLED CIRCULAR CONVOLUTION (Sequences with the same tokens in different order will
        # have the same representation)
        #sequence_encoding = torch.prod(input_sequence, dim=1)
        #sequence_encoding = F.normalize(sequence_encoding, p=2, dim=1)

        #APPO = torch.matmul(sequence_encoding,sequence_encoding.T)

        # shape (blocks,self.d)
        seq_enc_per_token = torch.matmul(input_sequence,self.W_v_single_head) / math.sqrt(self.h)
        seq_enc_per_token = seq_enc_per_token.reshape((blocks,self.d))
        # Updating local CMM
        if not is_last:
            # Penalizing factors to avoid multiple storage of the same sequence encoding in intermediate CMMs
            # This penalizing factor should merge the repetitions founds in the current sequence and the already stored
            # sequences in the current CMM
            # dimension = (blocks,1)
            #APPO = torch.matmul(sequence_encoding, sequence_encoding.T)

            all_sequences = torch.sum(sequence_encoding, dim=0)
            stored_sequences_filter = 1 - torch.round(torch.matmul(torch.matmul(seq_enc_per_token, self.CMM),all_sequences.reshape(self.d,1)))
            stored_sequences_filter[stored_sequences_filter < 0] = 0
            new_sequences = torch.round(torch.matmul(sequence_encoding, all_sequences.reshape(self.d, 1)))

            #### Penalizing factors : provided that it is correct, elements with 0 are problematic. For the moment,
            # there is a correcting added of 0.001 to avoid 0 elements.
            penalizing_factors = 1/(stored_sequences_filter*new_sequences)
            penalizing_factors[penalizing_factors == math.inf] = 0
            penalizing_factors[penalizing_factors == -math.inf] = 0
            surviving_vectors = sequence_encoding * penalizing_factors
            self.CMM = self.CMM + torch.matmul(torch.transpose(seq_enc_per_token, 0, 1), surviving_vectors )


        #### To be adjusted for taking into consideration the entire sequence
        seq_enc_plus_out = torch.matmul(torch.transpose(seq_enc_per_token,0,1),output_symbols)
        self.CMM_OUT = self.CMM_OUT + seq_enc_plus_out


        return sequence_encoding, seq_enc_plus_out
    def directly_memorize(self, input_sequence):
        self.CMM = self.CMM + input_sequence

    def forget(self, input_sequence,output_symbols, completely=True, is_last = False):
        '''#TO REWORK
        (blocks,_,_) = input_sequence.shape
        sequence_encoding = torch.matmul(input_sequence.reshape((blocks,self.d * self.h)),self.Prj)
        seq_enc_per_token = torch.matmul(input_sequence[-1],self.W_v_single_head).reshape((self.d,1)) / math.sqrt(self.h)
        if verbose:
            norm1 = torch.linalg.norm(input_sequence.reshape((blocks,self.d * self.h)), dim=1)
            norm2 = torch.linalg.norm(sequence_encoding, dim=1)
            norm3 = torch.linalg.norm(seq_enc_per_token, dim=0)
        # Updating local CMM if a complete forget is required
        if completely:
            self.CMM = self.CMM - torch.matmul(seq_enc_per_token,sequence_encoding[-1].reshape(1,self.d))
        seq_enc_plus_out = torch.matmul(seq_enc_per_token,torch.transpose(output_symbols,0,1))
        '''
        (blocks,h,d) = input_sequence.shape
        # shape (blocks,self.d)
        # WITH PROJECTION MATRIX
        sequence_encoding = torch.matmul(input_sequence.reshape((blocks,self.d * self.h)),self.Prj)
        # WITH AN APPROXIMATION OF SHUFFLED CIRCULAR CONVOLUTION (Sequences with the same tokens in different order will
        # have the same representation)
        #sequence_encoding = torch.prod(input_sequence, dim=1)
        #sequence_encoding = F.normalize(sequence_encoding, p=2, dim=1)

        # shape (blocks,self.d)
        seq_enc_per_token = torch.matmul(input_sequence,self.W_v_single_head) / math.sqrt(self.h)
        seq_enc_per_token = seq_enc_per_token.reshape((blocks,self.d))
        # Updating local CMM
        if not is_last and completely:
            # Penalizing factors to avoid multiple storage of the same sequence encoding in intermediate CMMs
            # This penalizing factor should merge the repetitions founds in the current sequence and the already stored
            # sequences in the current CMM
            # dimension = (blocks,1)
            all_sequences = torch.sum(sequence_encoding, dim=0)
            stored_sequences_filter = 1 - torch.round(torch.matmul(torch.matmul(seq_enc_per_token, self.CMM),all_sequences.reshape(self.d,1)))
            new_sequences = torch.round(torch.matmul(sequence_encoding, all_sequences.reshape(self.d, 1)))

            #### Penalizing factors : provided that it is correct, elements with 0 are problematic. For the moment,
            # there is a correcting added of 0.001 to avoid 0 elements.
            penalizing_factors = 1/(stored_sequences_filter*new_sequences)
            penalizing_factors[penalizing_factors == math.inf] = 0
            penalizing_factors[penalizing_factors == -math.inf] = 0
            surviving_vectors = sequence_encoding * penalizing_factors
            self.CMM = self.CMM - torch.matmul(torch.transpose(seq_enc_per_token, 0, 1), surviving_vectors )


        #### To be adjusted for taking into consideration the entire sequence
        seq_enc_plus_out = torch.matmul(torch.transpose(seq_enc_per_token,0,1),output_symbols)

        return sequence_encoding, seq_enc_plus_out

    def directly_forget(self, input_sequence):
        self.CMM = self.CMM - input_sequence

    def retrieve(self,input_sequence):
        (blocks,_,_) = input_sequence.shape
        seq_enc_per_token = torch.matmul(input_sequence,self.W_v_single_head).reshape((blocks,self.d)) / math.sqrt(self.h)
        retrieved_sequence_encoding = torch.matmul(seq_enc_per_token,self.CMM)
        computed_sequence_encoding = torch.matmul(input_sequence.reshape((blocks,self.d * self.h)),self.Prj)
        if verbose:
            sequence_encoding = torch.matmul(input_sequence.reshape((blocks, self.d * self.h)), self.Prj)
            out = torch.matmul(sequence_encoding,torch.transpose(retrieved_sequence_encoding,0,1))
            out1 = torch.linalg.norm(seq_enc_per_token,dim=1)
            if torch.max(out1).item() > 1.5 or torch.max(out).item()> 1.5:
                print("Errore")
        APPO_TEST = seq_enc_per_token[-1].reshape(1,self.d)

        locally_predicted = torch.matmul(seq_enc_per_token[-1].reshape(1,self.d),self.CMM_OUT)

        return computed_sequence_encoding, seq_enc_per_token[-1].reshape(1,self.d), locally_predicted
        #return retrieved_sequence_encoding, seq_enc_per_token[-1].reshape(1,self.d), locally_predicted
        #return (retrieved_sequence_encoding+computed_sequence_encoding)/2, seq_enc_per_token[-1].reshape(1,self.d), locally_predicted

    def directly_retrieve(self,vector):
        return torch.matmul(vector,self.CMM)

class MeMo:
    def __init__(self, inner_dim, num_of_heads, num_of_layers, chunk_length,token_set_in_list):
        self.d = inner_dim
        self.h = num_of_heads
        self.l = num_of_layers
        self.max_len = self.h**self.l
        self.chunk_length = chunk_length
        #if self.chunk_length/self.max_len != self.chunk_length//self.max_len:
        #    raise MeMoException("Chunk length "+ str(self.chunk_length) + \
        #        " should be divisible for number of heads power numer of layers ("+str(self.max_len) +")")
        self.layers = [MeMoLayer(inner_dim,num_of_heads) for _ in range(num_of_layers)]
        self.encoder = MeMoEncoder(inner_dim,self.h, dictionary=token_set_in_list, max_lenght=self.chunk_length)


    # The most simple implementation
    # Input sequence has a shape of (self.h**self.l,self.d), that is self.h sequences are proposed as input rows
    def memorize(self,input_sequence_w):
        input_sequence, output_symbols = self.encoder.encode(input_sequence_w)
        last_layer = self.layers[self.l-1]
        current_length = self.chunk_length

        for layer_level in range(self.l):
            #current_length = int(current_length/self.h)

            #input_sequence = input_sequence.reshape((current_length, self.h, self.d))
            #output_symbols = output_symbols[[(x+1)*self.h-1 for x in range(0,current_length)]]

            if self.h ** (layer_level + 1) < current_length + 1:
                #APPO = [output_symbols[i - heads ** (layer - 1)] for i in range(heads ** layer, sequenc_len)]
                output_symbols = torch.stack([output_symbols[i - self.h ** ((layer_level + 1) - 1)] for i in range(self.h ** (layer_level + 1), current_length + 1)])
                input_sequence = torch.stack([torch.stack([input_sequence[j] for j in range(i - self.h ** (layer_level + 1), i, self.h ** ((layer_level + 1) - 1))]) for i in
                         range(self.h ** (layer_level + 1), current_length + 1)])

                if debugging:
                    retreived_output_symbol_vector, max = self.encoder.decode_multi(output_symbols)
                    print(retreived_output_symbol_vector)

                input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].memorize(input_sequence, output_symbols, is_last=(last_layer == self.layers[layer_level]))
                last_layer.directly_memorize(seq_encoding_for_the_last_layer)
            else:
                break

    def memorize_text(self,tokenized_text):
        self.memorize(tokenized_text)
        #for i in range(0,self.h):
        #    self.memorize(tokenized_text[:len(tokenized_text)-i])


    def forget(self,input_sequence_w, completely=True):
        input_sequence, output_symbols = self.encoder.encode(input_sequence_w)
        last_layer = self.layers[self.l-1]
        current_length = self.chunk_length
        #for layer_level in range(self.l):
        #    current_length = int(current_length/self.h)
        #    input_sequence = input_sequence.reshape((current_length, self.h, self.d))
        #    output_symbols = output_symbols[[(x+1)*self.h-1 for x in range(0,current_length)]]
        #    input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].forget(input_sequence, output_symbols, completely=completely)
        #    last_layer.directly_forget(seq_encoding_for_the_last_layer)
        for layer_level in range(self.l):
            if self.h ** (layer_level + 1) < current_length + 1:
                output_symbols = torch.stack([output_symbols[i - self.h ** ((layer_level + 1) - 1)] for i in range(self.h ** (layer_level + 1), current_length + 1)])
                input_sequence = torch.stack([torch.stack([input_sequence[j] for j in range(i - self.h ** (layer_level + 1), i, self.h ** ((layer_level + 1) - 1))]) for i in
                         range(self.h ** (layer_level + 1), current_length + 1)])
                if debugging:
                    retreived_output_symbol_vector, max = self.encoder.decode_multi(output_symbols)
                    print(retreived_output_symbol_vector)
                input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].memorize(input_sequence, output_symbols, is_last=(last_layer == self.layers[layer_level]))
                last_layer.directly_forget(seq_encoding_for_the_last_layer)
            else:
                break

    def forget_text(self,tokenized_text, completely=False):
        #for i in range(0,self.h):
        self.forget(tokenized_text,completely=completely)

    def retrieve(self,input_sequence_w):
        input_sequence, _ = self.encoder.encode(input_sequence_w, only_input=True)
        last_layer = self.layers[self.l-1]
        encoding_for_the_last_layer = torch.zeros(1,self.d)
        current_length = min(self.chunk_length,self.max_len)
        if len(input_sequence) > current_length:
            input_sequence = input_sequence[len(input_sequence)-current_length:len(input_sequence)]
        for layer_level in range(self.l):
            current_length = int(current_length/self.h)
            input_sequence = input_sequence.reshape((current_length, self.h, self.d))
            input_sequence, seq_encoding_for_the_last_layer,locally_predicted = self.layers[layer_level].retrieve(input_sequence)
            encoding_for_the_last_layer = encoding_for_the_last_layer + seq_encoding_for_the_last_layer

            #### DEBUGGING
            if debugging:
                retreived_output_symbol_vector_APPO, max_APPO = self.encoder.decode(last_layer.directly_retrieve(seq_encoding_for_the_last_layer))
                print(f"OUT APPO: {retreived_output_symbol_vector_APPO} level: {layer_level}")
                retreived_output_symbol_vector_APPO, max_APPO = self.encoder.decode(locally_predicted)
                print(f"LOCALLY APPO: {retreived_output_symbol_vector_APPO} level: {layer_level}")


            if verbose:
                retreived_output_symbol_vector, max = self.encoder.decode(last_layer.directly_retrieve(seq_encoding_for_the_last_layer))
                print(f"NORM OF THE VECTOR:", torch.linalg.norm(seq_encoding_for_the_last_layer))
                print((retreived_output_symbol_vector, max))

        retreived_output_symbol_vector, max = self.encoder.decode(last_layer.directly_retrieve(encoding_for_the_last_layer))
        return max, retreived_output_symbol_vector


class Evaluation:
    def check_memorization(self, memo, text, starting_point=1):
        basic_block = memo.h ** memo.l
        count = 0
        correct = 0
        for i in range(starting_point, len(text)):
            IN = text[max(0,i - basic_block):i]
            if debugging: print(IN)
            max_value, out = memo.retrieve(text[max(0,i - basic_block):i])
            if debugging: print(">" + out + " : " + text[i])
            count += 1
            if out == text[i]:
                correct += 1
        return (correct / count)


#main()

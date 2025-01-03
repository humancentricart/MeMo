import torch
import math
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

verbose = False


def main():
    # the first attempt
    d,h,l = 1024, 2, 3
    with open("testo_di_prova.txt") as my_first_text_f:
        my_first_text_tokenized = my_first_text_f.read().split()

    memo = MeMo(d,h,l,list(set(my_first_text_tokenized)))

    memo.memorize_text(my_first_text_tokenized)
    #memo.memorize(my_first_text_tokenized[9:17],my_first_text_tokenized[17])
    #memo.memorize(my_first_text_tokenized[11:19],my_first_text_tokenized[19])

    print(my_first_text_tokenized[9:20])

    max_value , out = memo.retrieve(my_first_text_tokenized[9:17])
    print("Emitted  : " + out + " " + str(max_value))
    print("Expected : " + my_first_text_tokenized[17])
    seq2 = ['27', '27', '27', '27', '27', '27', '27', '27']
    print(seq2)
    max_value, out = memo.retrieve(seq2)
    print("Emitted  : " + out + " " + str(max_value))
    print("Expected : " + my_first_text_tokenized[17])
    # EVALUATE
    e = Evaluation()
    out = e.check_memorization(memo,my_first_text_tokenized)
    print("Degree of memorization: %f ", out)
    memo.forget_text(my_first_text_tokenized,completely=False)
    out = e.check_memorization(memo,my_first_text_tokenized)
    print("Degree of memorization: %f ", out)

class MeMoEncoder:
    def __init__(self, inner_dim, dictionary=None, max_lenght = 10):
        self.d = inner_dim
        # Encoding vectors for tokens - the first row is a zero row for padding
        self.dict_enc = torch.normal(0, 1 / math.sqrt(self.d), size=(len(dictionary) + 1, self.d))
        self.dict_enc[0] = torch.zeros(self.d)
        self.dictionary = dictionary
        self.max_len = max_lenght

    def encode(self,sequence, next_token=None):
        input_ = self.dict_enc[[0 for _ in range(0,self.max_len - len(sequence))]+ [self.dictionary.index(x)+1 for x in sequence]]
        output_ = None
        if next_token != None:
            output_ = self.dict_enc[self.dictionary.index(next_token)+1]
        return input_,output_

    def decode(self,vector):
        out = torch.matmul(self.dict_enc,torch.transpose(vector,0,1))
        return self.dictionary[torch.argmax(out) - 1], max(out)



class MeMoLayer:
    def __init__(self,inner_dim, num_of_heads):
        self.d = inner_dim
        self.h = num_of_heads
        self.d_k = self.d // self.h
        if self.d / self.h != self.d_k:
            print("exception")
        # W_v for a single head - at the very end, there should be num_of_heads W_v_single_head matrices or a W_v inner_dim x inner_dim
        self.W_v_single_head = torch.normal(0, 1/math.sqrt(self.d_k), size=(self.d,self.d_k))

        # Prj : projection matrix that determines the representation of a sequence
        self.Prj = torch.normal(0, 1/math.sqrt(self.d*self.h), size=(self.d,self.d*self.h))
        self.Prj = torch.transpose(self.Prj, 0, 1)
        # CMM : correlation matrix memory for the specific layer
        self.CMM = torch.zeros(size=(self.d,self.d), requires_grad=False)

    # The most simple implementation
    # Input sequence is has a shape of (self.h,self.d), that is self.h sequences are proposed as input rows
    def memorize(self, input_sequence,output_symbol, symbolic = False, symbols = None, is_last = False):
        (blocks,_,_) = input_sequence.shape
        sequence_encoding = torch.matmul(input_sequence.reshape((blocks,self.d * self.h)),self.Prj)
        seq_enc_per_token = torch.matmul(input_sequence[-1],self.W_v_single_head).reshape((self.d,1)) / math.sqrt(self.h)
        if verbose:
            norm1 = torch.linalg.norm(input_sequence.reshape((blocks,self.d * self.h)), dim=1)
            norm2 = torch.linalg.norm(sequence_encoding, dim=1)
            norm3 = torch.linalg.norm(seq_enc_per_token, dim=0).item()
            norm4 = torch.linalg.norm(output_symbol, dim=0).item()
            if norm3 > 1.5:
                print("careful")
        # Updating local CMM
        if not is_last:
            #if verbose:
            checking1 = torch.linalg.norm(torch.matmul(torch.transpose(seq_enc_per_token,0,1), self.CMM), dim=1).item()
            if checking1 < 0.5:
                #print("careful")
                self.CMM = self.CMM + torch.matmul(seq_enc_per_token,sequence_encoding[-1].reshape(1,self.d))
            if verbose:
                checking = torch.linalg.norm(torch.matmul(torch.transpose(seq_enc_per_token,0,1), self.CMM), dim=1).item()
                if checking > 1.5:
                    print("careful")
        seq_enc_plus_out = torch.matmul(seq_enc_per_token,output_symbol.reshape(1,self.d))
        return sequence_encoding, seq_enc_plus_out
    def directly_memorize(self, input_sequence):
        self.CMM = self.CMM + input_sequence

    def forget(self, input_sequence,output_symbol, completely=True):
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
        seq_enc_plus_out = torch.matmul(seq_enc_per_token,output_symbol.reshape(1,self.d))
        return sequence_encoding, seq_enc_plus_out

    def directly_forget(self, input_sequence):
        self.CMM = self.CMM - input_sequence

    def retrieve(self,input_sequence):
        (blocks,_,_) = input_sequence.shape
        seq_enc_per_token = torch.matmul(input_sequence,self.W_v_single_head).reshape((blocks,self.d)) / math.sqrt(self.h)
        retrieved_sequence_encoding = torch.matmul(seq_enc_per_token,self.CMM)
        if verbose:
            sequence_encoding = torch.matmul(input_sequence.reshape((blocks, self.d * self.h)), self.Prj)
            out = torch.matmul(sequence_encoding,torch.transpose(retrieved_sequence_encoding,0,1))
            out1 = torch.linalg.norm(seq_enc_per_token,dim=1)
            if torch.max(out1).item() > 1.5 or torch.max(out).item()> 1.5:
                print("Errore")
        return retrieved_sequence_encoding, seq_enc_per_token[-1].reshape(1,self.d)

    def directly_retrieve(self,vector):
        return torch.matmul(vector,self.CMM)

class MeMo:
    def __init__(self, inner_dim, num_of_heads, num_of_layers,token_set_in_list):
        self.d = inner_dim
        self.h = num_of_heads
        self.l = num_of_layers
        self.max_len = self.h**self.l
        self.layers = [MeMoLayer(inner_dim,num_of_heads) for _ in range(num_of_layers)]
        self.encoder = MeMoEncoder(inner_dim,dictionary=token_set_in_list, max_lenght=self.max_len)


    # The most simple implementation
    # Input sequence has a shape of (self.h**self.l,self.d), that is self.h sequences are proposed as input rows
    def memorize(self,input_sequence_w,output_symbol_w):
        input_sequence, output_symbol = self.encoder.encode(input_sequence_w,output_symbol_w)
        last_layer = self.layers[self.l-1]
        for layer_level in range(self.l):
            input_sequence = input_sequence.reshape((self.h ** (self.l - 1 - layer_level), self.h, self.d))
            input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].memorize(input_sequence, output_symbol, is_last=(last_layer == self.layers[layer_level]))
            last_layer.directly_memorize(seq_encoding_for_the_last_layer)

    def memorize_text(self,tokenized_text):
        for i in range(1,len(tokenized_text)):
            input_ = tokenized_text[max(0,i-self.max_len):i]
            output_ = tokenized_text[i]
            self.memorize(input_,output_)


    def forget(self,input_sequence_w,output_symbol_w, completely=True):
        input_sequence, output_symbol = self.encoder.encode(input_sequence_w,output_symbol_w)
        last_layer = self.layers[self.l-1]
        for layer_level in range(self.l):
            input_sequence = input_sequence.reshape((self.h ** (self.l - 1 - layer_level), self.h, self.d))
            input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].forget(input_sequence, output_symbol, completely=completely)
            last_layer.directly_forget(seq_encoding_for_the_last_layer)

    def forget_text(self,tokenized_text, completely=True):
        for i in range(1,len(tokenized_text)):
            input_ = tokenized_text[max(0,i-self.max_len):i]
            output_ = tokenized_text[i]
            self.forget(input_,output_,completely=completely)

    def retrieve(self,input_sequence_w):
        input_sequence, _ = self.encoder.encode(input_sequence_w)
        last_layer = self.layers[self.l-1]
        encoding_for_the_last_layer = torch.zeros(1,self.d)
        for layer_level in range(self.l):
            input_sequence = input_sequence.reshape((self.h ** (self.l - 1 - layer_level), self.h, self.d))
            input_sequence, seq_encoding_for_the_last_layer = self.layers[layer_level].retrieve(input_sequence)
            encoding_for_the_last_layer += seq_encoding_for_the_last_layer
            if verbose:
                retreived_output_symbol_vector, max = self.encoder.decode(last_layer.directly_retrieve(seq_encoding_for_the_last_layer))
                print(f"NORM OF THE VECTOR:", torch.linalg.norm(seq_encoding_for_the_last_layer))
                print((retreived_output_symbol_vector, max))

        retreived_output_symbol_vector, max = self.encoder.decode(last_layer.directly_retrieve(encoding_for_the_last_layer))
        return max, retreived_output_symbol_vector


class Evaluation:
    def check_memorization(self, memo, text, starting_point=None):
        if starting_point == None:
            basic_block = memo.h ** memo.l
        else:
            basic_block = starting_point
        count = 0
        correct = 0
        for i in range(basic_block, len(text) - 1):
            max_value, out = memo.retrieve(text[i - basic_block:i])
            count += 1
            if out == text[i]:
                correct += 1
        return (correct / count)


#main()

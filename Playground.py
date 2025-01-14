import numpy as np
'''
The puropose of this file is to study how sequences can be stored at different layers
'''
def gen_input_output_for_next_layer(plain_input, heads, sequenc_len, level):
    input = []
    for i in range(heads,sequenc_len+1):
        input.append(plain_input[i-heads:i])
    return input

def gen_input_output_for_next_layer_2(plain_input, prev_outputs, heads, sequenc_len, level):
    input = []
    output = []
    for i in range(heads**level,sequenc_len):
        elem = []
        for j in range(i-heads**level,i,heads**(level-1)):
            elem.append(plain_input[j])
        output.append(prev_outputs[j])
        if j != i - heads**(level-1):
            print(f'DIFF {i-1} - {j}')
        #elem = sum(elem,[])
        input.append(elem)
    return input,output

def gen_input_output_for_next_layer_3(plain_input, prev_outputs, heads, sequenc_len, level):
    input = []
    output = []
    for i in range(heads**level,sequenc_len):
        elem = [plain_input[j] for j in range(i-heads**level,i,heads**(level-1))]
        output.append(prev_outputs[i - heads**(level-1)])
        #elem = sum(elem,[])
        input.append(elem)
    return input,output


def gen_input_output_for_next_layer_4(plain_input, prev_outputs, heads, sequenc_len, layer):
    #sequenc_len = len(plain_input)
    #plain_input = heads**(layer-1)*["*" for _ in range(heads-1)] + plain_input
    #prev_outputs = heads**(layer-1)*["*" for _ in range(heads-1)] + prev_outputs
    #sequenc_len = sequenc_len + (layer-1)*heads + (heads-1)
    #sequenc_len = len(plain_input)
    print(f'{len(plain_input)} : {plain_input}')
    print(f'{len(prev_outputs)} : {prev_outputs}')
    input = [[plain_input[j] for j in range(i-heads**layer,i,heads**(layer-1))] for i in range(heads**layer,sequenc_len)]
    APPO = heads**layer
    output = [prev_outputs[i - heads**(layer-1)] for i in range(heads**layer,sequenc_len)]
    return input,output



input = [i+1 for i in range(0,17)]

heads = 2
layers = 3

#input = ["Z" for _ in range(heads**layers-1)] + input
seq_len = len(input)
#input = replicate_for_heads(input)
print(input)
output = input[1:seq_len]
for l in range(layers):
    input, output = gen_input_output_for_next_layer_4(input, output, heads, seq_len, layer=l+1)
    #print(f'INPUT :{input}')
    #print(f'OUTPUT:{output}')
    print(f'{len(input)} ---- {[{"IN":i, "OUT":o} for i,o in zip(input,output)]}')

#import torch

#l = [torch.tensor([1,4]),torch.tensor([2,9])]
#o = torch.concat(l,1)

#print(o)

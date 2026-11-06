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
    print(f'Plain input    : {len(plain_input)} : {plain_input}')
    print(f'Previous output: {len(prev_outputs)} : {prev_outputs}')
    input = [[plain_input[j] for j in range(i-heads**layer,i,heads**(layer-1))] for i in range(heads**layer,sequenc_len)]
    APPO = heads**layer
    output = ["no" for i in range(0,heads**layer) ] + [prev_outputs[i - heads**(layer-1)] for i in range(heads**layer,sequenc_len)]
    return input,output


def gen_input_output_for_next_layer_5(plain_input, prev_outputs, heads, sequenc_len, layer):
    plain_input = ["no" for x in range(0,(heads-1)*(heads**(layer-1)))] + plain_input
    prev_outputs = prev_outputs
    final = sequenc_len+(heads-1)*heads**(layer-1)
    print(f"Plain input    : {len(plain_input)}  - Prev : {len(prev_outputs)}  - Final : {final}")
#    input = [[plain_input[j] for j in range(i-heads**layer,i,heads**(layer-1))] for i in range(heads**layer,final)]
    input = [[plain_input[j*heads**(layer-1) + i] for j in range(0,heads)] for i in range(len(prev_outputs))]
    output =  [prev_outputs[i] for i in range(sequenc_len-1)]

    return input,output



input = [i+1 for i in range(0,65)]

heads = 4
layers = 5

#input = ["Z" for _ in range(heads**layers-1)] + input
seq_len = len(input)
#input = replicate_for_heads(input)
print(input)
output = input[1:seq_len]
for l in range(layers):
    input, output = gen_input_output_for_next_layer_5(input, output, heads, seq_len, layer=l+1)
    print(f'{len(input)} \t---- {[{"IN":i, "OUT":o} for i,o in zip(input,output)]}')

import tqdm
import torch

class Evaluation:
    def check_memorization(self, model, tokenizer, text, # device='cpu',
                           starting_point=None):
        if starting_point == None:
            basic_block = model.memo.h ** model.memo.l
        else:
            basic_block = starting_point
          
        
        input_ = tokenizer(my_first_text, padding='longest', truncation='do_not_truncate', max_length=None)
        input_ = tokenizer.pad(input_, pad_to_multiple_of=basic_block)
        input_ids = input_['input_ids']
                
        count = 0
        correct = 0
        max_length = tokenizer.max_length
        (batch_size, number_of_tokens) = input_ids.shape

        #print(f"(batch_size, number_of_tokens) = {(batch_size, number_of_tokens)}")
        
        for i in tqdm.tqdm(range(basic_block,  number_of_tokens - 1)):
            text_tokens = input_ids[:, i - basic_block:i]
            
            (batch_size, number_of_tokens) = text_tokens.shape
            
            text_tokens = torch.concat((torch.zeros((batch_size, max_length-1-number_of_tokens), 
                                                    dtype=torch.int), 
                                        text_tokens), axis=1
                                      )
            
            #print(i - basic_block, i)
            out, max_value = model.greedy_retrieve(text_tokens)
            #print(out, input_ids[:, i])
            #print(out[0].item())
            
            count += batch_size
            correct += torch.sum(out.to('cpu') == input_ids[:, i])
        
                           
        return correct / count

    def check_pretokenized(self, model, tokenizer, input_ids,# device='cpu',
                           starting_point=None):
        basic_block = model.memo.h ** model.memo.l
        if starting_point == None:
            starting_point = basic_block
                
        count = 0
        correct = 0
        max_length = tokenizer.max_length
        (batch_size, number_of_tokens) = input_ids.shape

        #print(f"(batch_size, number_of_tokens) = {(batch_size, number_of_tokens)}")
        
        for i in tqdm.tqdm(range(starting_point,  number_of_tokens - 1)):
            text_tokens = input_ids[:, max(0,i - basic_block):i]
            
            (batch_size, number_of_tokens) = text_tokens.shape
            
            text_tokens = torch.concat((torch.zeros((batch_size, max_length-1-number_of_tokens), 
                                                    dtype=torch.int), 
                                        text_tokens), axis=1
                                      )
            
            #print(i - basic_block, i)
            out, max_value = model.greedy_retrieve(text_tokens)
            #print(out, input_ids[:, i])
            #print(out[0].item())
            
            #OLD 
            #count += batch_size
            #correct += torch.sum(out.to('cpu') == input_ids[:, i])

            #count += batch_size
            #app1 = out.to('cpu')
            #app2 = input_ids[:, i]
            #app11 = tokenizer.decode(app1)
            #app21 = tokenizer.decode(app2)
            appo3 = out.to('cpu') == input_ids[:, i]
            appo4 = input_ids[:, i] != torch.tensor([0]) # To change with the padding token
            appo5 = appo3 & appo4
            count += torch.sum(appo4)
            #correct += torch.sum(out.to('cpu') == input_ids[:, i])
            correct += torch.sum(appo5)

                           
        return correct / count
        
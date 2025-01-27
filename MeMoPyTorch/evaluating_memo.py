import tqdm
import torch


class Evaluation:
    def check_memorization(self, model, tokenizer, text, # device='cpu',
                           starting_point=None):
        if starting_point == None:
            basic_block = model.h ** model.l
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
            out, max_value = model.retrieve(text_tokens)
            #print(out, input_ids[:, i])
            #print(out[0].item())
            
            count += batch_size
            correct += torch.sum(out.to('cpu') == input_ids[:, i])
        
                           
        return correct / count

    def check_pretokenized(self, model, tokenizer, input_ids,# device='cpu',
                           starting_point=None):

        basic_block = model.h ** model.l
        

        if starting_point == None:
            starting_point = basic_block
        print(f"Starting point : {starting_point}"  )
                
        count = 0
        correct = 0
        max_length = tokenizer.max_length
        (batch_size, number_of_tokens) = input_ids.shape

        #print(f"(batch_size, number_of_tokens) = {(batch_size, number_of_tokens)}")
        
        #for i in tqdm.tqdm(range(basic_block,  number_of_tokens - 1)):
        #    text_tokens = input_ids[:, i - basic_block:i]
        for i in tqdm.tqdm(range(starting_point,  number_of_tokens - 1)):
            text_tokens = input_ids[:, max(0,i - basic_block):i]
            
            (batch_size, number_of_tokens) = text_tokens.shape
            
            text_tokens = torch.concat((torch.zeros((batch_size, max(0,basic_block-i)), 
                                                    dtype=torch.int), 
                                        text_tokens), axis=1
                                      )
            #print(text_tokens.shape)
            #print(i - basic_block, i)
            out, max_value = model.retrieve(text_tokens)
            #ùprint(out, input_ids[:, i])
            #print(out[0].item())


            appo3 = out.to('cpu') == input_ids[:, i]
            appo4 = input_ids[:, i] != torch.tensor([tokenizer.pad_token_id]) # To change with the padding token
            appo5 = appo3 & appo4
            count += torch.sum(appo4)
            #correct += torch.sum(out.to('cpu') == input_ids[:, i])
            correct += torch.sum(appo5)

            #count += batch_size
            #correct += torch.sum(out.to('cpu') == input_ids[:, i])
        
                           
        return correct / count
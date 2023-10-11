from tqdm import tqdm

import torch

from config import MTS


    
def tokenize_ec_title(EC_title_pairs_total):
    input_ids_ec = []; attention_mask_ec = []
    input_ids_title = []; attention_mask_title = []
    labels = []
    for (ec, title), label in tqdm(EC_title_pairs_total):
        encoded_ec = MTS.tokenizer.encode_plus(
                                ec,
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = MTS.MAXLEN,
                                truncation=True,
                                padding='max_length',
                                pad_to_max_length = True,
                                return_attention_mask = True,   
                                return_tensors = 'pt',
                            )
        encoded_title = MTS.tokenizer.encode_plus(
                                title,
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = MTS.MAXLEN,  
                                truncation=True,     
                                padding='max_length',
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'pt',
                            )
        
        input_ids_ec.append(encoded_ec['input_ids'])
        attention_mask_ec.append(encoded_ec['attention_mask'])
        input_ids_title.append(encoded_title['input_ids'])
        attention_mask_title.append(encoded_title['attention_mask'])
        labels.append(label)

    # Convert the lists into tensors.
    input_ids_ec = torch.cat(input_ids_ec, dim=0)
    input_ids_title = torch.cat(input_ids_title, dim=0)
    attention_mask_ec = torch.cat(attention_mask_ec, dim=0)
    attention_mask_title = torch.cat(attention_mask_title, dim=0)
    labels = torch.tensor(labels)
    
    return input_ids_ec, input_ids_title, attention_mask_ec, attention_mask_title, labels


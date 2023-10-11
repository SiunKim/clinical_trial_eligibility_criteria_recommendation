import torch

from torch.utils.data import TensorDataset, random_split

from config import MTS


def set_datasets(dataset, tokenizer):
    # Tokenize all of the texts and map the tokens to thier word IDs
    from tqdm import tqdm
    input_ids = []; attention_masks = []
    for data_i in tqdm(dataset):
        encoded_EC_title = tokenizer.encode_plus(
                                data_i['text'], # to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = MTS.MAXLEN, # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                            )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_EC_title['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_EC_title['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor([d['label'] for d in dataset])

    dataset = TensorDataset(input_ids, attention_masks, labels)
    test_size = int(MTS.TEST_RATIO*len(dataset))
    val_size = int(MTS.TEST_RATIO*len(dataset))
    train_size = len(dataset) - test_size- val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    print('{:>5,} test samples'.format(test_size))
    
    return train_dataset, val_dataset, test_dataset

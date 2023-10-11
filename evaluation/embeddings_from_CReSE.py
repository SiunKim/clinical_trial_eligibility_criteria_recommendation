import pandas as pd

import torch

import sys; sys.path.append("C:CReSE/CReSE")
from model import CLIPModel, EcEncoder, ProjectionHead



class CReSEDatasetECs(torch.utils.data.Dataset):
    #CLIP-Dataset class for saving only ecs to get ec embeddings
    def __init__(self, ecs, tokenizer, max_len=256):
        self.ecs = list(ecs)
        self.encoded_ecs1 = tokenizer(list(ecs), 
                                      padding=True, 
                                      truncation=True, 
                                      max_length=max_len)
        
    def __getitem__(self, idx):
        #saving a tokenized ec in item dictionary before returning the individual
        item = {key + '_ecs': torch.tensor(values[idx]) 
                    for key, values in self.encoded_ecs1.items()}
        item['ecs'] = self.ecs[idx]
        
        return item

    def __len__(self):
        return len(self.ecs)



def get_ec_embeddings_from_CReSE(CLIP_model, 
                                tokenizer, 
                                ecs,
                                batch_size=64):
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #set dataloader for inference (ec-embeddings)
    dataset = CReSEDatasetECs(ecs, tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False)

    #get ec embddings from ec_encoder of CLIP_model
    ecs_embeddings_total = []
    for batch in dataloader:
        #batch to device
        batch["input_ids_ecs"] = batch["input_ids_ecs"].to(device)
        batch["attention_mask_ecs"] = batch["attention_mask_ecs"].to(device)
        #inference ec_encoder
        ecs_features = CLIP_model.ec_encoder(input_ids=batch["input_ids_ecs"], 
                                             attention_mask=batch["attention_mask_ecs"])
        #get through projection layer!
        ecs_embeddings = CLIP_model.ec_projection(ecs_features)
        #save ecs_embeddings in ecs_embeddings_total
        ecs_embeddings_total += ecs_embeddings.tolist()

    #set dataframe 
    assert len(ecs)==len(ecs_embeddings_total), "The lengths of ecs and ecs_embeddings_total must be same!"
    data = pd.DataFrame({'ecs': ecs, 'emb': ecs_embeddings_total})
    
    return data, ecs_embeddings_total
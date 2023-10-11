import time

from tqdm import tqdm 

from collections import Counter

import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer
from fast_pytorch_kmeans import KMeans as KMeans_torch

import sys; sys.path.append('G:/내 드라이브/[1] CCADD N CBDL/[1] Personal Research/2022_MSR_drug_repositioning/[2] Code/EC_title_recommendation')
from CLIP_EC_EC import CLIPModel, EcEncoder, ProjectionHead



class CLIPDataset_ecs(torch.utils.data.Dataset):
    #CLIP-Dataset class for saving only ecs to get ec embeddings
    def __init__(self, ecs, tokenizer, max_len=256):
        self.ecs = list(ecs)
        self.encoded_ecs1 = tokenizer(list(ecs), 
                                      padding=True, 
                                      truncation=True, 
                                      max_length=max_len)
        
    def __getitem__(self, idx):
        item['ecs'] = self.ecs[idx]
        #saving a tokenized ec in item dictionary before returning the individual
        item = {key + '_ecs': torch.tensor(values[idx]) 
                    for key, values in self.encoded_ecs1.items()}
        return item

    def __len__(self):
        return len(self.ecs)


def get_ec_embeddings_from_clip(CLIP_model, 
                                tokenizer, 
                                ecs,
                                device,
                                batch_size=64):
    #set dataloader for inference (ec-embeddings)
    dataset = CLIPDataset_ecs(ecs, tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False)

    #get ec embddings from ec_encoder of CLIP_model
    ecs_embeddings_total = []
    for batch in tqdm(dataloader):
        #batch to device
        batch["input_ids_ecs"] = batch["input_ids_ecs"].to(device)
        batch["attention_mask_ecs"] = batch["attention_mask_ecs"].to(device)
        #inference ec_encoder
        ecs_features = CLIP_model.ec_encoder(input_ids=batch["input_ids_ecs"], 
                                             attention_mask=batch["attention_mask_ecs"])
        ecs_embeddings = CLIP_model.ec_projection(ecs_features)
        #save ecs_embeddings in ecs_embeddings_total
        ecs_embeddings_total += ecs_embeddings.tolist()

    #set dataframe 
    assert len(ecs)==len(ecs_embeddings_total), "The lengths of ecs and ecs_embeddings_total must be same!"
    data = pd.DataFrame({'ecs': ecs, 'emb': ecs_embeddings_total})
    
    return data, ecs_embeddings_total
    
    
def clustering_ecs_by_embedding(data, num_clusters, device, print_out=False):
    #clustering ecs using KMeans_torch
    #set ec_embedding as torch
    if type(data['emb'])!=torch.Tensor:
        X = torch.tensor(data['emb']).to(device)
    else:
        X = torch.tensor(data['emb'].tolist()).to(device)
            
    #perform KMeans clustering
    stime = time.time()
    print(f'Start KMeasn clustering for {len(X)} ecs - NUM_CLUSTERS={num_clusters}')
    kmeans = KMeans_torch(n_clusters=num_clusters, 
                          mode='cosine', 
                          verbose=1)
    assigned_clusters = kmeans.fit_predict(X)
    print(f'Finish KMeans clustering - elapsed time: {(time.time()-stime):.1f} seconds')

    if print_out:
        print_clustering_results(data)
    
    return data, assigned_clusters, kmeans
    
    
def print_clustering_results(data):
    #clutser distribution
    clusters = data['cluster']
    print(f"Cluster distribution: {Counter(clusters).most_common(len(set(clusters)))}")
    #print by cluster
    for c in set(clusters):
        print(f'Cluster: {c}')
        print(list(data[clusters==c]['ecs']))


def set_CLIP_model_and_tokenizer(best_model_name):
    #import best model and set tokenizer
    dir_ec_ec_best = "C:/Users/Admin/Desktop/ec_ec_best_models"
    CLIP_model = torch.load(f"{dir_ec_ec_best}/{best_model_name}")
    tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
    
    return CLIP_model, tokenizer
    
    

import time

from tqdm import tqdm 

from collections import Counter

import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer
from fast_pytorch_kmeans import KMeans as KMeans_torch

import sys; sys.path.append('C:/CReSE/CReSE')
from model import CReSE, EcEncoder, ProjectionHead
from embeddings_from_CReSE import CReSEDatasetECs, get_ec_embeddings_from_CReSE

    
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
    

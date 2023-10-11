import os
import random
import pickle
from tqdm import tqdm

import torch


import sys; sys.path.append("C:CReSE/CReSE")
from embeddings_from_CReSE import get_ec_embeddings_from_clip
from train import set_CReSE_and_tokenizer
from model import CLIPModel, EcEncoder, ProjectionHead
from config import CFG


class CAS(): #clustering analysis settings
    n_sample = 100000
    embedding_group_n = 128 #batch size for get ec_embeddings
    num_cluster = 100
    embedding_model_name = \
        "best_model_projdim512_michiyasunaga_BioLinkBERT-base_bs16_elr5e-06_hlr0.0005_wd0.0001_nprompt4_nsample30K_list.pt"


#set customebedder for bertopic
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer
class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model_clip, tokenizer):
        super().__init__()
        self.embedding_model = embedding_model_clip
        self.tokenizer = tokenizer

    def embed(self, documents, verbose=False):
        _, embeddings = get_ec_embeddings_from_clip(self.embedding_model, 
                                                    self.tokenizer, 
                                                    documents)
        embeddings = torch.tensor(embeddings)
        
        return embeddings 

    
    
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_ec_embeddings_by_bert(model, tokenizer, ecs, device):
    ec_embeddings_total = torch.Tensor()
    ec_embeddings_total = ec_embeddings_total.to(device)
    
    for i in range(int(len(ecs)/CAS.embedding_group_n)+1):
        #Tokenize ecs
        encoded_input = tokenizer(ecs[i*CAS.embedding_group_n:(i+1)*CAS.embedding_group_n], 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=CAS.embedding_group_n, 
                                  return_tensors='pt')
        encoded_input.to(device)
        
        #Compute token embeddings
        with torch.no_grad():
            encoded_input.to(device)
            model_output = model(**encoded_input)
        
        ec_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        ec_embeddings_total = torch.cat((ec_embeddings_total, ec_embeddings), dim=0)
        
    return ec_embeddings_total


def resampling_ecs(ecs_total):
    ecs = random.sample(ecs_total, CAS.n_sample)
    random.shuffle(ecs)
    
    return ecs
   
      
def main():
    #import ECs
    dir_ecs = "F:/Datasets_230315/MSRA/CTgov/tdt_230414"
    tdt_noncommon = "train_data_tuples_train_total_noncommon_EC1509495_CT153361_230509"
    with open(f"{dir_ecs}/{tdt_noncommon}.p", "rb") as f:
        tdt_train = pickle.load(f)
    ecs_total = list(set([tdt[0] for tdt in tdt_train]))

    #set random_seed
    for random_seed in tqdm(range(20)):
        random.seed(random_seed)
        #check whether bertopic for the random_seed is already saved
        fname_bertopic = f"bertopic_model_cluster{CAS.num_cluster}_seed{random_seed}_0629"
        if fname_bertopic in os.listdir('bertopic_save'):
            continue

        #set bertopic
        model, tokenizer = set_CReSE_and_tokenizer(CAS.embedding_model_name)
        custom_embedder = CustomEmbedder(embedding_model_clip=model,
                                        tokenizer=tokenizer)
        topic_model = BERTopic(embedding_model=custom_embedder)
        #resampling ecs
        ecs = resampling_ecs(ecs_total=ecs_total)
        #fit-transform bertopic
        topic_model.fit_transform(ecs)
        topic_model.reduce_topics(ecs, nr_topics=CAS.num_cluster)
        #reduce topics and outliers
        topics = topic_model.topics_
        labels = topic_model.reduce_outliers(ecs, topics)

        #save bertopic model
        topic_model.save(f"bertopic_save/{fname_bertopic}", save_embedding_model=True)
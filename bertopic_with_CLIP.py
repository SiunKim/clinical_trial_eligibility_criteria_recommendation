import os

import random; random.seed(44)
import pickle

import torch

#import clip_embedding
dir_clipembedding = "G:/내 드라이브/[1] CCADD N CBDL/[1] Personal Research/2022_MSR_drug_repositioning/[2] Code/EC_title_recommendation/clustering_EC"
import sys; sys.path.append(dir_clipembedding)
from clip_embedding import set_CLIP_model_and_tokenizer
from clip_embedding import get_ec_embeddings_from_clip
#need to set CustomEmbedder object
from CLIP_EC_EC import CLIPModel, EcEncoder, ProjectionHead

#set CustomEmbedder for making CLIP-based bertopic objects
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
class CustomEmbedder(BaseEmbedder):
    def __init__(self, clip_model, tokenizer):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer

    def embed(self, documents, verbose=False):
        #get ec_embeddings through ec_encoder and projection layer
        _, embeddings = get_ec_embeddings_from_clip(self.clip_model, 
                                                    self.tokenizer, 
                                                    documents)
        return embeddings 



embedding_model_name = \
    "best_model_projdim512_michiyasunaga_BioLinkBERT-base_bs16_elr5e-06_hlr0.0005_wd0.0001_nprompt4_nsample30K_list.pt"
CLIP_model, tokenizer = set_CLIP_model_and_tokenizer(embedding_model_name)
custom_embedder = CustomEmbedder(clip_model=CLIP_model,
                                 tokenizer=tokenizer)
topic_model = BERTopic(embedding_model=custom_embedder)
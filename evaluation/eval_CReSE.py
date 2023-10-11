import random
import pickle
from tqdm import tqdm

from collections import defaultdict

import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from transformers import AutoTokenizer, AutoModel
import torch

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

#for setting bertopic from CReSE
import sys; sys.path.append("C:CReSE/CReSE"); sys.path.append("C:/CReSE/evaluations")
from train import set_CReSE_and_tokenizer
from model import CReSE, EcEncoder, ProjectionHead
from config import CFG
from embeddings_from_CReSE import get_ec_embeddings_from_CReSE
from clustering_with_CReSE import clustering_ecs_by_embedding
#set customebedder for bertopic
from bertopic.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer
class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model_clip, tokenizer):
        super().__init__()
        self.embedding_model = embedding_model_clip
        self.tokenizer = tokenizer

    def embed(self, documents, verbose=False):
        _, embeddings = get_ec_embeddings_from_CReSE(self.embedding_model, 
                                                               self.tokenizer, 
                                                               documents)
        return embeddings 

from utils_eval_CReSE import evaluate_clustering, resampling_ecs
from utils_eval_CReSE import mean_pooling, min_max_scaling, print_out_95CI_corr



class CAS(): #clsutering analysis settings
    repeated_n = 20 #number of repeated experiments for calculating 95% CI of performances
    print_out = False 
    embedding_group_n = 128 
    num_clusters = [100, 200, 300]
    embedding_model_name = "best_model_projdim512_michiyasunaga_BioLinkBERT-base_bs16_elr5e-06_hlr0.0005_wd0.0001_nprompt4_nsample30K_list.pt"



def clustering_TDIDF(ecs, num_clusters):
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(ecs)
    
    # # Apply K-means clustering -- original
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    labels = kmeans.labels_
    
    return labels
    

def get_ec_embeddings_by_bert(model, tokenizer, ecs, device):
    ec_embeddings_total = torch.Tensor()
    ec_embeddings_total = ec_embeddings_total.to(device)
    
    for i in tqdm(range(int(len(ecs)/CAS.embedding_group_n)+1)):
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


def update_crs_dict_list(crs_dict_list, method, labling_func):
    for _ in tqdm(range(CAS.repeated_n)):
        ecs = resampling_ecs(ecs_in_EC_pairs_cr, ecs_total=ecs_total)
        for num_cluster in CAS.num_clusters:
            #get labels for ecs
            labels = labling_func(ecs, num_cluster) 
            #calculate correlation measures
            _, _, corr_pearson, corr_spearman = evaluate_clustering(EC_pairs_cr, method, ecs, labels, num_cluster)
            crs_dict_list[method][num_cluster]['corr_pearson'].append(corr_pearson)
            crs_dict_list[method][num_cluster]['corr_spearman'].append(corr_spearman)
            
    return crs_dict_list
            
            
def update_crs_dict_list_bertopic(crs_dict_list, method, sentence_transformer):
    #revesre num_clustering for reducing topic numbers
    num_clusters = CAS.num_clusters[::-1]
    #repeated...
    for _ in tqdm(range(CAS.repeated_n)):
        #set bertopic
        if sentence_transformer:
            if sentence_transformer=='our-embedding':
                model, tokenizer = set_CReSE_and_tokenizer(CAS.embedding_model_name)
                custom_embedder = CustomEmbedder(embedding_model_clip=model,
                                                tokenizer=tokenizer)
                topic_model = BERTopic(embedding_model=custom_embedder)
            else:
                sentence_model = SentenceTransformer(sentence_transformer)
                topic_model = BERTopic(embedding_model=sentence_model)
        else:
            topic_model = BERTopic()
        #resampling ecs
        ecs = resampling_ecs(ecs_total=ecs_total)
        #get labels for ecs
        #fit-transform bertopic
        topic_model.fit_transform(ecs)
        topic_num_ori = len(topic_model.get_topic_info())
        for num_cluster in num_clusters:
            topic_model.reduce_topics(ecs, nr_topics=num_cluster)
            #reduce topics and outliers
            topics = topic_model.topics_
            labels = topic_model.reduce_outliers(ecs, topics)
            #calculate correlation measures
            _, _, corr_pearson, corr_spearman = evaluate_clustering(EC_pairs_cr, method, ecs, labels, num_cluster)
            crs_dict_list[method][num_cluster]['corr_pearson'].append(corr_pearson)
            crs_dict_list[method][num_cluster]['corr_spearman'].append(corr_spearman)
        
    return crs_dict_list, topic_num_ori, topic_model, ecs
            
            
def get_ec_embeddings_by_biogpt(model, tokenizer, ecs, device):
    ec_embeddings_total = torch.Tensor()
    ec_embeddings_total = ec_embeddings_total.to(device)
    for i in tqdm(range(int(len(ecs)/CAS.embedding_group_n)+1)):
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
    
            
def update_crs_dict_list_embedding(crs_dict_list, method, model_name):
    #set bert model amd bert tokenizer
    if model_name=='our-embedding':
        model, tokenizer = set_CReSE_and_tokenizer(CAS.embedding_model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.to(device)
        
    #get ec_embeddings
    for _ in tqdm(range(CAS.repeated_n)):
        #resampling ecs
        ecs = resampling_ecs(ecs_total=ecs_total)
        if model_name=='our-embedding':
            _, ec_embeddings_total = get_ec_embeddings_from_CReSE(model, tokenizer, ecs)
        elif model_name=='microsoft/biogpt':
            ec_embeddings_total = get_ec_embeddings_by_biogpt(model, tokenizer, ecs, device)
        else:
            ec_embeddings_total = get_ec_embeddings_by_bert(model, tokenizer, ecs, device)
            
        ec_embeddings_total.to('cpu')
        data = {'ecs': ecs, 'emb': ec_embeddings_total}
        #perform clustering
        for num_cluster in CAS.num_clusters:
            #get labels for ecs
            _, labels, _ = clustering_ecs_by_embedding(data=data, 
                                                       num_clusters=num_cluster,
                                                       device=device)
            #calculate correlation measures
            _, _, corr_pearson, corr_spearman = evaluate_clustering(EC_pairs_cr, ecs, labels, 
                                                                    num_cluster)
            crs_dict_list[method][num_cluster]['corr_pearson'].append(corr_pearson)
            crs_dict_list[method][num_cluster]['corr_spearman'].append(corr_spearman)
            
    return crs_dict_list


#import ECs
dir_ecs = "C:/CReSE/datasets/CTgov_preprocessed"
tdt_noncommon = "train_data_tuples_train_total_noncommon_EC1509495_CT153361_230509"
with open(f"{dir_ecs}/{tdt_noncommon}.p", "rb") as f:
    tdt_train = pickle.load(f)
ecs_total = [tdt[0] for tdt in tdt_train]

#import clinical relevance
dir_cr = "C:/CReSE/datsets/clinical_relevance"
cr_fname = "clinical_relevance_gpt35turbo_ECpairs500_forclusteringeval"
with open(f"{dir_cr}/{cr_fname}.p", "rb") as f:
    EC_pairs_cr = pickle.load(f)
ecs_in_EC_pairs_cr = [[ec1, ec2] for ec1, ec2, _ in EC_pairs_cr]
ecs_in_EC_pairs_cr = [ec for ecs in ecs_in_EC_pairs_cr for ec in ecs]

##main code
crs_dict_list = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
total_results_str = ""
for method in [
    'TF-IDF', 
    'Bertopic_plain', 
    'Bertopic_biolink', 
    'Bertopic_trialbert', 
    'Bertopic_biogpt',
    'Bertopic_ours', 
    'Bertopic_biosimcse',
    'embedding_only_basebert', 
    'embedding_only_biolink',
    'embedding_only_trialbert', 
    'embedding_only_biogpt',
    'embedding_only_biosimcse',
    'embedding_only_ours'
    ]:
    print(f"Start analysis on method {method}!")
    if method=='TF-IDF':
        crs_dict_list = update_crs_dict_list(crs_dict_list=crs_dict_list,
                                             method=method,
                                             labling_func=clustering_TDIDF)
        results_str = print_out_95CI_corr(crs_dict_list, method, CAS.num_clusters)
        total_results_str += results_str
        
    elif 'Bertopic_' in method:
        if method=='Bertopic_plain':
            sentence_transformer = ''
        elif method=='Bertopic_biolink':
            sentence_transformer = 'michiyasunaga/BioLinkBERT-base'
        elif method=='Bertopic_trialbert':
            sentence_transformer = 'phdf33/trialbert-base'
        elif method=='Bertopic_biogpt':
            sentence_transformer = 'microsoft/biogpt'
        elif method=='Bertopic_biosimcse':
            sentence_transformer = 'kamalkraj/BioSimCSE-BioLinkBERT-BASE'
        elif method=='Bertopic_ours':
            sentence_transformer = 'our-embedding'
        crs_dict_list, topic_num_ori, topic_model, ecs_topic_model = \
            update_crs_dict_list_bertopic(crs_dict_list=crs_dict_list,
                                          method=method,
                                          sentence_transformer=sentence_transformer)
        print(f"Original number of topics: {topic_num_ori}")
        total_results_str += f"Original number of topics: {topic_num_ori}\n"
        results_str = print_out_95CI_corr(crs_dict_list, method, CAS.num_clusters)
        total_results_str += results_str
            
    elif 'embedding_only_' in method:
        if method=='embedding_only_basebert':
            model_name = "bert-base-uncased"
        elif method=='embedding_only_biolink':
            model_name = "michiyasunaga/BioLinkBERT-base"
        elif method=='embedding_only_trialbert':
            model_name == "phdf33/trialbert-base"
        elif method=='embedding_only_biogpt':
            model_name = "microsoft/biogpt"
        elif method=='embedding_only_biosimcse':
            model_name = 'kamalkraj/BioSimCSE-BioLinkBERT-BASE'
        elif method=='embedding_only_ours':
            model_name = 'our-embedding'
        crs_dict_list = update_crs_dict_list_embedding(crs_dict_list=crs_dict_list,
                                                      method=method,
                                                      model_name=model_name)
        results_str = print_out_95CI_corr(crs_dict_list, method, CAS.num_clusters)
        total_results_str += results_str
#save total_results_str as txt file
with open(f"CReSE_eval_results/clustering_eval_results_nsample50000_repeatedn{CAS.repeated_n}", 
          'w') as f:
    f.write(total_results_str)
    
    
if method=='BIOSSES':
    #compare embedding models on BIOSSES dataset
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity


    for model_name in [
        "michiyasunaga/BioLinkBERT-base",
        "phdf33/trialbert-base",
        "microsoft/biogpt",
        "kamalkraj/BioSimCSE-BioLinkBERT-BASE"
        ]:
        
        dataset = load_dataset("biosses", split="train")
        sentence_pairs_BIOSSES = [(d['sentence1'], d['sentence2'], d['score']) for d in dataset]
        sentences1 = [spb[0] for spb in sentence_pairs_BIOSSES]
        sentences2 = [spb[1] for spb in sentence_pairs_BIOSSES]
        scores = [spb[2] for spb in sentence_pairs_BIOSSES]

        #set bert model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        sentences1_embeddings = get_ec_embeddings_by_bert(model, tokenizer, sentences1, 'cpu')
        sentences2_embeddings = get_ec_embeddings_by_bert(model, tokenizer, sentences2, 'cpu')
        consine_similarities = [cosine_similarity(np.array(sent1_emb).reshape(1, -1), np.array(sent2_emb).reshape(1, -1)) for sent1_emb, sent2_emb in zip(sentences1_embeddings, sentences2_embeddings)]

        consine_similarities = min_max_scaling(consine_similarities)    
        consine_similarities = [c[0][0] for c in consine_similarities]

        corr_pearson, _ = stats.pearsonr(scores, consine_similarities)
        corr_spearman, _ = stats.spearmanr(scores, consine_similarities)
        print(f"Pearson correlation coefficient: {corr_pearson:.3f}")
        print(f"Spearman correlation coefficient: {corr_spearman:.3f}")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(consine_similarities, scores, s=25, alpha=0.5)
        ax.set_xlabel('Consine similarity of EC-EC CReSE embeddings')
        ax.set_ylabel('Simialrity score in BIOSSES')
        ax.set_title('Correlation between embedding similarity and score in BIOSSES')
        plt.show()

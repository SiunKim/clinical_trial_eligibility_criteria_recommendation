import os
import random; random.seed(42)
import pickle

from collections import defaultdict

from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader    
from transformers import AutoTokenizer

import sys; sys.path.append("C:CReSE/CReSE")
#for setting bertopoic from CReSE
from config import CFG
from model import CReSE, EcEncoder, ProjectionHead
from train import set_CReSE_and_tokenizer
from inference_fast import foward_with_cls_hidden_state_ec
from get_bertopic_from_CReSE import CAS, CustomEmbedder
from bertopic import BERTopic

from utils_eval_EC_recom import get_fname_ec_title_model
from utils_eval_EC_recom import measure_precision_k, measure_recall_k
from utils_eval_EC_recom import measure_average_precision_k
from utils_eval_EC_recom import ci_100



class ESV():
    do_eval = True 
    
    #directories
    dir_best_model = "C:/CReSE/CReSE/best_CReSE_models"
    dir_eval_dataset = "C:/CReSE/datasets/evaluation_datasets"
    
    #EC recommendation model setting - searching model files in dir_best_model
    input_type = 'only_title'
    # input_type = 'title+summary'
    # input_type = 'title+CTinfo'
    # input_type = 'title+summary+CTinfo'
    sample_n = 'use_total_positive' 
    Ent = 8
    lr = '1e-05'
    pnr = 1
    #batch size for inference
    batch_size = 8 #can use a larger batch_size if you have larger VRAM...
    
    #Bert-model
    bert_model = "michiyasunaga/BioLinkBERT-base"
    
    #setting categories for evaluation
    # categories = ["before2010", "beforeCOVID", "afterCOVID", "oncology", "infectious", "cardiology", "gastrenterology", "rheumatology", "immunology", "pulmonology", "hematology", "neurology", "nephrology", "metabolic", "dermatology"]
    categories = ["total"]
    #number of clinical trials in evaluation dataset (fixed to 100)
    CT_num = 100
    top_Ks_eval = [1, 5, 'ec_num_ori']
    topNs = [1]; topN_selected = 1

    #topic number for evaluation!
    ecs_topicn = 100 # [100, 200, 300]
    #number of random topic rankings
    random_repeat_n = 100
    
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

def tokenize_ecs_titles(tokenizer, ecs_titles):
    #wrap single ec_title as list
    if type(ecs_titles)==str:
        ecs_titles = [ecs_titles]
    #tokenizing ecs_titles       
    input_ids = []; attention_mask = []
    for ec_title in ecs_titles:
        encoded_ec_title = tokenizer.encode_plus(ec_title,
                                              add_special_tokens = True, 
                                              max_length = 256,
                                              pad_to_max_length = True,
                                              return_attention_mask = True, 
                                              return_tensors = 'pt'
                                              )
        input_ids.append(encoded_ec_title['input_ids'])
        attention_mask.append(encoded_ec_title['attention_mask'])
    # Convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    
    return input_ids, attention_mask
        
    
def get_topic_probs_for_title(ec_title_model,
                              dataloader):
    #get probs of ecs for a given title through inference
    probs_total = []
    cls_hidden_state_ec_total = []
    for batch in dataloader:
        #express batch for inference
        input_ids_ec, input_ids_title, attention_mask_ec, attention_mask_title = \
            batch
        #get logits from ec_title_model
        logits, cls_hidden_state_ec, _, _, _ = \
            ec_title_model(
                input_ids_ec=input_ids_ec.to(ESV.device),
                input_ids_title=input_ids_title.to(ESV.device),
                attention_mask_ec=attention_mask_ec.to(ESV.device), 
                attention_mask_title=attention_mask_title.to(ESV.device)
                )
        #calculate probs for each ecs through softmax
        probs = [F.softmax(l, dim=0)[1].item() for l in logits]
        #extend to probs_total
        probs_total += probs
        
        #append cls_hidden_state_ec
        cls_hidden_state_ec_total.append(cls_hidden_state_ec.detach().to('cpu'))
        
    return probs_total, cls_hidden_state_ec_total


def get_topic_probs_for_title_fast(ec_title_model,
                                   cls_hidden_state_ec_total,
                                   dataloader,):
    #get probs of ecs for a given title through inference
    probs_total = []
    for cls_hidden_state_ec, batch in zip(cls_hidden_state_ec_total, dataloader):
        #express batch for inference
        _, input_ids_title, _, attention_mask_title = batch
        #get logits from ec_title_model
        logits  = \
            foward_with_cls_hidden_state_ec(ec_title_model,
                                            cls_hidden_state_ec.to(ESV.device),
                                            input_ids_title=input_ids_title.to(ESV.device),
                                            attention_mask_title=attention_mask_title.to(ESV.device))
        #calculate probs for each ecs through softmax
        probs = [F.softmax(l, dim=0)[1].item() for l in logits]
        #extend to probs_total
        probs_total += probs
        
    return probs_total
        

def get_topic_ranking_from_probs(probs_total, repr_ecs, ranking_by_topics=True):
    #check the lenght of probs_total
    assert len(probs_total)==len(repr_ecs), \
        "The length of probs_total and ecs_selected must be same!"

    #get topic_rankings for a given title
    topic_rankings = []; topic_probs = []
    repr_ecs_selected = []
    #sort topics in terms of prob-scores from ec_title_model
    probs_total_sorted = sorted(probs_total, reverse=True)    
    for prob in probs_total_sorted:
        ec_index = probs_total.index(prob)
        ec = repr_ecs[ec_index][0]
        topic = repr_ecs[ec_index][1]
        #append ec_topic (except -1, 0 topics)
        if (topic not in [-1, 0]):
            if ranking_by_topics:
                if (topic not in topic_rankings):
                    topic_rankings.append(topic)
                    topic_probs.append(prob)
                    repr_ecs_selected.append(ec)
            else:
                topic_rankings.append(topic)
                topic_probs.append(prob)
                repr_ecs_selected.append(ec)
                
    return topic_rankings, topic_probs, repr_ecs_selected



def import_ec_title_model_and_tokenizer(fname_ec_title_model, device):
    ec_title_model = torch.load(f"{ESV.dir_best_model}/{fname_ec_title_model}")
    tokenizer = AutoTokenizer.from_pretrained(ESV.bert_model,
                                              do_lower_case=False)
    ec_title_model = ec_title_model.to(device)
    
    return ec_title_model, tokenizer


def import_eval_dataset(ecs_topicn, category, CT_num, topNs):
    #import repr_ecs
    dir_repr_ecs = f"{ESV.dir_eval_dataset}/repr_ecs"
    if type(ecs_topicn)==str:
        with open(f"{dir_repr_ecs}/repr_ecs_total_100000.p", 'rb') as f:
            repr_ecs = pickle.load(f)
        ecs_topicn = int(ecs_topicn.split('_')[1])
    else:        
        with open(f"{dir_repr_ecs}/repr_ecs_topicn{ecs_topicn}.p", 'rb') as f:
            repr_ecs = pickle.load(f)
        
    #import bertopic model for evaluation and ecs (docs) used to train the bertopic model
    dir_bertopic_eval = f"{ESV.dir_eval_dataset}/bertopic_eval"
    bertopic_model = BERTopic()
    bertopic_model = bertopic_model.load(f"{dir_bertopic_eval}/bertopic_eval_topicn{ecs_topicn}")
    with open(f"{dir_bertopic_eval}/ecs_eval_100000.p", 'rb') as f:
        ecs_for_bertopic = pickle.load(f)
        
    #import titles and topics data for evaluation
    dir_titles_ec_topics_eval = f"{ESV.dir_eval_dataset}/titles_ec_topics_eval"
    dir_tete = f"{dir_titles_ec_topics_eval}/{category}"
    #import fname_titles 
    fname_titles = f"titles_by_nct_ids_{category}_nctids{CT_num}_topicn{ecs_topicn}"
    with open(f"{dir_tete}/{fname_titles}.p", 'rb') as f:
        titles_by_nct_ids = pickle.load(f)
    #import topics_topNs_by_nct_ids
    topics_topNs_by_nct_ids = {}
    for topN in topNs:
        fname_topics_topN = fname_titles.replace('titles', f'topics_top{topN}')
        with open(f"{dir_tete}/{fname_topics_topN}.p", 'rb') as f:
            topics_topN = pickle.load(f)
        topics_topNs_by_nct_ids[f'top{topN}'] = topics_topN
        
    return repr_ecs, ecs_for_bertopic, titles_by_nct_ids, topics_topNs_by_nct_ids


def inference_for_titles(titles, ec_title_model, tokenizer,
                         repr_ecs,
                         input_ids_ec, attention_mask_ec,
                         ranking_by_topics=True):
    topic_rankings_total = []
    topic_probs_total = []
    repr_ecs_selected_total = []
    is_First = True
    for title in tqdm(titles):
        input_ids_title, attention_mask_title = tokenize_ecs_titles(tokenizer, title)
        #reshape titles for inference
        input_ids_title = input_ids_title.expand(len(input_ids_ec), -1) 
        attention_mask_title = attention_mask_title.expand(len(input_ids_ec), -1) 
        #set dataloader
        dataset = TensorDataset(input_ids_ec, input_ids_title, 
                                attention_mask_ec, attention_mask_title)
        dataloader = DataLoader(dataset,
                                batch_size=ESV.batch_size)
        #get topic_rankings and topic_probs for a given title
        if is_First:
            probs_total, cls_hidden_state_ec_total = \
                get_topic_probs_for_title(ec_title_model,
                                          dataloader)
            is_First = False
        else:
            probs_total = \
                get_topic_probs_for_title_fast(ec_title_model,
                                               cls_hidden_state_ec_total,
                                               dataloader)
                
        #finalized inference, from probs_total
        topic_rankings, topic_probs, repr_ecs_selected = get_topic_ranking_from_probs(probs_total, repr_ecs, ranking_by_topics)
            
        #append topic_rankings, topic_probs and repr_ecs_selected
        topic_rankings_total.append(topic_rankings)
        topic_probs_total.append(topic_probs)
        repr_ecs_selected_total.append(repr_ecs_selected)
        
    return topic_rankings_total, topic_probs_total, repr_ecs_selected_total
        
        
def generate_random_total_rankings_probs_repeat(CT_num):
    #get topic_rankings_random_total_repeat
    topic_rankings_random_total_repeat = []
    topic_probs_random_total_repeat = []
    for i in range(ESV.random_repeat_n):
        topic_rankings_random_total = []
        topic_probs_random_total = []
        for j in range(CT_num):
            #genearte topic_rankings_random
            topic_rankings_random = list(range(1, ESV.ecs_topicn-1))
            random.seed(1000*i + j)
            random.shuffle(topic_rankings_random)
            topic_rankings_random_total.append(topic_rankings_random)
            #generate topic_probs_random
            topic_probs_random = [random.uniform(0, 1) for _ in range(298)]
            topic_probs_random.sort()
            topic_probs_random_total.append(topic_probs_random)
            
        topic_rankings_random_total_repeat.append(topic_rankings_random_total)
        topic_probs_random_total_repeat.append(topic_probs_random_total)
    
    return topic_rankings_random_total_repeat, topic_probs_random_total_repeat


def eval_topic_ranking(topics_total, 
                       topic_rankings_total, 
                       topic_probs_total,
                       top_Ks_eval,
                       std=True):
    measure_func_names = [(measure_precision_k, 'precision@K'),
                          (measure_recall_k, 'racall@K'), 
                          (measure_average_precision_k, 'MAP@K')]
    
    eval_metrics_total = defaultdict(lambda: defaultdict(list))
    #evaluation on topic_rankings
    for topics_true, topic_rankings, topic_probs in \
        zip(topics_total, 
            topic_rankings_total, 
            topic_probs_total):
        for top_K in top_Ks_eval:
            #calculate precision, recall, AP@K
            for measure_func, measure_name in measure_func_names:
                measure_value = measure_func(topics_true, topic_rankings, top_K) 
                #append measure_value
                eval_metrics_total[top_K][measure_name].append(measure_value)
                                
    #get average for all top_K and measure_name in eval_metrics_total
    if std:
        eval_metrics_total = {top_K: {measure_name: 
            f"{np.average(perf_metrics):.2f} +/- {np.std(perf_metrics)}" 
                                        for measure_name, perf_metrics in eval_m.items()} 
                                for top_K, eval_m in eval_metrics_total.items()}
    else:
        eval_metrics_total = {top_K: {measure_name: f"{np.average(perf_metrics):.2f}" 
                                        for measure_name, perf_metrics in eval_m.items()} 
                                for top_K, eval_m in eval_metrics_total.items()}
    
    return eval_metrics_total


def eval_topic_ranking_random(topics_total,
                              topic_rankings_random_total_repeat,
                              topic_probs_random_total_repeat,
                              top_Ks_eval):
    eval_metrics_random_total_repeat = defaultdict(lambda: defaultdict(list))
    for topic_rankings_random_total, topic_probs_random_total \
        in zip(topic_rankings_random_total_repeat, topic_probs_random_total_repeat):
            eval_metrics_random_total = eval_topic_ranking(topics_total, 
                                                           topic_rankings_random_total,
                                                           topic_probs_random_total,
                                                           top_Ks_eval,
                                                           std=False)
            
            #append perf_metrics in eval_metrics_random_total_repeat
            for top_K, eval_m in eval_metrics_random_total.items():
                for measure_name, perf_metrics in eval_m.items():
                    #change measure_name (add '_random')
                    eval_metrics_random_total_repeat[top_K][measure_name + '_random'].append(perf_metrics)
                    
    #calculate average and 95 CI again
    eval_metrics_random_total = defaultdict(lambda: defaultdict(list))
    for top_K, eval_m in eval_metrics_random_total_repeat.items():
        for measure_name, perf_metrics_repeat in eval_m.items():
            perf_metrics_repeat = [float(p) for p in perf_metrics_repeat]
            measure_average = np.average(perf_metrics_repeat)
            CI_95_LB = np.percentile(perf_metrics_repeat, 2.5)
            CI_95_UB = np.percentile(perf_metrics_repeat, 97.5)
            
            eval_metrics_random_total[top_K][measure_name] = \
                f"{measure_average:.2f} [{CI_95_LB:.2f}, {CI_95_UB:.2f}]"
                
    return eval_metrics_random_total
    
    

##main##
def main(fname_ec_title_model):
    #import ec_title_best_model
    ec_title_model, tokenizer = import_ec_title_model_and_tokenizer(fname_ec_title_model, 
                                                                    ESV.device)

    #evaluation settings
    for category in ESV.categories:
        print(f"Category: {category}")
        print(f"Ent: {ESV.Ent}")
        repr_ecs, _, titles_by_nct_ids, topics_topNs_by_nct_ids = \
            import_eval_dataset(ESV.ecs_topicn, category, ESV.CT_num, ESV.topNs)
            
        #set titles for model inference
        nct_ids = list(titles_by_nct_ids.keys())
        titles = [titles_by_nct_ids[nct_id][ESV.input_type.replace('CT', 'CT_')] 
                    for nct_id in nct_ids]
        topics_total = [topics_topNs_by_nct_ids[f'top{ESV.topN_selected}'][nct_id] 
                            for nct_id in nct_ids]
        
        #model inference - get ranking of repr_ecs for a given title
        repr_ecs_only_ec = [ec for ec, _ in repr_ecs]
        input_ids_ec, attention_mask_ec = tokenize_ecs_titles(tokenizer, repr_ecs_only_ec)
        #inference for titles
        topic_rankings_total, topic_probs_total, repr_ecs_selected_total = \
            inference_for_titles(titles, ec_title_model, tokenizer,
                                 repr_ecs,
                                 input_ids_ec, attention_mask_ec)
            
        #generate random
        topic_rankings_random_total_repeat, topic_probs_random_total_repeat = \
            generate_random_total_rankings_probs_repeat(ESV.CT_num)
            
        #perform model inference for an individual title
        #calculate performance measures
        eval_metrics_total = eval_topic_ranking(topics_total, 
                                                topic_rankings_total, 
                                                topic_probs_total,
                                                ESV.top_Ks_eval)
        eval_metrics_random_total = \
            eval_topic_ranking_random(topics_total, 
                                      topic_rankings_random_total_repeat,
                                      topic_probs_random_total_repeat,
                                      ESV.top_Ks_eval)
        
        #result print-out
        print(f" - input_type: {ESV.input_type}")
        print(f" - pnr: {ESV.pnr}")
        print(f" - ecs_topicn: {ESV.ecs_topicn}")
        print(f" - category: {category}")
        print(f" - CT_num: {ESV.CT_num}")
        print(f" - topN_selected: {ESV.topN_selected}")
        print(f" - performance metrics: {eval_metrics_total}")
        print(f" - eval_metrics_random_total: {eval_metrics_random_total}")
        
        
        print("-"*30)
        pre_at_1 =  float(eval_metrics_total[1]['precision@K'].split()[0])*100
        map_at_5 = float(eval_metrics_total[5]['MAP@K'].split()[0])*100
        pre_at_ecnumori = float(eval_metrics_total['ec_num_ori']['precision@K'].split()[0])*100
        pre_95ci = eval_metrics_random_total[1]['precision@K_random']
        map_95ci = eval_metrics_random_total[5]['MAP@K_random']
        pre2_95ci = eval_metrics_random_total['ec_num_ori']['precision@K_random']
        eval_result_focused = f"\n    - precision@1: {pre_at_1:.1f} ({ci_100(pre_95ci)})\n    - MAP@5: {map_at_5:.1f} ({ci_100(map_95ci)})\n    - precision@ecnumori: {pre_at_ecnumori:.1f} ({ci_100(pre2_95ci)})"
        
        print(f" - eval_result_focused: {eval_result_focused}")
        
        #save evaluation results as .txt file
        results_text = f" - input_type: {ESV.input_type}\n - pnr: {ESV.pnr}\n - ecs_topicn: {ESV.ecs_topicn}\n - category: {category}\n - CT_num: {ESV.CT_num}\n - topN_selected: {ESV.topN_selected}\n\n\n - performance metrics: {eval_metrics_total} \n - performance metrics (random): {eval_metrics_random_total}"
        
        with open(f"EC_recommended_eval_results/performance_eval_Ent{ESV.Ent}_{category}_ecs_topicn{ESV.ecs_topicn}_0717.txt", "w") as f:
            f.write(results_text)
            
        return pre_at_1, map_at_5, pre_at_ecnumori
      
      
      
#main
if ESV.do_eval:
    fname_ec_title_model = get_fname_ec_title_model(ESV.input_type, ESV.sample_n, 
                                                    ESV.lr, ESV.pnr, ESV.Ent)
    main(fname_ec_title_model)
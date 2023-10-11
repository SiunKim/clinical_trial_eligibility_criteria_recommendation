
import os
import sys
import random
import pickle

from tqdm import tqdm

from collections import defaultdict

import sys; sys.path.append('C:/CReSE/evaluations')
from get_bertopic_from_CReSE import CustomEmbedder
from load_bertopic_from_CREEP import load_bertopic_with_embeddings
from CREEP.model import CLIPModel
from CREEP.modules import EcEncoder, ProjectionHead



def set_ct_title_info(tdt, input_type):
    title = tdt[1]; summary = tdt[3]; ctinfo = tdt[5]
    #set ct_title_info
    if input_type=='only_title':
        ct_title_info = title
    elif input_type=='title+summary':
        ct_title_info = ' [SEP] '.join([title, summary])
    elif input_type=='title+CTinfo':
        ct_title_info = ' [SEP] '.join([title, ctinfo])
    elif input_type=='title+summary+CTinfo':
        ct_title_info = ' [SEP] '.join([title, summary, ctinfo])
        
    return ct_title_info


def get_topics_top_n(bertopic_model, ecs_total, top_n=5):
    topics_top_n_total = []
    for ec in ecs_total:
        #find_topics
        topics, _ = bertopic_model.find_topics(ec, top_n=top_n)
        topics_top_n_total.append(topics)
        
    return topics_top_n_total


def get_topics_top_n_total(bertopic_model, ecs_total, 
                           tdt_fname, dir_CTgov,
                           top_n=5):
    #get or load topics_total
    tdt_fname_type = tdt_fname.split('_noncommon')[0].split('_tuples_')[1]
    topics_total_dir = f"{dir_CTgov}/prep_EC_recom_dataset/ec_topics"
    topics_total_50_fname = f"{tdt_fname_type}_topics_top50.p"
    topics_total_fname = f"{tdt_fname_type}_topics_top{top_n}.p"
    
    if topics_total_50_fname in os.listdir(topics_total_dir) and top_n<=50: # load topics_total
        with open(f"{topics_total_dir}/{topics_total_50_fname}", "rb") as f:
            topics_top_50_total = pickle.load(f)
        topics_top_n_total = [topics_top_50[:top_n] for topics_top_50 in topics_top_50_total]
    else:
        #get topic number from bertopic model
        print("Start getting topics for ecs_total!")
        topics_top_n_total = []; subgroup_len = 200
        for i in tqdm(range(int(len(ecs_total)/subgroup_len)+1)):
            topics_top_n = get_topics_top_n(bertopic_model, 
                                            ecs_total[subgroup_len*i:subgroup_len*(i+1)]) 
            topics_top_n_total += topics_top_n
        #save 
        with open(f"{topics_total_dir}/{topics_total_fname}", "wb") as f:
            pickle.dump(topics_top_n_total, f)
        print("Finish getting topics for ecs_total!")
            
    return topics_top_n_total


def get_tdts_ecs_topics_top_by_nctids(ecs_total,
                                      topics_top_n_total,
                                      tdt_fname, 
                                      dir_CTgov,
                                      top_n=5):
    #get or load topics_total
    tdt_fame_short = tdt_fname.replace('train_data_tuples_', '')
    by_nctids_dir = f"{dir_CTgov}/prep_EC_recom_dataset/by_nctids"
    tdts_by_nctids_fname = f"tdts_by_nctids_{tdt_fame_short}.p"
    ecs_by_nctids_fname = f"ecs_by_nctids_{tdt_fame_short}.p"
    topics_top_by_nctids_fname = f"topics_top{top_n}_by_nctids_{tdt_fame_short}.p"
    
    if topics_top_by_nctids_fname not in os.listdir(by_nctids_dir):
        print("Start sorting tdts and ecs by nct_ids")
        #get nct_ids from tdt_total
        nct_ids = list(set([tdt[2] for tdt in tdt_total]))
        print(f"Len of nct_ids: {len(nct_ids)}")
        #get tdts, ecs, topics_top by_nctids
        tdts_by_nctids = {}; ecs_by_nctids = {}; topics_top_by_nctids = {}
        for nct_id in tqdm(nct_ids):
            tdts = [tdt for tdt in tdt_total if tdt[2]==nct_id]
            tdts_by_nctids[nct_id] = tdts
            ecs = [tdt[0] for tdt in tdts]
            ecs_by_nctids[nct_id] = ecs
            topics_top_by_nctids[nct_id] = [topics_top_n_total[ecs_total.index(ec)] for ec in ecs]
        with open(f"{by_nctids_dir}/{tdts_by_nctids_fname}", "wb") as f:
            pickle.dump(tdts_by_nctids, f)
        with open(f"{by_nctids_dir}/{ecs_by_nctids_fname}", "wb") as f:
            pickle.dump(ecs_by_nctids, f)
        with open(f"{by_nctids_dir}/{topics_top_by_nctids_fname}", "wb") as f:
            pickle.dump(topics_top_by_nctids, f)
            
    else: 
        with open(f"{by_nctids_dir}/{tdts_by_nctids_fname}", "rb") as f:
            tdts_by_nctids = pickle.load(f)
        with open(f"{by_nctids_dir}/{ecs_by_nctids_fname}", "rb") as f:
            ecs_by_nctids = pickle.load(f)
        with open(f"{by_nctids_dir}/{topics_top_by_nctids_fname}", "rb") as f:
            topics_top_by_nctids = pickle.load(f)
            
    return tdts_by_nctids, ecs_by_nctids, topics_top_by_nctids


def get_positive_EC_recom_data(tdt_total):
    print("Start getting positive pairs for EC recommendation!")
    ec_title_positive_pairs = defaultdict(list)
    for tdt in tqdm(tdt_total):
        ec = tdt[0]
        ct_title_info = {}
        for input_type in INPUT_TYPES:
            ct_title_info[input_type] = set_ct_title_info(tdt, input_type)
            #append ec and ct_title_info tuple
            ec_title_positive_pairs[input_type].append((ec, ct_title_info[input_type]))
    print("Finish getting positive pairs for EC recommendation!")
    
    return ec_title_positive_pairs


def select_nctids_above_EC_numb_th(tdts_by_nctids, EC_numb_th):
    #Only used ecs in CTs where the number of ecs are larger than the threshold
    nct_ids_above_EC_numb_th = [nct_id for nct_id, tdts in tdts_by_nctids.items()
                            if len(tdts)>=EC_numb_th]
    
    return nct_ids_above_EC_numb_th


def get_negative_EC_recom_data(tdts_by_nctids, 
                               topics_top_by_nctids, 
                               topics_top_n_total, 
                               EC_numb_th):
    #select nct_ids above_EC_numb_th
    nct_ids_above_EC_numb_th = select_nctids_above_EC_numb_th(tdts_by_nctids, EC_numb_th)
    #get negative pairs for EC recommendation
    print(f"Start getting negative pairs for EC recommendation! - EC_numb_th {EC_numb_th}")
    etn_pairs = defaultdict(list)
    for input_type in INPUT_TYPES:
        #calculate n_negative_per_ct
        n_negative_per_ct = int(len(etp_pairs[input_type])/len(nct_ids_above_EC_numb_th))
        for nct_id in tqdm(nct_ids_above_EC_numb_th):
            #get unique topics in a given nct_id
            topics_top_nctid = set([t for topics in topics_top_by_nctids[nct_id] for t in topics])
            #get tdt for a given nct_id
            try:
                tdt = tdts_by_nctids[nct_id][0]
            except IndexError:
                continue
            #set ct_title_info for input_type
            ct_title_info = set_ct_title_info(tdt, input_type)
            #randomly sample individual ec and check the ec appropriate for negative sample
            count = 0
            while count<=n_negative_per_ct:
                ec_rand_index = random.choice(range(len(ecs_total)))
                ec_rand = ecs_total[ec_rand_index]
                ec_rand_topics_top = topics_top_n_total[ec_rand_index]
                #only when where is no overlap topic between ec_rand_topics_top and topics_top_nctid
                if len(set(ec_rand_topics_top)&set(topics_top_nctid))==0: 
                    etn_pairs[input_type].append((ec_rand, ct_title_info))
                    count += 1
                    
    return etn_pairs, n_negative_per_ct


#preprocessing train dataset for recommeding ECs from CT infomation
INPUT_TYPES = ['only_title',
                'title+summary',  
                'title+CTinfo',
                'title+summary+CTinfo']
EC_numb_th = 8 #for selecting negative ec_title pairs
    
#load bertopic model
bertopic_seed = 0
bertopic_model = load_bertopic_with_embeddings(cluster_n=100, 
                                               bertopic_seed=bertopic_seed,
                                               add_emb_model=False)

#prepare EC-recommendation dataset - positive/negative
dir_CTgov = "C:/CReSE/datasets"
dir_tdt = dir_CTgov + "/CTgov_preprocessed"
dir_etp = f"{dir_CTgov}/prep_EC_recom_dataset/"
for tdt_fname in [
    "train_data_tuples_after_2021_Sep_test_noncommon_EC50483_CT4938_230523",
    "train_data_tuples_before_2021_Sep_test_noncommon_EC48932_CT4899_230523",
    "train_data_tuples_before_2021_Sep_valid_noncommon_EC47865_CT4905_230523",
    "train_data_tuples_train_total_noncommon_EC1509495_CT153361_230509"
                  ]:
    print(f"tdt_fname: {tdt_fname}")
    tdt_fame_short = tdt_fname.split('tuples_')[1].split('_noncommon')[0]
    dir_train_test = 'train' if 'train_total' in tdt_fname else 'test'
        
    #import tdt_total and get ecs_total
    with open(f"{dir_tdt}/{tdt_fname}.p", "rb") as f:
        tdt_total = pickle.load(f)    
    ecs_total = [tdt[0] for tdt in tdt_total]
    
    #get topics_top_n_total
    topics_top_n_total = get_topics_top_n_total(bertopic_model, ecs_total, 
                                                tdt_fname, dir_CTgov,
                                                top_n=1)
    
    #get or load tdts_by_nctids and ecs_by_nctids
    tdts_by_nctids, ecs_by_nctids, topics_top_by_nctids = \
        get_tdts_ecs_topics_top_by_nctids(ecs_total,
                                          topics_top_n_total,
                                          tdt_fname, 
                                          dir_CTgov,
                                          top_n=1)
        
    #get positive EC-title pairs for training
    etp_pairs = get_positive_EC_recom_data(tdt_total)
    #save ec_title_positive_pairs
    for input_type in tqdm(INPUT_TYPES):
        etp_fname = f"etp_positive_{tdt_fame_short}_eclens{len(etp_pairs[input_type])}_inputtype_{input_type}.p"
        with open(f"{dir_etp}/{dir_train_test}/{etp_fname}", "wb") as f:
            pickle.dump(etp_pairs[input_type], f)
    
    #get negative EC-title pairs for training
    etn_pairs, n_negative_per_ct = get_negative_EC_recom_data(tdts_by_nctids, 
                                            topics_top_by_nctids, 
                                            topics_top_n_total, 
                                            EC_numb_th)
    #save ec_title_negative_pairs
    for input_type in INPUT_TYPES:        
        enp_fname = f"etp_negative_{tdt_fame_short}_eclens{len(etn_pairs[input_type])}_nnpc{n_negative_per_ct}_Ent{EC_numb_th}_inputtype_{input_type}.p"
        with open(f"{dir_etp}/{dir_train_test}/{enp_fname}", "wb") as f:
            pickle.dump(etn_pairs[input_type], f)
    print("End getting negative pairs for EC recommendation!")

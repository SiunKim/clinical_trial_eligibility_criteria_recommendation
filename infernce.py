import random; random.seed(42)

from model import CReSE

import torch

import sys; sys.append('C:/CReSE/evaluations')
from utils_eval_EC_recom import get_fname_ec_title_model
from eval_EC_recom_model import import_ec_title_model_and_tokenizer, import_eval_dataset
from eval_EC_recom_model import tokenize_ecs_titles, inference_for_titles



def inference_for_title(title: str, top_K: int, ranking_by_topics: bool):
    _, _, repr_ecs_selected_total = \
            inference_for_titles([title], 
                                 ec_title_model, 
                                 tokenizer,
                                 repr_ecs,
                                 input_ids_ec, attention_mask_ec,
                                 ranking_by_topics)
            
    return repr_ecs_selected_total[0][:top_K]


if __name__ == "__main__":
    ##main##
    dir_best_model = "C:/CReSE/CReSE/best_CReSE_models"
    input_type = 'only_title'
    sample_n = 'use_total_positive' # 1000000
    pnr = 1
    Ent = 8
    lr = '1e-05'
    fname_ec_title_model = get_fname_ec_title_model(input_type, sample_n, lr, pnr, Ent)

    #import ec_title_best_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ec_title_model, tokenizer = import_ec_title_model_and_tokenizer(fname_ec_title_model, device)

    #import evaluation dataset
    ecs_topicn = 100 # [50, 100, 150, 200, 300, 'total_{topicn}']
    category = "total"; CT_num = 100; topNs = [1]
    repr_ecs, _, _, _ = \
        import_eval_dataset(ecs_topicn, category, CT_num, topNs)
    #re-select repr_ecs
    repr_ecs_n = 50000
    random.shuffle(repr_ecs); repr_ecs = repr_ecs[:repr_ecs_n]
    
    #set input_ids_ec for inference
    repr_ecs_only_ec = [ec for ec, _ in repr_ecs]
    input_ids_ec, attention_mask_ec = tokenize_ecs_titles(tokenizer, repr_ecs_only_ec)
                
    #inference for titles
    while True:
        title = input("Enter a title of clinical trial (or press Enter to exit): ")
        if title=="":
            break
        top_K = int(input("Enter the integer for top_K: "))
        top_K_repr_ecs = inference_for_title(title, top_K=top_K, ranking_by_topics=True)
                
        #print top_K_repr_ecs
        for index, top_repr_ec in enumerate(top_K_repr_ecs):
            print(f"{index + 1}. {top_repr_ec}")

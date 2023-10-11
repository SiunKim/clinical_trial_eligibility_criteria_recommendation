def measure_precision_k(topics_true, topics_pred, top_K):
    #set topics_pred_top_K and probs_top_K
    if top_K=='ec_num_ori':
        topics_pred_top_K = topics_pred[:len(topics_true)]
        # probs_top_K = topic_probs[:len(topics_true)]
    else:
        topics_pred_top_K = topics_pred[:top_K]
        # probs_top_K = topic_probs[:top_K]
    
    if type(topics_pred_top_K)==list:
        topics_pred_top_K = set(topics_pred_top_K)
    return len(topics_true & topics_pred_top_K)/len(topics_pred_top_K)
        
        
def measure_recall_k(topics_true, topics_pred, top_K):
    #set topics_pred_top_K and probs_top_K
    if top_K=='ec_num_ori':
        topics_pred_top_K = topics_pred[:len(topics_true)]
        # probs_top_K = topic_probs[:len(topics_true)]
    else:
        topics_pred_top_K = topics_pred[:top_K]
        # probs_top_K = topic_probs[:top_K]
    
    if type(topics_pred_top_K)==list:
        topics_pred_top_K = set(topics_pred_top_K)
    return len(topics_true & topics_pred_top_K)/len(topics_true)


def measure_average_precision_k(topics_true, topics_pred, top_K):
    if type(top_K)==str:
        return 0.0
    else:
        precision_sum = 0.0
        for top_i in range(1, top_K + 1):
            precision_sum += measure_precision_k(topics_true, topics_pred, top_i)
            
        return precision_sum / top_K


def get_fnames_ec_title_model(input_type, sample_n, lr, pnr, Ent):
    fnames_ec_title_model = [fname for fname in os.listdir(ESV.dir_best_model) 
                            if (f'input_type{input_type}_' in fname)
                            and (f'samplen{sample_n}_' in fname) 
                            and (f'lr_{lr}_' in fname)
                            and (f'posnegratio{pnr}_' in fname)
                            and (f'Ent{Ent}_' in fname)
                            and ('.pt' in fname)]
    
    return fnames_ec_title_model

    
def get_fname_ec_title_model(input_type, sample_n, lr, pnr, Ent):
    fname_ec_title_model = get_fnames_ec_title_model(input_type, sample_n, lr, pnr, Ent)[0]
    
    return fname_ec_title_model


def ci_100(ci_str):
    median = float(ci_str.split()[0])
    ci_lb = float(ci_str.split('[')[1].split(',')[0])
    ci_ub = float(ci_str.split('[')[1].split()[1].replace(']', ''))
    
    return f"{median*100:.1f} [{ci_lb*100:.1f}, {ci_ub*100:.1f}]"
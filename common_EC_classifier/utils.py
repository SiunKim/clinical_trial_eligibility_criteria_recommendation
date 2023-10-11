import random
import datetime

import numpy as np

from sklearn.metrics import f1_score



def impute_by_last_obs(list_nan):
    list_nan = list(list_nan); last_observation = np.nan
    for i in range(len(list_nan)):
        if type(list_nan[i])==str:
            last_observation = list_nan[i]
        else:
            list_nan[i] = last_observation
            
    return list_nan


def get_negative_EC_title_pairs(EC_title_pairs):
    EC_title_pairs_negative = []
    while len(EC_title_pairs_negative)<len(EC_title_pairs):
        EC_NCT_ind = random.choice(EC_title_pairs)
        title_NCT_ind = random.choice(EC_title_pairs)
        
        if EC_NCT_ind[2]!=title_NCT_ind[2]: #from differnt clinical trials/NCT id
            EC_title_pairs_negative.append((EC_NCT_ind[0], title_NCT_ind[1]))
            
    return EC_title_pairs_negative
        
        
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_f1(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return f1_score(pred_flat, labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))
import random

from collections import Counter

import torch

import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt



def min_max_scaling(values):
    min_value = min(values); max_value = max(values)
    return [(value - min_value) / (max_value - min_value) for value in values]


def evaluate_clustering(EC_pairs_cr, method, ecs, labels, num_cluster):
    # Get same_clusters
    same_clusters = []; crs = []
    for ec1, ec2, cr in EC_pairs_cr:
        lable1 = labels[ecs.index(ec1)]
        lable2 = labels[ecs.index(ec2)]
        same_cluster = int(lable1==lable2)
        
        same_clusters.append(same_cluster)
        crs.append(cr)

    # Calculate Spearman correlation coefficient
    corr_pearson, _ = stats.pearsonr(crs, same_clusters)
    corr_spearman, _ = stats.spearmanr(crs, same_clusters)
    #print_out
    if CAS.print_out:
        print(f"Pearson correlation coefficient: {corr_pearson:.3f}")
        print(f"Spearman correlation coefficient: {corr_spearman:.3f}")
        draw_barplot_for_clustering(same_clusters, crs, method, num_cluster)
    
    return same_clusters, crs, corr_pearson, corr_spearman


def draw_barplot_for_clustering(same_clusters, crs, method, num_cluster):
    # Count ec_pairs for same_cluster and crs
    crs_same_cluster = []; crs_diff_cluster = []
    for same_cluster, cr in zip(same_clusters, crs):
        if same_cluster:
            crs_same_cluster.append(cr)
        else:
            crs_diff_cluster.append(cr)
    crs_same_cluster_counter = dict(Counter(crs_same_cluster))
    crs_diff_cluster_counter = dict(Counter(crs_diff_cluster))
        
    crs_labels = [0, 1, 2, 3]
    crs_same_cluster_count = [crs_same_cluster_counter[cr] if cr in crs_same_cluster_counter.keys()
                              else 0 for cr in crs_labels]
    crs_diff_cluster_count = [crs_diff_cluster_counter[cr] if cr in crs_diff_cluster_counter.keys()
                              else 0 [cr] for cr in crs_labels]
    crs_count = [cscc+cdcc for cscc, cdcc in zip(crs_same_cluster_count, crs_diff_cluster_count)]
    crs_same_cluster_freq = [cscc/cc for cscc, cc in zip(crs_same_cluster_count, crs_count)]
    crs_diff_cluster_freq = [cdcc/cc for cdcc, cc in zip(crs_diff_cluster_count, crs_count)]

    x = np.arange(len(crs_labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, crs_same_cluster_freq, width, 
           label='Within same EC cluster', 
           color='goldenrod')
    ax.bar(x + width/2, crs_diff_cluster_freq, width, 
           label='Within differnt EC cluster',
           color='royalblue')

    ax.set_xlabel('Clinical relevance between two ECs')
    ax.set_ylabel('EC pair Frequency')
    ax.set_title(f'{method} (n_cluster: {num_cluster})')
    ax.set_xticks(x)
    ax.set_xticklabels(crs_labels)
    legend = plt.legend()
    legend.set_bbox_to_anchor((-0.1, -0.1))

    plt.show()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return sum_embeddings / sum_mask


def resampling_ecs(ecs_in_EC_pairs_cr, ecs_total):
    ecs = random.sample(ecs_total, 50000)
    #add ecs in EC_pairs_cr in total ecs
    ecs += ecs_in_EC_pairs_cr 
    ecs = list(set(ecs)); random.shuffle(ecs)
    
    return ecs
    
        
def print_out_95CI_corr(crs_dict_list, method, num_clusters):
    results_str = ""
    
    print(f"95% CI for clustering method: {method}")
    results_str += f"95% CI for clustering method: {method}\n"
    
    for num_cluster in num_clusters:
        print(f"Cluster {num_cluster}")
        results_str += f"Cluster {num_cluster}\n"
        
        corrs_spearman = crs_dict_list[method][num_cluster]['corr_spearman']
        corrs_pearson = crs_dict_list[method][num_cluster]['corr_pearson']
        
        print(f"    Spearman corr 95CI: {np.median(corrs_spearman):.3f} [{np.percentile(corrs_spearman, 2.5):.3f}, {np.percentile(corrs_spearman, 97.5):.3f}]")
        results_str += f"    Spearman corr 95CI: {np.median(corrs_spearman):.3f} [{np.percentile(corrs_spearman, 2.5):.3f}, {np.percentile(corrs_spearman, 97.5):.3f}]\n"
        
        print(f"    Pearson corr 95CI: {np.median(corrs_pearson):.3f} [{np.percentile(corrs_pearson, 2.5):.3f}, {np.percentile(corrs_pearson, 97.5):.3f}]")
        results_str += f"    Pearson corr 95CI: {np.median(corrs_pearson):.3f} [{np.percentile(corrs_pearson, 2.5):.3f}, {np.percentile(corrs_pearson, 97.5):.3f}]\n"
        
    return results_str

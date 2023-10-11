
import torch
from transformers import AutoTokenizer



#config
class CFG:
    #dataset path
    ECs_path = "dataset/original_rephrased_EC_pairs"
    ECs_fname = "original_rephrased_EC_pairs_total_50k_list"
    #saving path
    best_model_dir = "best_CReSE_models"
            
    #training hyperparameters
    batch_size = 32
    ec_encoder_lr = 1e-5
    head_lr = 5e-4
    weight_decay = 1e-4
    patience = 1
    factor = 0.8
    
    #training setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    print_within_epoch = False
    print_batches = 2000
    
    #validation setting
    valid_by_loss = False #if False, save best model based on the correlation with clinical relevance data within training epochs - if True, save best model based on the validation losses within training epochs
    corr = "Spearman" #correlation measure for selecting best model
    clinical_relevance_dir = "dataset/clinical_relevance"
    clinical_relevance_fname = "clinical_relevance_gpt35turbo_ECpairs500_fortraining"
    
    #model name
    ec_encoder_model = "michiyasunaga/BioLinkBERT-base" 
    trainable = True # for ec encoder
    #tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ec_encoder_model)
    ec_embedding = title_embedding = 768
    max_length = 256

    # for projection head; used for ec encoder
    temperature = 1.0
    projection_dim = 256 
    dropout = 0.1
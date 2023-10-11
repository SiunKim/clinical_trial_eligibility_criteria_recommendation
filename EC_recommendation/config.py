
from transformers import AutoTokenizer, BertConfig

class MTS:
    do_training = True
    
    #set best_model dir
    dir_best_model = "best_recom_models"

    #set tokenizer for ec-title classification
    BERT_NAME = "michiyasunaga/BioLinkBERT-base" 
    # or "bert-base-uncased", "dmis-lab/biobert-base-cased-v1.2"
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME, do_lower_case=False)
    bert_config = BertConfig.from_pretrained(BERT_NAME)
    
    #training - input settings of EC recommendation model
    input_type = 'only_title'
    # input_type = 'title+summary'
    # input_type = 'title+CTinfo'
    # input_type = 'title+summary+CTinfo'
    pos_neg_ratio = 1 # 1:N ratior for positive, negative ec-title pairs in training dataset
    
    #training hyper-parameters
    BATCH_SIZE = 32
    EPOCHS = 3
    #train data sample size 
    EC_title_pairs_n = 1000000 # set 'use_total_positive' when wanting to use all the positive EC-title data
    #Ent - EC number threshold for generating negative EC-title data
    Ent = 5
    MAXLEN = 256 #maximum token lengths of ec-encoder (up to 512)
    VAL_RATIO = TEST_RATIO = 5.0 #Percentages of validation and test dataset
    num_labels = 2 #binary classification
    #other hyper-parameters
    lr = 1e-5
    eps = 1e-10
    hidden_size = 768*2
    hidden_layer_dim = 512
    dropout = 0.1
    grad_clip = True
    grad_clip_max_norm = 1.0
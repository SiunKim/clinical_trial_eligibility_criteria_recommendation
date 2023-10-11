class MTS(): #model training settings
    #training settings
    model_names = [
    # "distilbert-base-uncased" , 
    "bert-base-uncased", 
    "emilyalsentzer/Bio_ClinicalBERT", 
    "dmis-lab/biobert-base-cased-v1.2", 
    "michiyasunaga/BioLinkBERT-base", 
    "google/electra-base-discriminator", 
    "xlm-roberta-base"
    ]
    MAXLEN = 256 #max token lengths
    TEST_RATIO = 0.1; VALID_RATIO = 0.1
    EPOCHS = 20
    
    #hyper-parameters
    batch_sizes = [16, 32]
    lrs = [5e-5, 2e-5, 5e-6, 1e-6]
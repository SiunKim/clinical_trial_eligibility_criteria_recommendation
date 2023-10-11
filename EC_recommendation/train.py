import os
import random
import pickle
from datetime import datetime 

from tqdm.notebook import tqdm

import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import MTS
from model import EcRecomModel
from dataset import tokenize_ec_title



#set random seed
SEED_VAL = 42
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

#import train EC-title pair data
INPUT_TYPES = ['only_title',
               'title+summary',
               'title+CTinfo',
               'title+summary+CTinfo']
assert MTS.input_type in INPUT_TYPES, f"input_type must be one of {INPUT_TYPES}"
    
    
    
def validation_model(model, device, dataloader_val):
    #model for evaluation
    model.eval()
    #saving validation results
    total_val_loss = 0.0
    val_labels_true = []; val_labels_pred = []
    for batch in dataloader_val:
        #prepare batch
        input_ids_ec, input_ids_title, attention_mask_ec, attention_mask_title, labels = \
            batch
        input_ids_ec = input_ids_ec.to(device)
        input_ids_title = input_ids_title.to(device)
        attention_mask_ec = attention_mask_ec.to(device)
        attention_mask_title = attention_mask_title.to(device)
        labels = labels.to(device)
        #forward pass
        with torch.no_grad():        
            logits, _, _, _, _ = model(input_ids_ec=input_ids_ec.to(device),
                                       input_ids_title=input_ids_title.to(device),
                                       attention_mask_ec=attention_mask_ec.to(device),
                                       attention_mask_title=attention_mask_title.to(device))
            loss = loss_f(logits, labels.to(device))
        #calculate total_valid_loss
        total_val_loss += loss.item()
        #extend labels_true and labels_pred
        labels_true = labels.to('cpu').numpy()
        labels_pred = torch.argmax(logits, dim=1).cpu().numpy()
        val_labels_true.extend(labels_true.tolist())
        val_labels_pred.extend(labels_pred.tolist())
        
    avg_val_loss = total_val_loss / len(dataloader_val)  
    accuracy = accuracy_score(val_labels_true, val_labels_pred)
    precision = precision_score(val_labels_true, val_labels_pred)
    recall = recall_score(val_labels_true, val_labels_pred)
    f1 = f1_score(val_labels_true, val_labels_pred)
    print(f"   Accuracy: {accuracy:.2f}")
    print(f"   Precision: {precision:.2f}")
    print(f"   Recall: {recall:.2f}")
    print(f"   F1 Score: {f1:.2f}")
    print(f"   Validation Loss: {avg_val_loss:.2f}")
    
    return accuracy, precision, recall, f1, avg_val_loss



#training loops - main code
if MTS.do_training:
    #import EC_title data
    dir_data = "data/train"
    #find fname_positive and fname_negative
    fnames_positive = [fname for fname in os.listdir(dir_data) if fname.startswith('etp_positive_train')]
    fnames_negative = [fname for fname in os.listdir(dir_data) if fname.startswith('etp_negative_train')]
    fname_positive = [fname for fname in fnames_positive if MTS.input_type in fname][0]
    fname_negative = [fname for fname in fnames_negative 
                      if (MTS.input_type in fname) & (f"Ent{MTS.Ent}" in fname)][0]
    
    #load EC_title_positive_pairs and EC_title_negative_pairs
    with open(f"{dir_data}/{fname_positive}", 'rb') as f:
        EC_title_positive_pairs = pickle.load(f)    
    with open(f"{dir_data}/{fname_negative}", 'rb') as f:
        EC_title_negative_pairs = pickle.load(f)
    random.shuffle(EC_title_positive_pairs)
    random.shuffle(EC_title_negative_pairs)
    
    #select EC_title_positive_pairs and EC_title_negative_pairs for fixed size
    if MTS.EC_title_pairs_n=='use_total_positive':
        EC_title_positive_pairs_used = EC_title_positive_pairs
    else:
        EC_title_positive_pairs_used = EC_title_positive_pairs[:int(MTS.EC_title_pairs_n/(1+MTS.pos_neg_ratio))]
    if MTS.EC_title_pairs_n=='use_total_positive':
        try:
            EC_title_negative_pairs_used = EC_title_negative_pairs[:len(EC_title_positive_pairs)*MTS.pos_neg_ratio]
        except IndexError:
            print(f"Length of EC_title_negative_pairs ({len(EC_title_negative_pairs)}) is not enough long for the used pos_neg_ratio({MTS.pos_neg_ratio}) and the total length of EC_title_positive_pairs ({len(EC_title_positive_pairs)})!")
            EC_title_negative_pairs_used = EC_title_negative_pairs
    else:
        EC_title_negative_pairs_used = EC_title_negative_pairs[:MTS.EC_title_pairs_n-len(EC_title_positive_pairs_used)]
    EC_title_pairs_total = [(etp, 1) for etp in EC_title_positive_pairs_used] + [(etp, 0) for etp in EC_title_negative_pairs_used]
    random.shuffle(EC_title_pairs_total)
    print(f"Length of EC_title_positive_pairs_used: {len(EC_title_positive_pairs_used)}")
    print(f"Length of EC_title_negative_pairs_used: {len(EC_title_negative_pairs_used)}")
    
    #check length statistics for ECs and titles
    EC_lens = [len(etp[0]) for etp, _ in EC_title_pairs_total]
    title_lens = [len(etp[1]) for etp, _ in EC_title_pairs_total]
    print(f"Maximum length of ecs: {max(EC_lens)}")    
    print(f"Maximum length of titles: {max(title_lens)}")   
    print(f"50 percentiles of ecs: {np.percentile(EC_lens, 50)}")    
    print(f"50 percentiles of titles: {np.percentile(title_lens, 50)}") 
    print(f"95 percentiles of ecs: {np.percentile(EC_lens, 95)}")    
    print(f"95 percentiles of titles: {np.percentile(title_lens, 95)}")  
    print(f"90 percentiles of ecs: {np.percentile(EC_lens, 90)}")    
    print(f"90 percentiles of titles: {np.percentile(title_lens, 90)}")  
    
    MTS.hidden_size = MTS.bert_config.hidden_size*2
    #tokenizing
    print("Start tokenizing ecs and titles!")
    input_ids_ec, input_ids_title, attention_mask_ec, attention_mask_title, labels =\
        tokenize_ec_title(EC_title_pairs_total)
    #dataloader preparation
    dataset = TensorDataset(input_ids_ec, input_ids_title, 
                            attention_mask_ec, attention_mask_title, 
                            labels)
    train_size = int((1 - MTS.VAL_RATIO/100 - MTS.TEST_RATIO/100)*len(dataset))
    val_size = int(MTS.VAL_RATIO/100 *len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    print('{:>5,} test samples'.format(test_size))
    #set DataLoader needs to know our batch size for training, so we specify it 
    dataloader_trn = DataLoader(train_dataset, 
                                sampler = RandomSampler(train_dataset), 
                                batch_size = MTS.BATCH_SIZE)
    dataloader_val = DataLoader(val_dataset,
                                sampler = SequentialSampler(val_dataset),        
                                batch_size = MTS.BATCH_SIZE)
    dataloader_tst = DataLoader(test_dataset,
                                sampler = SequentialSampler(test_dataset),        
                                batch_size = MTS.BATCH_SIZE)

    #set model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model_ec = AutoModel.from_pretrained(MTS.BERT_NAME)
    bert_model_title = AutoModel.from_pretrained(MTS.BERT_NAME)
    model = EcRecomModel(bert_model_ec, bert_model_title)
    model.to(device)

    #set loss, optimizer and scheduler
    loss_f = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),
                    lr=MTS.lr,
                    eps=MTS.lr
                    )
    total_steps = len(dataloader_trn) * MTS.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    #set model_name and check the model_name is already in dir_best_model
    time_str = datetime.now().strftime("%m%d")
    for i in range(100):
        model_name = f"ec_title_best_model_{MTS.BERT_NAME.replace('/', '_')}_input_type{MTS.input_type}_samplen{MTS.EC_title_pairs_n}_lr_{MTS.lr}_posnegratio{MTS.pos_neg_ratio}_Ent{MTS.Ent}_{time_str}_{i+1}"
        if model_name + '.pt' not in os.listdir(MTS.dir_best_model):
            break
    
    #training loop
    print("Start training EC-title recommendation model!")
    best_avg_val_loss = np.inf
    train_losses = []
    val_losses = []
    for epoch_i in range(0, MTS.EPOCHS):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, MTS.EPOCHS))
        print('Training...')

        total_train_loss = 0
        for step, batch in enumerate(tqdm(dataloader_trn)):
            optimizer.zero_grad()
            #prepare batch
            input_ids_ec, input_ids_title, attention_mask_ec, attention_mask_title, labels = \
                batch
            #forward pass
            logits, _, _, _, _ = model(input_ids_ec=input_ids_ec.to(device),
                                       input_ids_title=input_ids_title.to(device),
                                       attention_mask_ec=attention_mask_ec.to(device),
                                       attention_mask_title=attention_mask_title.to(device))
            loss = loss_f(logits, labels.to(device))
            #backward pass and optimization
            loss.backward()
            optimizer.step()
            if MTS.grad_clip:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            scheduler.step()
            #calculate total_train_loss
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(dataloader_trn)    
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
        #model validation
        print("")
        print("Running Validation...")
        accuracy, precision, recall, f1, avg_val_loss = validation_model(model, device, dataloader_val)
        #save train, val_losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        #check best model and save it 
        if avg_val_loss <= best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            training_settings = {
                "EC_title_pairs_n": MTS.EC_title_pairs_n,
                "pos_neg_ratio": MTS.pos_neg_ratio,
                "hidden_layer_dim": MTS.hidden_layer_dim,
                "hidden_size": MTS.hidden_size,
                "dropout": MTS.dropout,
                "BATCH_SIZE": MTS.BATCH_SIZE,
                "MAXLEN": MTS.MAXLEN,
                "epoch": epoch_i + 1,
                "lr": MTS.lr,
                "eps": MTS.eps,
                "grad_clip": MTS.grad_clip,
                "grad_clip_max_norm": MTS.grad_clip_max_norm,
                "avg_val_loss": avg_val_loss,
                "accuracy-val": accuracy,
                "precision-val": precision,
                "recall-val": recall,
                "f1-val": f1,
                }
            torch.save(model, f"{MTS.dir_best_model}/{model_name}.pt")
            print("Saved Best Model!")
    print("")
    print("Evaluate test performance!")
    
    #get test performances and update training_settings
    accuracy, precision, recall, f1, avg_tst_loss = validation_model(model, device, dataloader_tst)
    training_settings['avg_tst_loss'] = avg_val_loss
    training_settings['accuracy-tst'] = accuracy
    training_settings['precision-tst'] = precision
    training_settings['recall-tst'] = recall
    training_settings['f1-tst'] = f1
    with open(f"{MTS.dir_best_model}/{model_name}_training_settings.txt", 'w') as f:
        f.write(str(training_settings))
    
    # Plot the loss curves
    import matplotlib.pyplot as plt
    plt.plot(range(1, MTS.EPOCHS+1), train_losses, label='Training Loss')
    plt.plot(range(1, MTS.EPOCHS+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    print("Training complete!")
    
import time
import random

import pandas as pd
import numpy as np

from collections import defaultdict

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AutoTokenizer    
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
    
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from config import MTS
from utils import format_time
from dataset import set_datasets



    
#set random seed
SEED_VAL = 42
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)
    
#set model and train loops for classifying common ECs
def train_loop_for_classifying_common_ECs(train_dataset, val_dataset, BERT_NAME, device, LR,
                                          print_while_training=True):
    train_dataloader = DataLoader(train_dataset, 
                                sampler = RandomSampler(train_dataset), 
                                batch_size = BATCH_SIZE)
    validation_dataloader = DataLoader(val_dataset,
                                    sampler = SequentialSampler(val_dataset),           
                                    batch_size = BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(
        BERT_NAME,
        num_labels = 2, 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    model.to(device)
    optimizer = AdamW(model.parameters(),
                    lr = LR, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    #training loop
    training_stats = []; total_t0 = time.time()
    best_f1 = 0.0; training_stats_best = defaultdict(str)
    for epoch_i in range(0, EPOCHS):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0

        model.train()
        preds_total_trn = []; labels_ids_total_trn = []
        for batch in train_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()        
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            
            loss = outputs[0]
            total_train_loss += loss.item()
            loss.backward()
            
            logits = outputs[1].detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            label_ids = b_labels.to('cpu').numpy()
            preds_total_trn += list(preds); labels_ids_total_trn += list(label_ids)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)    
        training_time = format_time(time.time() - t0)

        print("  Average training loss: {0:.4f}".format(avg_train_loss))
        if print_while_training:
            if (epoch_i+1)%5==0:
                print("")
                print("  accuracy_score: {0:.4f}".format(accuracy_score(labels_ids_total_trn, preds_total_trn)))
                print("  recall_score: {0:.4f}".format(recall_score(labels_ids_total_trn, preds_total_trn)))
                print("  precision_score: {0:.4f}".format(precision_score(labels_ids_total_trn, preds_total_trn)))
                print("  f1_score: {0:.4f}".format(f1_score(labels_ids_total_trn, preds_total_trn)))
            
        #evaluatoin!
        print("")
        print("Running Validation...")
        model.eval()
        _, _, avg_val_loss, best_f1, best_scores_valid = \
            valid_test_loop(model=model, 
                            valid_test_dataloader=validation_dataloader, 
                            device=device,
                            best_f1=best_f1)
            
        # Record all statistics from this epoch.
        training_stats_i = {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'accuracy': best_scores_valid['accuracy'],
                'recall': best_scores_valid['recall'],
                'precision': best_scores_valid['precision'],
                'f1': best_scores_valid['f1'],
                'Training Time': training_time
            }
        training_stats.append(training_stats_i)
        f1_best = max([ts['f1'] for ts in training_stats])
        if f1_best <= training_stats_i['f1']:
            training_stats_best = training_stats_i

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
    #set model_name
    model_name = f"best_model{BERT_NAME.replace('/', '_')}_0509_bs_{BATCH_SIZE}_lr{LR}.pt"
    
    return model_name, training_stats_best
    
    
def valid_test_loop(model, valid_test_dataloader, device, best_f1, mode='valid'):
    total_eval_loss = 0
    preds_total = []; labels_ids_total = []
    best_model_name = None; best_scores = defaultdict(float)
    for batch in valid_test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            
        total_eval_loss += outputs[0].item()
        
        logits = outputs[1].detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        label_ids = b_labels.to('cpu').numpy()
        preds_total += list(preds); labels_ids_total += list(label_ids)
        
    avg_val_loss = total_eval_loss / len(valid_test_dataloader)  
    f1_current = f1_score(labels_ids_total, preds_total)  
    
    if best_f1 <= f1_current:
        if mode=='valid':
            best_f1 = f1_current
            best_model_dir = 'best_common_EC_cls_models/'
            best_model_name = f"best_model{BERT_NAME.replace('/', '_')}_0509_bs_{BATCH_SIZE}_lr{LR}.pt"
            torch.save(model, best_model_dir + best_model_name)
            print("Saved Best Model!")
            
        best_scores = {'accuracy': accuracy_score(labels_ids_total, preds_total),
                        'recall': recall_score(labels_ids_total, preds_total),
                        'precision': precision_score(labels_ids_total, preds_total),
                        'f1': f1_score(labels_ids_total, preds_total)}
            
        print("  accuracy_score: {0:.4f}".format(best_scores['accuracy']))
        print("  recall_score: {0:.4f}".format(best_scores['recall']))
        print("  precision_score: {0:.4f}".format(best_scores['precision']))
        print("  f1_score: {0:.4f}".format(best_scores['f1']))
        
    if mode=='valid':
        print("  Validation Loss: {0:.4f}".format(avg_val_loss))

    return labels_ids_total, preds_total, avg_val_loss, best_f1, best_scores



#main code
for BERT_NAME in MTS.model_names:
    print(f"BERT_NAME is {BERT_NAME}!")
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME, do_lower_case=True)
    ann_DF = pd.read_csv('annotated_common_EC.csv', encoding= 'unicode_escape')
    if 'CT_titles' in ann_DF.keys():
        ann_DF.rename(columns={'CT_titles': 'titles', 'ECs': 'ecs', 'common': 'drop'}, inplace=True); ann_DF
    ann_DF['drop'] = [0 if np.isnan(d) else int(d) for d in ann_DF['drop']]
    ann_DF = ann_DF.sample(frac=1, random_state=SEED_VAL); ann_DF.reset_index(inplace=True, drop=True)

    #get datasets for training 
    dataset = [{"label":ann_DF['drop'][i],
                "text": ann_DF['ecs'][i] + " [SEP] " + ann_DF['titles'][i]} for i in range(len(ann_DF))]
    print(f'Common ECs count: {len([d for d in dataset if d["label"]==1])} ({len([d for d in dataset if d["label"]==1])/len(dataset)*100:.1f}%)')

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    model_names = []; training_stats_best_total = []
    for BATCH_SIZE in MTS.batch_sizes:
        #set datasets 
        train_dataset, val_dataset, test_dataset = set_datasets(dataset=dataset, tokenizer=tokenizer)
        for LR in MTS.lrs:
            if LR<5e-6:
                EPOCHS = 35
            elif LR<2e-5:
                EPOCHS = 30
            elif LR<5e-5:
                EPOCHS = 20
            else:          
                EPOCHS = 15
            #train loop
            model_name, training_stats_best = \
                train_loop_for_classifying_common_ECs(
                    train_dataset=train_dataset, 
                    val_dataset=val_dataset,
                    BERT_NAME=BERT_NAME, 
                    device=device,
                    LR=LR,
                    print_while_training=False
                    )
            model_names.append(model_name)
            training_stats_best_total.append(training_stats_best)
            
    f1_best = max([tsb['f1'] for tsb in training_stats_best_total])
    tsb_best = [tsb for tsb in training_stats_best_total if tsb['f1']==f1_best][0]
    best_model_name = model_names[training_stats_best_total.index(tsb_best)]

    print(f"Best model! {best_model_name} - test performance")
    print(f"Epoch - {training_stats_best['epoch']}")
    
    best_model_dir = 'C:/Users/Admin/Desktop/classify_common_EC/'
    test_dataloader = DataLoader(test_dataset,
                                 sampler = SequentialSampler(test_dataset),
                                 batch_size = BATCH_SIZE)
    best_model = torch.load(best_model_dir + best_model_name, 
                            map_location=device)
    _, _,  _, _, best_scores_test = valid_test_loop(
        model=best_model, 
        valid_test_dataloader=test_dataloader, 
        device=device,
        best_f1=0.0,
        mode='test')
    
    #save best_model_performance
    print(f"save model - Best_performance_{best_model_name.replace('.pt', '')}")
    print("best_scores_test: " + str(best_scores_test) + 
            "\nbest_scores_valid: " + str(tsb_best))
    with open(best_model_dir + f"Best_performance_{best_model_name.replace('.pt', '')}.txt", "w") as f:
        f.write("best_scores_test" + str(best_scores_test) + "\nbest_scores_valid" + str(tsb_best))
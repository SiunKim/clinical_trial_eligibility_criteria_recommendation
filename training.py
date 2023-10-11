import pickle
from tqdm import tqdm
from datetime import datetime

import pandas as pd 
import numpy as np

import torch
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split 
from scipy.stats import spearmanr
    
from config import CFG
from model import CReSE
from dataset import CReSEDataset

from utils import AvgMeter, get_match_scores, min_max_scaling
from utils import draw_scatter_plot_mss_crs, draw_epoch_graph
    


#functions for preparing CRSE model and training the model
def set_CReSE_and_tokenizer(model_path, tokenizer_name):
    model = torch.load(model_path, map_location=CFG.device); model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    return model, tokenizer
        
        
def make_train_valid_dfs():
    with open(f"{CFG.ECs_path}/{CFG.ECs_fname}.p", 'rb') as f:
        ECs_total = pickle.load(f)
        
    ECs_train, ECs_valid = train_test_split(ECs_total, test_size=0.1)
        
    train_dataframe = pd.DataFrame({'ecs1': [ecs[0] for ecs in ECs_train],
                                    'ecs2': [ecs[1] for ecs in ECs_train]})
    valid_dataframe = pd.DataFrame({'ecs1': [ecs[0] for ecs in ECs_valid],
                                    'ecs2': [ecs[1] for ecs in ECs_valid]})
    
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, mode):
    dataset = CReSEDataset(
        dataframe["ecs1"].values,
        dataframe["ecs2"].values
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True if mode == "train" else False
    )

    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step,
                print_within_epoch=True,
                print_batches=1000,
                valid_loader=False,
                ECs_cr_triplets=False):
    loss_meter = AvgMeter()
    for i, batch in enumerate(tqdm(train_loader)):
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k not in ['ecs1', 'ecs2']}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["input_ids_ecs1"].size(0)
        loss_meter.update(loss.item(), count)
        
        if print_within_epoch:
            if i%print_batches==0:
                print(f'train_loss_meter (batch {i}): {str(loss_meter).split()[1]}')
                if valid_loader:
                    with torch.no_grad():
                        valid_loss = valid_epoch(model, valid_loader)
                        print(f"valid_loss_meter (batch {i}): {str(valid_loss).split()[1]}")
                if ECs_cr_triplets:
                    _ = valid_by_clinical_relevance(model, ECs_cr_triplets)
                    
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    for batch in valid_loader:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k not in ['ecs1', 'ecs2']}
        loss = model(batch)
        count = batch["input_ids_ecs1"].size(0)
        loss_meter.update(loss.item(), count)

    return loss_meter

    
def valid_by_clinical_relevance(model, ECs_cr_triplets):
    #get mss and crs
    match_scores = get_match_scores(model, ECs_cr_triplets)
    #min-max scaling for match_scores
    match_scores = min_max_scaling(match_scores)
    crs_plus1 = [ect[2] + 1 for ect in ECs_cr_triplets]
    
    #measure Pearson correlation coefficient
    corr_pearson = np.corrcoef(match_scores, crs_plus1)[0, 1]
    #measure Spearman rank correlation coefficient
    corr_spearman, _ = spearmanr(match_scores, crs_plus1)
    print(f"Pearson correlation: {corr_pearson:.3f}")
    print(f"Spearman rank correlation: {corr_spearman:.3f}")
    # print(f"Match scores:  mean - {np.mean(match_scores):.3f}, lower 25% - {np.percentile(match_scores, 25):.3f}, minimum - {min(match_scores):.3f}, ")
    
    #draw scatter plot
    draw_scatter_plot_mss_crs(match_scores, crs_plus1)
    # draw_scatter_plot_mss_crs2(match_scores, crs_plus1)
    
    return match_scores, corr_pearson, corr_spearman
            

def training_loop(model, 
                  train_loader, 
                  valid_loader, 
                  ECs_cr_triplets, 
                  step="epoch",
                  print_within_epoch=False,
                  print_batches=2000):
    #set optimizer and lr_scheduler
    params = [
        {"params": model.ec_encoder.parameters(), 
         "lr": CFG.ec_encoder_lr},
        {"params": model.ec_projection.parameters(), 
         "lr": CFG.head_lr, 
         "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=CFG.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    #validate by clinical relevance - before start training
    valid_by_clinical_relevance(model, ECs_cr_triplets)
    #training loop
    best_loss = float('inf') 
    best_corr = 0; val_loss_at_best_corr = 0.0
    best_model_setting = {}
    #for epoch plot
    train_loss_total = []; val_loss_total = []
    prs_corr_total = []; spr_corr_total = []
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        #train with print
        train_loss = train_epoch(model=model, 
                                 train_loader=train_loader, 
                                 optimizer=optimizer, 
                                 lr_scheduler=lr_scheduler, 
                                 step=step,
                                 print_within_epoch=print_within_epoch,
                                 print_batches=print_batches,
                                 valid_loader=valid_loader,
                                 ECs_cr_triplets=ECs_cr_triplets)
        print(f"train_loss: {train_loss}")
    
        #model validation
        model.eval()
        #calculate valid loss
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            print(f"valid_loss: {valid_loss}")
        lr_scheduler.step(valid_loss.avg)
        #validate by clinical relevance
        match_scores, corr_pearson, corr_spearman = \
            valid_by_clinical_relevance(model, ECs_cr_triplets)
        if CFG.corr == "Pearson":
            corr = corr_pearson
        else:
            corr = corr_spearman 
        
        #for epoch plot
        train_loss_total.append(train_loss.avg); val_loss_total.append(valid_loss.avg)
        prs_corr_total.append(corr_pearson)
        spr_corr_total.append(corr_spearman)
        
        #save best model
        today = datetime.today().strftime("%d%m%y") 
        if CFG.valid_by_loss:
            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                is_best = True
            else:
                is_best = False
        else:
            if best_corr <= corr :
                best_corr = corr
                val_loss_at_best_corr = valid_loss.avg
                is_best = True
            else:
                is_best = False
                
        if is_best:
            best_model_setting = {
                "projection_dim": CFG.projection_dim,
                "batch_size" : CFG.batch_size,
                "head_lr" : CFG.head_lr,
                "ec_encoder_lr" : CFG.ec_encoder_lr,
                "weight_decay" : CFG.weight_decay,
                "patience" : CFG.patience,
                "factor" : CFG.factor,
                "epoch": epoch+1,
                "train_loss": train_loss.avg,
                "valid_loss": valid_loss.avg,
                "corr_pearson": corr_pearson,
                "corr_spearman": corr_spearman,
                "valid_by_loss": CFG.valid_by_loss
                }
            best_model_name = f"best_model_valbyloss{CFG.valid_by_loss}_projdim{CFG.projection_dim}_{CFG.ec_encoder_model.replace('/', '_')}_bs{CFG.batch_size}_hlr{CFG.head_lr}_elr{CFG.ec_encoder_lr}_wd{CFG.weight_decay}_{today}.pt"
            torch.save(model, f'{CFG.best_model_dir}/{best_model_name}.pt')
            with open(f'{CFG.best_model_dir}/{best_model_name}_train_setting.txt', "w") as f:
                f.write(str(best_model_setting))
            print(f"Saved Best Model! - epoch {epoch+1}")
        
    #draw epoch plot
    draw_epoch_graph(train_loss_total, val_loss_total, 'train_loss_total', 'val_loss_total')
    draw_epoch_graph(prs_corr_total, [0]*len(prs_corr_total), 'prs_corr_total', 'dump')
    draw_epoch_graph(spr_corr_total, [0]*len(spr_corr_total), 'spr_corr_total', 'dump')
    # draw_epoch_graph(ms_mean_total, ms_min_total, 'ms_mean_total', 'ms_min_total')
    
    return model, best_corr, val_loss_at_best_corr, best_model_setting, train_loss_total, val_loss_total, prs_corr_total, spr_corr_total
       
    
    
#main code
def main():
    #import training dataset and ECs-clinical relevance data (model selection within epochs)
    CFG.ECs_fname = "ECs_rephrased_50k_list_total_0522"
    train_df, valid_df = make_train_valid_dfs()
    with open(f"{CFG.clinical_relevance_dir}/{CFG.clinical_relevance_fname}.p", "rb") as f:
        ECs_cr_triplets = pickle.load(f)
    
    #set dataloader
    train_loader = build_loaders(train_df, mode="train")
    valid_loader = build_loaders(valid_df, mode="valid")
    
    #training loop        
    print(f"Start training loop for --\n projection dim: {CFG.projection_dim}\n batch size: {CFG.batch_size}\n encoder lr: {CFG.ec_encoder_lr}\n head lr: {CFG.head_lr}\n weight decay: {CFG.weight_decay}")
        
    #set model and training environments
    model = CReSE().to(CFG.device)
    #training loop
    model, best_corr, val_loss_at_best_corr, best_model_setting, train_loss_total, val_loss_total, prs_corr_total, spr_corr_total = \
        training_loop(model, 
                      train_loader, 
                      valid_loader, 
                      ECs_cr_triplets, 
                      step="epoch",
                      print_within_epoch=False,
                      print_batches=2000)
        
    print(f"Best corr - {best_corr}")
    print("End trainign loop!")
    print("==="*20)
        
import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel

from config import CFG
from dataset import CReSEDataset



#Ec, Title encoder (dim=768)
class EcEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(CFG.ec_encoder_model)
                
        for p in self.model.parameters():
            p.requires_grad = CFG.trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    
    
#projection head
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    

#utilities
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    

class CReSE(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        ec_embedding=CFG.ec_embedding
    ):
        super().__init__()
        self.ec_encoder = EcEncoder()
        self.ec_projection = ProjectionHead(embedding_dim=ec_embedding)
        self.temperature = temperature
    

    def forward(self, batch):
        # Getting two ec Features
        ecs1_features = self.ec_encoder(
            input_ids=batch["input_ids_ecs1"], attention_mask=batch["attention_mask_ecs1"]
        )
        ecs2_features = self.ec_encoder(
            input_ids=batch["input_ids_ecs2"], attention_mask=batch["attention_mask_ecs2"]
        )
        # Getting ec Embeddings (with same dimension)
        ecs1_embeddings = self.ec_projection(ecs1_features)
        ecs2_embeddings = self.ec_projection(ecs2_features)

        # Calculating the Loss
        logits = (ecs2_embeddings @ ecs1_embeddings.T) / self.temperature
        ecs1_similarity = ecs1_embeddings @ ecs1_embeddings.T
        ecs2_similarity = ecs2_embeddings @ ecs2_embeddings.T
        targets = F.softmax((ecs1_similarity + ecs2_similarity) / 2 * self.temperature, dim=-1)
        titles_loss = cross_entropy(logits, targets, reduction='none')
        ecs_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (ecs_loss + titles_loss) / 2.0 # shape: (batch_size)
        
        return loss.mean()
    
    def get_ec_embeddings(self, ecs):
        if type(ecs)==str:
            ecs = [ecs]
        #set dataloader for model
        dataset = CReSEDataset(ecs, [""]*len(ecs))
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=CFG.batch_size,
                                                 shuffle=False)
        
        ecs_embeddings_total = []
        with torch.no_grad():
            for batch in dataloader:
                ecs_features = self.ec_encoder(
                    input_ids=batch["input_ids_ecs1"].to(CFG.device), 
                    attention_mask=batch["attention_mask_ecs1"].to(CFG.device)
                )
                ecs_embeddings = self.ec_projection(ecs_features)
                ecs_embeddings_total.append(ecs_embeddings)
        
        return torch.cat(ecs_embeddings_total)

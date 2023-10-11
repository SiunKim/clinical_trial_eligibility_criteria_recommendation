import torch
import torch.nn as nn

from config import MTS


class EcRecomModel(nn.Module):
    def __init__(self, model_ec, model_title):
        super().__init__()
        self.model_ec = model_ec
        self.model_title = model_title
        self.num_labels = MTS.num_labels
        self.hidden_size = MTS.hidden_size
        self.hidden_layer_dim = MTS.hidden_layer_dim
        self.dropout_prob = MTS.dropout
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_layer_dim, self.num_labels),
        )
        
    def forward(self,
                input_ids_ec=None,
                input_ids_title=None,
                attention_mask_ec=None,
                attention_mask_title=None
                ):
        output_ec = self.model_ec(input_ids=input_ids_ec,      
                                  attention_mask=attention_mask_ec)
        output_title = self.model_title(input_ids=input_ids_title,
                                        attention_mask=attention_mask_title)
        cls_hidden_state_ec = output_ec.last_hidden_state[:, 0, :]
        cls_hidden_state_title = output_title.last_hidden_state[:, 0, :]
        
        #concatenate hidden state of ecs and titles
        cls_hidden_state_ec_title = torch.cat((cls_hidden_state_ec, cls_hidden_state_title),
                                              dim=1)
        #classify ec-title pair
        logits = self.linear_relu_stack(cls_hidden_state_ec_title)
        
        return logits, cls_hidden_state_ec, cls_hidden_state_title, output_ec.attentions, output_title.attentions
    
    

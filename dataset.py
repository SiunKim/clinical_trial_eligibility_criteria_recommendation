import torch 

from config import CFG



#datset
class CReSEDataset(torch.utils.data.Dataset):
    def __init__(self, ecs1, ecs2):
        self.ecs1 = list(ecs1)
        self.ecs2 = list(ecs2)
        self.encoded_ecs1 = CFG.tokenizer(
            list(ecs1), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.encoded_ecs2 = CFG.tokenizer(
            list(ecs2), padding=True, truncation=True, max_length=CFG.max_length
        )
        
    def __getitem__(self, idx):
        item = {}
        for key, values in self.encoded_ecs1.items():
            item[key + '_ecs1'] =  torch.tensor(values[idx])
        for key, values in self.encoded_ecs2.items():
            item[key + '_ecs2'] =  torch.tensor(values[idx])
            
        item['ecs1'] = self.ecs1[idx]
        item['ecs2'] = self.ecs2[idx]

        return item

    def __len__(self):
        return len(self.ecs1)

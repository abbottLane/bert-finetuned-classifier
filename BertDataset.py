import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

class BertDataset(Dataset):
    def __init__(self, data, label_map, max_seq_length):
        self.data=pd.DataFrame(data,columns=['target','text'])
        self.texts = self.data['text']
        self.labels = self.data['target']
        self.label_map = self.invert_label_map(label_map)
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def invert_label_map(self, label_map):
        return {v: k for k, v in label_map.items()}
    
def get_dataloader(data, label_map, batch_size=32, max_seq_length=128):
    dataset = BertDataset(data, label_map, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm



class ArgumentDataset(Dataset):
    def __init__(self, arguments, scores, tokenizer, max_length=512):
        self.arguments = arguments
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.arguments)

    def __getitem__(self, idx):
        argument = self.arguments[idx]
        score = self.scores[idx]
        encoding = self.tokenizer(argument, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)
        return {**{key: torch.squeeze(val, dim=0) for key, val in encoding.items()}, 'label': torch.tensor(score)}



def load_dataset(file_path):
    df = pd.read_csv(file_path)
    arguments = df.iloc[:, 0].tolist()
    scores = [int(score) for score in df.iloc[:, 1].tolist()]
    return arguments, scores

#file_path = 'BINARY_TRAIN_QUAL_FORM_RAW.csv'
#print("Loading dataset...")
#arguments, scores = load_dataset(file_path)
#tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

#train_args, val_args, train_scores, val_scores = train_test_split(arguments, scores, test_size=0.1, random_state=42)

#train_dataset = ArgumentDataset(train_args, train_scores, tokenizer)
#val_dataset = ArgumentDataset(val_args, val_scores, tokenizer)

#train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

#base_model = RobertaForSequenceClassification.from_pretrained('roberta-base')


class ModelQualityPredictor(nn.Module):
    def __init__(self, base_model):
        super(ModelQualityPredictor, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(base_model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        outputs = self.base_model.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

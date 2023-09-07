import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import random
import nltk
from torch.utils.data import Dataset
import os



class GIDSDataset(Dataset):
    def __init__(self, data_file):
        self.data = self._load_data(data_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        Sentences = sample['Sentences']
        label = sample['label']
        label = 0 if label == 'NA' else 1
        entity_pair = sample['entity_pair']
        
        return Sentences, label , entity_pair
    
    def _load_data(self, data_file):
        data = []
        with open(data_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                label = parts[4]
                Sentences = parts[5] 
                entity1 = parts[2]
                entity2 = parts[3]
                entity_pair = (entity1, entity2)
                data.append({'Sentences': Sentences, 'label': label , 'entity_pair': entity_pair})
        return data



train_data_file = os.path.join(
    os.getcwd(), "data/train.tsv"
)  # '/Users/samira/Desktop/ErrorDetection/F_N_Detection/data/GIDS/train.tsv'
test_data_file = os.path.join(
    os.getcwd(), "data/dev.tsv"
)  # '/Users/samira/Desktop/ErrorDetection/F_N_Detection/data/GIDS/test.tsv'


train_dataset = GIDSDataset(train_data_file)
test_dataset = GIDSDataset(test_data_file)

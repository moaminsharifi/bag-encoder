from torch.utils.data import Dataset
import os
import json
import torch
import numpy as np




class NYTDatasetGIDStyle(Dataset):
    def __init__(self, data_file):
        self.data = self._load_data(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sentence = sample["Sentences"]
        label = 0 if sample["label"] == 'NA' else 1
        #entity_pair = sample["entity_pair"]
        return sentence, label

    def _load_data(self, data_file):
        data = []
        with open(data_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                parts = json.loads(line)
                label = parts.get("relation")
                entity1 = parts.get("h").get("name")
                entity2 = parts.get("t").get("name")
                entity_pair = (entity1, entity2)
                Sentences = parts.get("text")
                data.append(
                    {
                        "Sentences": Sentences,
                        "label": label, 
                        #"entity_pair": entity_pair,
                    }
                )
        return data


train_data_file = os.path.join(
    os.getcwd(), "data/nyt10_train.txt"
)  # '/Users/samira/Desktop/ErrorDetection/F_N_Detection/data/GIDS/train.tsv'
test_data_file = os.path.join(
    os.getcwd(), "data/Nyt10Test.txt"
)  # '/Users/samira/Desktop/ErrorDetection/F_N_Detection/data/GIDS/test.tsv'

tn_data_file = os.path.join(
    os.getcwd(), "data/TrueNegativeNYT10.json"
)  # '/Users/samira/Desktop/ErrorDetection/F_N_Detection/data/GIDS/test.tsv'

train_dataset = NYTDatasetGIDStyle(train_data_file)
test_dataset = NYTDatasetGIDStyle(test_data_file)

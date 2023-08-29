import os
import torch
from torch.utils.data import Dataset
# Assume these are all your unique labels
labels = ['NA' , '/people/deceased_person/place_of_death', '/people/person/place_of_birth', '/people/person/education./education/education/institution', '/people/person/education./education/education/degree']

# Create a dictionary that maps each label to a unique integer
label_to_int = {label: (0 if label == 'NA' else 1) for label in labels}


# GIDSDataset class
class GIDSDataset(Dataset):
    def __init__(self, data_file):
        self.data = self._load_data(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        Sentences = sample["Sentences"]
        label = sample["label"]
        label = label_to_int[label]

        return Sentences, label
    def _load_data(self, data_file):
        data = []
        with open(data_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split("\t")
                label = parts[4]  # You may need to adjust this to match the structure of your data
                Sentences = parts[5]  # You may need to adjust this to match the structure of your data
                data.append({"Sentences": Sentences, "label": label})
        return data



train_data_file = os.path.join(
    os.getcwd(), "train.tsv"
)  # '/Users/samira/Desktop/ErrorDetection/F_N_Detection/data/GIDS/train.tsv'
test_data_file = os.path.join(
    os.getcwd(), "test.tsv"
)  # '/Users/samira/Desktop/ErrorDetection/F_N_Detection/data/GIDS/test.tsv'
val_data_file = os.path.join(os.getcwd(), "dev.tsv")

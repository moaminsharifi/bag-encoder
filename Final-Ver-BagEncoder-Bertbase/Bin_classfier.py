import torch
from torch import nn


import torch.nn.functional as f

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, dropout_prob=0.5):
        super(MLPClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(hidden_size3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = f.normalize(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        #out = self.sigmoid(out) # apply sigmoid function
        return out





def dynamic_batches(data, max_batch_size):
    sentences_batch = []
    labels_batch = []
    entity_pairs_batch = []
    for sentences, labels, entity_pairs in data:
        sentences_batch.extend(sentences)
        labels_batch.extend(labels)
        entity_pairs_batch.extend(entity_pairs)
        if len(sentences_batch) >= max_batch_size:
            yield sentences_batch[:max_batch_size], labels_batch[:max_batch_size], entity_pairs_batch[:max_batch_size]
            sentences_batch = sentences_batch[max_batch_size:]
            labels_batch = labels_batch[max_batch_size:]
            entity_pairs_batch = entity_pairs_batch[max_batch_size:]
    if sentences_batch:
        yield sentences_batch, labels_batch, entity_pairs_batch

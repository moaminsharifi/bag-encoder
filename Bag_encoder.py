from collections import defaultdict

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Create a DataLoader
# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
# print(train_dataset[0])
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# batch_size = 32


class BagEncoder:
    def __init__(self, pretrained_model_name, batch_size):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.batch_size = batch_size
        self.bert_model = BertModel.from_pretrained(pretrained_model_name)
        self.input_dim = self.bert_model.config.hidden_size
        self.attention = nn.Sequential(
            nn.Linear(self.bert_model.config.hidden_size, 1),
            nn.Tanh(),
            nn.Softmax(dim=1)
        ).to(device)

    def encode_sentences(self, sentences, labels, entity_pairs):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = self.bert_model(**inputs)
        sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)
        data = self.create_bags(sentence_embedding, labels, entity_pairs)
        return data

    def create_bags(self, sentence_embeddings, labels, entity_pairs):
        data = defaultdict(lambda: defaultdict(list))
        for sentence_embedding, label, entity_pair_0, entity_pair_1, in zip(sentence_embeddings,
                                                                            labels,
                                                                            entity_pairs[0],
                                                                            entity_pairs[1]):

            if entity_pair_0 >= entity_pair_1:
                entity_pair_1, entity_pair_0 = entity_pair_0, entity_pair_1
            data[(entity_pair_0, entity_pair_1)]['embeddings'].append(sentence_embedding)
            data[(entity_pair_0, entity_pair_1)]['labels'].append(label)
            data[(entity_pair_0, entity_pair_1)]['entity_pairs'].append((entity_pair_0, entity_pair_1))

        for pair_key in data:
            bag = torch.stack(data[pair_key]['embeddings']).to(device)
            attention_weights = self.attention(bag)  # shape: (seq_len, 1)
            attention_weights = torch.softmax(attention_weights, dim=0)  # take softmax over seq_len dimension
            attention_applied = (bag * attention_weights)  # shape: (hidden_dim,)
            # attention_weights_1D = attention_weights.view(-1)  # This reshapes it into a 1D tensor.
            # corr = torch.outer(attention_weights_1D, attention_weights_1D)

            # corr = torch.outer(attention_weights, attention_weights)  # shape: (seq_len, seq_len)
            data[pair_key]['bag'] = bag
            data[pair_key]['relation representation'] = attention_applied

            # data[corr]['Instant Corelation'] = corr
        return data

# bag_encoder = BagEncoder(pretrained_model_name="bert-base-uncased", batch_size=batch_size)

# for i, (sentences, labels, entity_pairs) in enumerate(train_dataloader):
#    data = bag_encoder.encode_sentences(sentences, labels, entity_pairs)
#    print(f'Batch {i+1}:')
#    for i, pair in enumerate(data.keys()):
#        print(f"Token: {pair}")
#        print(f"Entity pairs: {data[pair]['entity_pairs']}")
#        print(f"Label: {data[pair]['labels']}")
#        print(f"Bag: {data[pair]['bag']}")
#        print(f"Relation representation: {data[pair]['relation representation']}")
#        print(f"Correlation: {data[pair]['relation representation']}")
#        print('-----------------------------')
#    break

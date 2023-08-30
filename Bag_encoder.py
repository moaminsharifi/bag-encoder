class BagEncoder:
    def __init__(self, pretrained_model_name, batch_size):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.batch_size = batch_size
        self.bert_model = BertModel.from_pretrained(pretrained_model_name).to(device)
        self.input_dim = self.bert_model.config.hidden_size
        self.attention = nn.Sequential(
            nn.Linear(self.bert_model.config.hidden_size, 1),
            nn.Tanh(),
            nn.Softmax(dim=1)
        ).to(device)

    def encode_sentences(self, sentences, labels, entity_pairs):
        entity_pair_keys = [(min(ep0, ep1), max(ep0, ep1)) for ep0, ep1 in zip(entity_pairs[0], entity_pairs[1])]
        grouped_data = defaultdict(lambda: {"sentences": [], "labels": [], "entity_pairs": []})

        for sentence, label, key in zip(sentences, labels, entity_pair_keys):
            grouped_data[key]['sentences'].append(sentence)
            grouped_data[key]['labels'].append(label)
            grouped_data[key]['entity_pairs'].append(key)

        data = {}
        for key, group in grouped_data.items():
            data[key] = self.process_group(group['sentences'], group['labels'], group['entity_pairs'])
        
        return data

    def process_group(self, group_sentences, group_labels, group_entity_pairs):
        num_samples = len(group_sentences)
        data = defaultdict(lambda: defaultdict(list))

        for i in range(0, num_samples, self.batch_size):
            batch_sentences = group_sentences[i:i+self.batch_size]
            batch_labels = group_labels[i:i+self.batch_size]
            batch_entity_pairs = group_entity_pairs[i:i+self.batch_size]

            inputs = self.tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
            outputs = self.bert_model(**inputs)
            sentence_embeddings = torch.mean(outputs.last_hidden_state, dim=1).to(device)
            self.create_bags(data, sentence_embeddings, batch_labels, batch_entity_pairs)
            
        return data

    def create_bags(self, data, sentence_embeddings, labels, entity_pairs):
        for sentence_embedding, label, (entity_pair_0, entity_pair_1) in zip(sentence_embeddings, labels, entity_pairs):
            data[(entity_pair_0, entity_pair_1)]['embeddings'].append(sentence_embedding)
            data[(entity_pair_0, entity_pair_1)]['labels'].append(label)
            data[(entity_pair_0, entity_pair_1)]['entity_pairs'].append((entity_pair_0, entity_pair_1))

        for pair_key in data:
            if 'embeddings' not in data[pair_key]:
                continue
            bag = torch.stack(data[pair_key]['embeddings']).to(device)
            attention_weights = self.attention(bag)
            attention_applied = (bag * attention_weights).sum(dim=0)
            
            data[pair_key]['bag'] = bag
            data[pair_key]['relation representation'] = attention_applied

        return data

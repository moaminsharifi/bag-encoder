import os
import datetime

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
from GIDS_dataset import GIDSDataset, train_data_file, test_data_file, val_data_file
#from data_loader_NYT import train_dataset, test_dataset
from Bin_classfier import MLPClassifier


# Hyperparameters

num_epochs = 2
lr = 0.00001
batch_size = 64
max_length = 500
n_classes = 2
input_size = 768 
hidden_size = 200 
output_size = n_classes
train_losses = []
precisions = []
recalls = []
f1s = []
aucs = []
val_losses = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_date = datetime.datetime.now().strftime("%Y-%m-%d")
t_dt = datetime.datetime.now().strftime("%Y%m%d-%H%M")
log_dir = f"tensorboard/e-{num_epochs}-b-{batch_size}-{t_dt}"
df_data_loss = {"loss": [], "epoch": [], "step": []}
metrics_df = pd.DataFrame(columns=['Epoch', 'Precision', 'Recall', 'F1', 'auc', 'Validation loss'])
writer = SummaryWriter(log_dir)
print(writer)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
classifier = MLPClassifier(input_size, hidden_size).to(device)
#loss_fn = BCELoss()
loss_fn = BCEWithLogitsLoss()


best_val_loss = float('inf')
#early_stopping_tolerance = 3
#early_stopping_threshold = 0.03
#early_stopping_counter = 0

optimizer = torch.optim.Adadelta(classifier.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Initialize Dataset and DataLoader
test_dataset = GIDSDataset(test_data_file)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
val_dataset = GIDSDataset(test_data_file)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

train_dataset = GIDSDataset(train_data_file)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
LABELS = np.array([0, 1]).astype(int)
embeddings = []
scaler = Normalizer()
is_scaler_trained = False
# Training loop
data_set_size = (len(train_dataset) // batch_size) + 1
bert_model.train()  # Set the model to train mode
classifier.train()
for epoch in range(num_epochs):
    train_loss = 0.0
    step_count = 0
    for batch_sentences, batch_labels in tqdm(train_dataloader):
        step_count += 1
        batch_labels = batch_labels.float().to(device)
        batch_embeddings = []
        for sentence in batch_sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}  # Move tensors to dev
            bert_outputs = bert_model(**inputs, output_attentions=True)
            last_hidden_state = bert_outputs[0]
            sentence_embedding = torch.mean(bert_outputs.last_hidden_state, dim=1)
            batch_embeddings.append(sentence_embedding)

        batch_embeddings = torch.stack(batch_embeddings).to(device)
        optimizer.zero_grad()
        # classifier_inputs = sentence_embedding.detach()
        classifier_outputs = classifier(batch_embeddings)
        classifier_outputs = classifier_outputs.view(-1, 1)
        batch_labels = batch_labels.view(-1, 1)

        loss = loss_fn(classifier_outputs, batch_labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        train_loss += loss.item()
        df_data_loss["loss"].append(loss.item())
        df_data_loss["epoch"].append(epoch)
        ep_step_count = (data_set_size * epoch) + step_count - 1
        df_data_loss["step"].append(ep_step_count)
        predictions = (classifier_outputs > 0.5).float().cpu().numpy().astype(int)
        batch_labels_c = batch_labels.float().cpu().numpy().astype(int)
        precision = precision_score(batch_labels_c, predictions, average='binary', labels=LABELS, zero_division=0)
        recall = recall_score(batch_labels_c, predictions, average='binary', labels=LABELS, zero_division=0)
        f1 = f1_score(batch_labels_c, predictions, average='binary', labels=LABELS, zero_division=0)
        writer.add_scalar("Loss", loss.item(), ep_step_count)
        writer.add_scalar("Precision", precision, ep_step_count)
        writer.add_scalar("Recall", recall, ep_step_count)
        writer.add_scalar("F1 Score", f1, ep_step_count)
        print(f'Epoch: {epoch}, Training Loss: {train_loss / step_count}')

    classifier.eval()
    bert_model.eval()
    with torch.no_grad():
        val_relations = []
        val_outputs = []

        val_loss = []
        step_count = 0
        for batch_sentences, batch_labels in tqdm(val_dataloader):

            batch_labels = batch_labels.float().to(device)
            batch_embeddings = []
            for sentence in batch_sentences:
                inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True,
                                   max_length=max_length)
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}  # Move tensors to dev
                bert_outputs = bert_model(**inputs, output_attentions=True)
                sentence_embedding = torch.mean(bert_outputs.last_hidden_state, dim=1)
                batch_embeddings.append(sentence_embedding)

            batch_embeddings = torch.stack(batch_embeddings).to(device)
            # batch_embeddings = torch.stack(batch_embeddings).to(device)
            # classifier_inputs = sentence_embedding.detach()
            classifier_outputs = classifier(batch_embeddings)
            classifier_outputs = classifier_outputs.view(-1)
            classifier_outputs = classifier_outputs.squeeze()
            loss = loss_fn(classifier_outputs, batch_labels)
            batch_labels = batch_labels.view(-1, 1)
            predictions = (classifier_outputs > 0.5).float()
            # predictions = torch.max(classifier_outputs, dim=1)[1]
            val_loss.append(loss.item())
            val_relations.extend(batch_labels.cpu().numpy())
            val_outputs.extend(predictions.cpu().numpy())
            step_count += 1

        val_outputs = np.array(val_outputs).astype(int).reshape(-1)
        val_relations = np.array(val_relations).astype(int).reshape(-1)

        precision = precision_score(val_relations, val_outputs, average='binary', labels=LABELS, zero_division=0)
        recall = recall_score(val_relations, val_outputs, average='binary', labels=LABELS, zero_division=0)
        f1 = f1_score(val_relations, val_outputs, average='binary', labels=LABELS, zero_division=0)
        auc = roc_auc_score(val_relations, val_outputs)
        avg_val_loss = sum(val_loss) / len(val_loss)
        #if best_val_loss - avg_val_loss > early_stopping_threshold:
        #    best_val_loss = avg_val_loss
        #    early_stopping_counter = 0  # reset counter
        #else:
        #    early_stopping_counter += 1
        #if early_stopping_counter >= early_stopping_tolerance:
        #    print("Early stopping due to no improvement in validation loss")
        #    break
        print(f'Epoch: {epoch}, Precision: {precision}, Recall: {recall}, F1: {f1}')
            
        d_ = {'Epoch': epoch, 'Precision': precision, 'Recall': recall, 'F1': f1,
                  'auc': auc, 'Validation loss': avg_val_loss}
        metrics_df.loc[len(metrics_df.index)] = list(d_.values())

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        aucs.append(auc)
        val_loss.append(loss.item())
        val_losses.append(avg_val_loss)
        classifier_outputs_cpu = classifier_outputs.cpu().numpy()
        predictions_cpu = predictions.cpu().numpy()
        #print(classification_report(classifier_outputs_cpu, predictions_cpu))

        print(f'Epoch: {epoch}, Precision: {precision}, Recall: {recall}, F1: {f1},  auc : {auc}, Validation loss : {avg_val_loss}')

# Save the trained model
torch.save(classifier.state_dict(), 'relation_extraction_model.pt')

df = pd.DataFrame(df_data_loss)
df.to_csv(f"loss-{datetime.datetime.now().strftime('%Y-%m-%d')}-e-{num_epochs}.csv")
metrics_df.to_csv(f"results-{datetime.datetime.now().strftime('%Y-%m-%d')}-e-{num_epochs}.csv", index=False)
writer.close()
print("Training completed!")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Training and Validation Loss.png")
# plt.show()


# Plot precision, recall, and F1 score
plt.figure(figsize=(10, 5))
plt.title("Precision, Recall, and F1 Score")
plt.plot(precisions, label="Precision")
plt.plot(recalls, label="Recall")
plt.plot(f1s, label="F1 Score")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.savefig("Precision, Recall, and F1 Score.png")

# plt.show()

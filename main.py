import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from Bag_Encoder_Dataloader_NYT10 import train_dataset, test_dataset
from Bag_Encoder_Dataloader_GIDS import train_dataset, test_dataset
from Bag_encoder import BagEncoder
from Bin_classfier import MLPClassifier

# Define hyperparameters
epochs = 5
learning_rate = 0.00001
batch_size = 240
input_size = 768
hidden_size1 = 200
hidden_size2 = 100
hidden_size3 = 100
max_length = 500
num_labels = 2

precisions = []
recalls = []
f1s = []
aucs = []
val_losses = []

LABELS = np.array([0, 1]).astype(int)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_date = datetime.datetime.now().strftime("%Y-%m-%d")
t_dt = datetime.datetime.now().strftime("%Y%m%d-%H%M")
log_dir = f"tensorboard/e-{epochs}-b-{batch_size}-{t_dt}"
df_data_loss = {"loss": [], "epoch": [], "step": []}
metrics_df_testing = pd.DataFrame(columns=['Epoch', 'Precision', 'Recall', 'F1', 'auc', 'Validation loss'])

writer = SummaryWriter(log_dir)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

weights = torch.tensor([0.25]).to(device)  # Set weight of class 0 to 1, weight of class 1 to 4

bag_encoder = BagEncoder(pretrained_model_name="bert-base-uncased", batch_size=batch_size)
mlp_classifier = MLPClassifier(input_size, hidden_size1, hidden_size2, hidden_size3).to(device)

total_params = sum(p.numel() for p in mlp_classifier.parameters())
print(total_params)
optimizer = torch.optim.Adam(mlp_classifier.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
best_val_loss = float('inf')
data_set_size = (len(train_dataset) // batch_size) + 1
mlp_classifier.train()
train_losses = []
for epoch in range(epochs):
    mlp_classifier.train()  # Set the model to training mode
    train_loss = 0
    step_count = 0
    for batch in tqdm.tqdm(train_dataloader):
        step_count += 1
        sentences, labels, entity_pairs = batch

        bag_embeddings = bag_encoder.encode_sentences(sentences, labels, entity_pairs)

        batch_loss = 0
        for pair_key in bag_embeddings:
            sentence_embeddings = bag_embeddings[pair_key]['relation representation'].to(device)
            # bag_embedding = torch.mean(sentence_embeddings, dim=0, keepdim=True)
            label = torch.tensor(bag_embeddings[pair_key]['labels'], dtype=torch.float).to(device)

            # print(f" sentence_embeddings: {sentence_embeddings.size()}")
            # print(f" bag_embedding: {bag_embedding.size()}")
            # print(f" label: {label.size()}")
            optimizer.zero_grad()
            outputs = mlp_classifier(sentence_embeddings).squeeze(0).to(device)
            loss = criterion(outputs.view(-1), label)
            loss.backward(retain_graph=True)
            optimizer.step()
            train_losses.append(loss.item())
            train_loss += loss.item()

        ep_step_count = (data_set_size * epoch) + step_count - 1
        avg_train_loss = train_loss / step_count
        df_data_loss['loss'].append(avg_train_loss)
        df_data_loss['epoch'].append(epoch + 1)
        df_data_loss["step"].append(ep_step_count)

    print(f'Epoch: {epoch}, Training Loss: {avg_train_loss}')

    # Validation phase
    mlp_classifier.eval()
    with torch.no_grad():
        all_labels = []
        all_predictions = []
        losses = []
        val_loss = 0
        val_steps = 0

        for batch in tqdm.tqdm(test_dataloader):
            sentences, labels, entity_pairs = batch
            bag_embeddings = bag_encoder.encode_sentences(sentences, labels, entity_pairs)

            for pair_key in bag_embeddings:
                sentence_embeddings = bag_embeddings[pair_key]['relation representation'].to(device)
                bag_embedding = torch.mean(sentence_embeddings, dim=0, keepdim=True)
                label = torch.tensor(bag_embeddings[pair_key]['labels'], dtype=torch.float).to(device)
                # print (f"label:{label}")
                outputs = mlp_classifier(sentence_embeddings).squeeze(0).to(device)
                # print (f"label:{outputs}")
                loss = criterion(outputs.view(-1), label)
                # print (f"label:{loss}")
                losses.append(loss.item())
                predicted = (outputs > 0.5).float()
                # print (f"label:{predicted}")
                all_labels.extend(label.cpu().numpy().tolist())
                all_predictions.extend(predicted.view(-1).cpu().numpy().tolist())

            val_loss += losses[-1]
            val_steps += 1
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_predictions)
        # print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss / val_steps}')
        writer.add_scalar("Validation Loss", val_loss / val_steps, epoch)
        writer.add_scalar("Validation Precision", precision, epoch)
        writer.add_scalar("Validation Recall", recall, epoch)
        writer.add_scalar("Validation F1 Score", f1, epoch)
        writer.add_scalar("Validation AUC", auc, epoch)

        print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / val_steps}')

    metrics_df_testing.loc[epoch, ['Epoch', 'Precision', 'Recall', 'F1', 'auc', 'Validation loss']] = [epoch + 1,
                                                                                                       precision,
                                                                                                       recall, f1, auc,
                                                                                                       val_loss / val_steps]

torch.save(mlp_classifier.state_dict(), 'relation_extraction_model.pt')
df = pd.DataFrame(df_data_loss)
df.to_csv(f"loss-{datetime.datetime.now().strftime('%Y-%m-%d')}-e-{epochs}.csv")
metrics_df_testing.to_csv(f"results-{datetime.datetime.now().strftime('%Y-%m-%d')}-e-{epochs}.csv", index=False)
writer.close()
print("Training completed!")

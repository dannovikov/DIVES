import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import wandb
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
os.environ['WANDB_SILENT'] = "true"
import pickle
import random
import traceback
from validation import compute_model_statistics

if __name__ == "__main__":
    wandb.init(project="seqvae2")

from tqdm import tqdm

TRAIN=False

EPOCHS = 25
BATCH_SIZE = 64
LR = 0.001
EMBED_DIM = 2
NUM_WORKERS = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data/preproc"
TRAINING_MODELS = "./models/"
TRAINING_IMAGES = "./images/"

LOSS_RECON = 0.2
LOSS_CLASS = 0.8



class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)
        return x, self.y[idx], idx


def get_data():

    #train
    X_train = torch.load(f"{DATA_DIR}/train_seqs_tensor.pt")
    y_train = torch.load(f"{DATA_DIR}/train_labels_tensor.pt")
    with open(f"{DATA_DIR}/train_map_seqid_to_row.pkl", "rb") as f:
        train_map_seqid_to_row = pickle.load(f)
    with open(f"{DATA_DIR}/train_map_row_to_seqid.pkl", "rb") as f:
        train_map_row_to_seqid = pickle.load(f)
    with open(f"{DATA_DIR}/train_map_subtype_to_seqids.pkl", "rb") as f:
        train_map_subtype_to_seqids = pickle.load(f)
    with open(f"{DATA_DIR}/train_map_seqid_to_subtype.pkl", "rb") as f:
        train_map_seqid_to_subtype = pickle.load(f)

    #test
    X_test = torch.load(f"{DATA_DIR}/test_seqs_tensor.pt")
    y_test = torch.load(f"{DATA_DIR}/test_labels_tensor.pt")
    with open(f"{DATA_DIR}/test_map_seqid_to_row.pkl", "rb") as f:
        test_map_seqid_to_row = pickle.load(f)
    with open(f"{DATA_DIR}/test_map_row_to_seqid.pkl", "rb") as f:
        test_map_row_to_seqid = pickle.load(f)
    with open(f"{DATA_DIR}/test_map_subtype_to_seqids.pkl", "rb") as f:
        test_map_subtype_to_seqids = pickle.load(f)
    with open(f"{DATA_DIR}/test_map_seqid_to_subtype.pkl", "rb") as f:
        test_map_seqid_to_subtype = pickle.load(f)

    #common
    with open(f"{DATA_DIR}/map_label_to_subtype.pkl", "rb") as f:
        map_label_to_subtype = pickle.load(f)
    with open(f"{DATA_DIR}/map_subtype_to_label.pkl", "rb") as f:
        map_subtype_to_label = pickle.load(f)

    result = {
        "train": {
            "X": X_train,
            "y": y_train,
            "map_seqid_to_row": train_map_seqid_to_row,
            "map_row_to_seqid": train_map_row_to_seqid,
            "map_subtype_to_seqids": train_map_subtype_to_seqids,
            "map_seqid_to_subtype": train_map_seqid_to_subtype,
        },
        "test": {
            "X": X_test,
            "y": y_test,
            "map_seqid_to_row": test_map_seqid_to_row,
            "map_row_to_seqid": test_map_row_to_seqid,
            "map_subtype_to_seqids": test_map_subtype_to_seqids,
            "map_seqid_to_subtype": test_map_seqid_to_subtype,
        },
        "map": {
            "label_to_subtype": map_label_to_subtype,
            "subtype_to_label": map_subtype_to_label,
        }
    }
    return result


class MultiKernelConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1):
        super().__init__()
        self.kernels = nn.ModuleList([nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2) for kernel_size in kernel_sizes])

    def forward(self, x):
        return torch.cat([kernel(x) for kernel in self.kernels], dim=1)
    
class PrintShape(nn.Module):
    def __init__(self, text=""):
        super().__init__()
        self.text = text
    def forward(self, x):
        print(self.text, x.shape)
        return x

class simpleNet(nn.Module):
    def __init__(self, sequence_length, sequence_embedding_dim, num_classes):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_embedding_dim = sequence_embedding_dim
        self.num_classes = num_classes

        self.sequence_embedding = nn.Sequential(
            MultiKernelConv1d(1, 100, [3, 5, 7]),
            nn.ReLU(),
            # PrintShape("After multiconv"),
            nn.MaxPool1d(2),
            # PrintShape("After maxpool"),
            nn.Flatten(),
            # PrintShape("After flatten"),
            nn.Linear(1500 * 300, sequence_embedding_dim),
            # PrintShape("After linear"),
            nn.BatchNorm1d(sequence_embedding_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(sequence_embedding_dim, sequence_embedding_dim*2),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*2, sequence_embedding_dim*4),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*4, sequence_embedding_dim*8),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*8, sequence_embedding_dim*16),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*16, sequence_embedding_dim*32),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*32, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
        )
        self.decoder = nn.Sequential(
            nn.Linear(sequence_embedding_dim, sequence_embedding_dim*2),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*2, sequence_embedding_dim*4),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*4, sequence_embedding_dim*8),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*8, sequence_embedding_dim*16),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*16, sequence_embedding_dim*32),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*32, sequence_length//4),
            nn.ReLU(),
            nn.Linear(sequence_length//4, sequence_length//2),
            nn.ReLU(),
            nn.Linear(sequence_length//2, sequence_length),
        )

    def forward(self, x):
        embedding = self.sequence_embedding(x)
        classification = self.classifier(embedding)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction.unsqueeze(1), classification

    def classify(self, z):
        return self.classifier(z)

def plot_latent_space(model, dataloader, epoch, with_labels = False, map_label_to_subtype=None, map_row_to_seqid=None, map_subtype_to_seqids=None):
    model.eval()
    latent_space = []
    seq_ids = []
    labels = []
    with torch.no_grad():
        for i, (x, y, idx) in tqdm(enumerate(dataloader), desc="Plotting latent space"):
            x = x.to(DEVICE)
            emb, rec, clas = model(x)
            latent_space.append(emb.squeeze(1).cpu().numpy())
            labels.append(y.cpu().numpy())
            if with_labels:
                seq_ids.extend([map_row_to_seqid[i] for i in idx.cpu().numpy()])
    latent_space = np.concatenate(latent_space)
    print(latent_space.shape)
    labels = np.concatenate(labels)

    plt.figure(figsize=(10, 10))
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c=labels)
    plt.colorbar()

    if with_labels:
        subtype_label_positions = {}
        for subtype in map_subtype_to_seqids.keys():
            subtype_label_positions[subtype] = np.mean(latent_space[np.isin(seq_ids, map_subtype_to_seqids[subtype])], axis=0)
        for subtype, pos in subtype_label_positions.items():
            plt.annotate(subtype, pos, fontsize=8)

        plt.savefig(f"{TRAINING_IMAGES}/latent_space_{epoch}_labeled.png")
    else:
        plt.savefig(f"{TRAINING_IMAGES}/latent_space_{epoch}.png")
    plt.close()

def classify_latent_space(model, dataloader, epoch, fr=-3, to=3):
    # create a grid of points in the latent space and classify them

    grid = np.mgrid[fr:to:0.005, fr:to:0.005].reshape(2, -1).T
    grid = torch.from_numpy(grid).float().to(DEVICE)

    model.eval()
    # now classify the grid
    
    with torch.no_grad():
        classification = model.classify(grid)
        _, predicted = torch.max(classification.data, 1)

    grid = grid.cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(grid[:, 0], grid[:, 1], c=predicted.cpu().numpy()) 
    plt.colorbar()

    plt.savefig(f"{TRAINING_IMAGES}/latent_space_{epoch}_classified.png")


def main():
    # Initialize data structures
    data = get_data()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]
    map_row_to_seqid = data["train"]["map_row_to_seqid"]
    map_label_to_subtype = data["map"]["label_to_subtype"]

    # Initialize dataloaders
    train_dataset = SeqDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    test_dataset = SeqDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # create the model
    model = simpleNet(sequence_length=X_train.shape[1], sequence_embedding_dim=EMBED_DIM, num_classes=len(map_label_to_subtype))
    # model.load_state_dict(torch.load('saved96_model_26.pt'))
    model.to(DEVICE)

    # Set up criteria, optimizer, and scheduler
    reconstruction_loss = nn.MSELoss()
    classification_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)

    # Train the model
    for epoch in range(EPOCHS):
        plot_latent_space(model, train_dataloader, epoch, True, map_label_to_subtype, map_row_to_seqid, data["train"]["map_subtype_to_seqids"])
        for i, (x, y, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch: {epoch}'):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            model.train()
            optimizer.zero_grad()
            embedding, reconstruction, classification = model(x)
            loss = reconstruction_loss(reconstruction, x) * LOSS_RECON + classification_loss(classification, y) * LOSS_CLASS 
            loss.backward()
            wandb.log({"train_loss": loss.item()})
            optimizer.step()

        model.eval()
        with torch.no_grad():
            avg_loss = 0
            TP = 0
            N = 0
            for x, y, rowidx in test_dataloader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                embedding, reconstruction, classification = model(x)
                rec_loss = reconstruction_loss(reconstruction, x)
                class_loss = classification_loss(classification, y)
                loss = reconstruction_loss(reconstruction, x) * LOSS_RECON + classification_loss(classification, y) * LOSS_CLASS 
                avg_loss += loss.item()
                TP += torch.sum(torch.argmax(classification, dim=1) == y).item()
                N += len(y)
            avg_loss /= len(test_dataloader)
            accuracy = TP / N
            wandb.log({"test_loss": avg_loss})
            wandb.log({"test_accuracy": accuracy})
            print(f"Epoch {epoch}: Test loss: {avg_loss}, Test accuracy: {accuracy}")
            scheduler.step(avg_loss)
        torch.save(model.state_dict(), f"{TRAINING_MODELS}/model_{epoch}.pt")
        



def eval_model(model_path):
    # Initialize data structures
    data = get_data()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]
    map_row_to_seqid = data["train"]["map_row_to_seqid"]
    map_label_to_subtype = data["map"]["label_to_subtype"]

    # Initialize dataloaders
    train_dataset = SeqDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    test_dataset = SeqDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # create the model
    model = simpleNet(sequence_length=X_train.shape[1], sequence_embedding_dim=EMBED_DIM, num_classes=len(map_label_to_subtype))
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)

    # plot_latent_space(model, train_dataloader, 25, with_labels=True, map_label_to_subtype=map_label_to_subtype, map_row_to_seqid=map_row_to_seqid, map_subtype_to_seqids=data["train"]["map_subtype_to_seqids"])

    # evaluate the model on classification accuracy of the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _ in test_dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            _, _, outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"Accuracy of the network on the {total} sequences: {100 * correct / total}%")

    classify_latent_space(model, train_dataloader, 25)

    # print("Computing model statistics")
    # stats = compute_model_statistics(model, test_dataloader, map_label_to_subtype)
    # print(stats)

    # #create a pandas dataframe and save it to disk, rows are subtypes, columns are statistics
    # import pandas as pd
    # df = pd.DataFrame(stats)
    # df = df.transpose()
    # df.columns = ["accuracy", "precision", "sensitivity", "specificity"]
    # df.to_csv(f"{TRAINING_MODELS}/model_statistics.csv")




if __name__ == "__main__":
    if TRAIN:
        main()
    else:
        eval_model(r"C:\Users\Dan\Desktop\CDC\Projects\dives\models\model_24.pt")
        






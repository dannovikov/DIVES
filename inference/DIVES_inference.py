import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
import sys


TRAIN=False
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.0001
EMBED_DIM = 2
NUM_WORKERS = 2
LOSS_RECON = 0.2
LOSS_CLASS = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data/preproc"
TRAINING_MODELS = "./models/"
TRAINING_IMAGES = "./images/"

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)
        return x, self.y[idx], idx

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

class divenet(nn.Module):
    def __init__(self, sequence_length, sequence_embedding_dim, num_classes):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_embedding_dim = sequence_embedding_dim
        self.num_classes = num_classes

        self.sequence_embedding = nn.Sequential(
            MultiKernelConv1d(1, 100, [3, 5, 7]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(self.sequence_length//2 * 100 * 3, sequence_embedding_dim),
            # nn.ReLU(),
            # nn.Linear(sequence_embedding_dim*8, sequence_embedding_dim),
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

def classify_latent_space(model, dataloader, epoch, fr=-3, to=3, with_labels=True, map_label_to_subtype=None, map_row_to_seqid=None, map_subtype_to_seqids=None):
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

    if with_labels:
        subtype_label_positions = {}
        for subtype in map_subtype_to_seqids.keys():                                                         #reverse lookup (todo: fix properly)
            subtype_label_positions[subtype] = np.mean(grid[predicted.cpu().numpy() == list(map_label_to_subtype.keys())[list(map_label_to_subtype.values()).index(subtype)]], axis=0)
        for subtype, pos in subtype_label_positions.items():
            plt.annotate(subtype, pos, fontsize=8)

    plt.savefig(f"{TRAINING_IMAGES}/latent_space_{epoch}_classified.png")

def get_data_inference(data_dir):
    X = torch.load(f"{data_dir}/seqs_tensor.pt")
    with open(f"{data_dir}/map_seqid_to_row.pkl", "rb") as f:
        map_seqid_to_row = pickle.load(f)
    with open(f"{data_dir}/map_row_to_seqid.pkl", "rb") as f:
        map_row_to_seqid = pickle.load(f)
    with open(f"{data_dir}/map_label_to_subtype.pkl", "rb") as f:
        map_label_to_subtype = pickle.load(f)
    return X, map_seqid_to_row, map_row_to_seqid, map_label_to_subtype

def inference(model_path, data_dir, output_dir):
    # Initialize data structures
    X, map_seqid_to_row, map_row_to_seqid, map_label_to_subtype = get_data_inference(data_dir)
    dataset = SeqDataset(X, torch.zeros(X.shape[0]))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    model = divenet(sequence_length=3000, sequence_embedding_dim=2, num_classes=103)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)

    model.eval()
    with torch.no_grad():
        # results = {"embeddings": []}
        results = {}

        for i, (x, _, idx) in tqdm(enumerate(dataloader), desc="Inference"):
            x = x.to(DEVICE)
            emb, rec, clas = model(x)
            # store the embeddings
            emb = emb.squeeze(1).cpu().numpy()
            # results["embeddings"].append(emb)
            # store the classifications
            for j, row in enumerate(idx):
                pred = clas[j].argmax().item()
                subtype = map_label_to_subtype[pred]
                seqid = map_row_to_seqid[row.item()]
                results[seqid] = subtype
            
    import pandas as pd
    df = pd.DataFrame(results.items(), columns=["seqid", "subtype"])
    df.to_csv(f"{output_dir}/inference_results.csv", index=False)

    print("Inference results saved to", f"{output_dir}/inference_results.csv")


if __name__ == "__main__":
    print("Running DIVES inference using device:", DEVICE)
    data_directory = sys.argv[1]
    output_directory = sys.argv[2]
    model_path = "./model_weights.pt"
    inference(model_path, data_directory, output_directory)



        






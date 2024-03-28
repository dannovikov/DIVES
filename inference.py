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
from validation import compute_model_statistics
from tqdm import tqdm


BATCH_SIZE = 64
EMBED_DIM = 2
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data/preproc"
MODEL_PATH = "./models/model_71.pt"

"""
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)
        return x, self.y[idx], idx
"""

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        return x, self.y[idx], idx
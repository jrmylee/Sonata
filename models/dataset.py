import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
class ChordDataset(Dataset):
    def __init__(self, paths):
        self.le = LabelEncoder()
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.paths[idx]
        file = torch.load(filename)

        audio = torch.log(torch.abs(file['feature']) + 1e-6)
        sample = { 'audio': audio, 'chord': file['chord']}
        return sample
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np

import os
class ChordDataset(Dataset):
    def __init__(self, main_dir, paths):
        self.le = LabelEncoder()
        self.main_dir = main_dir
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.paths[idx]
        file = torch.load(os.path.join(self.main_dir, filename))

        audio = torch.log(torch.abs(file['feature']) + 1e-6)
        sample = { 'audio': audio, 'chords': self.le.fit_transform(file['chords'])}
        return sample
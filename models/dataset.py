import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
class ChordDataset(Dataset):
    def __init__(self, dataset):
        self.le = LabelEncoder()
        self.dataset = [torch.load(d) for d in dataset]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio = torch.log(torch.abs(torch.tensor(self.dataset[idx]['features'], dtype=torch.float32)) + 1e-6)
        chords = self.le.fit_transform(self.dataset[idx]['chords'])
        sample = { 'audio': audio, 'chord': chords}
        return sample
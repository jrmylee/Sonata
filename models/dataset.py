from torch import is_tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
class ChordDataset(Dataset):
    def __init__(self, dataset):
        le = LabelEncoder()
        d = [[None, None] for i in range(len(dataset))]
        for i in range(len(dataset)):
            d[i][0] = dataset[i][0]
            d[i][1] = le.fit_transform(dataset[i][1])
        self.dataset = d
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        audio = np.log(np.abs(self.dataset[idx][0]) + 1e-6)
        sample = { 'audio': audio, 'chord': self.dataset[idx][1]}
        return sample
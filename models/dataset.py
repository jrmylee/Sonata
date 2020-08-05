from torch import is_tensor
from torch.utils.data import Dataset, DataLoader
class ChordDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        sample = { 'audio': self.dataset[idx][0], 'chord': self.dataset[idx][1]}
        return sample
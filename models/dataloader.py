from torch.utils.data import DataLoader
import torch

def _collate_fn(batch):
    batch_size = len(batch)

    features = []
    chords = []

    for i in range(batch_size):
        sample = batch[i]
        feature = sample['audio']
        chord = sample['chord']
        features.append(feature)
        chords.append(chord)
    features = torch.tensor(features, dtype=torch.float32)
    chords = torch.tensor(chords, dtype=torch.int64)

    return features, chords

class ChordDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ChordDataloader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
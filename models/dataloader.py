from torch.utils.data import DataLoader
import torch

def _collate_fn(batch):
    batch_size = len(batch)

    features = torch.Tensor(batch_size, 108, 108)
    featues_arr = []
    chords = []

    for i in range(batch_size):
        sample = batch[i]
        feature = sample['audio']
        chord = sample['chord']
        featues_arr.append(feature)
        chords.append(chord)
    torch.cat(featues_arr, out=features)
    chords = torch.tensor(chords, dtype=torch.int64)

    return features, chords

class ChordDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ChordDataloader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
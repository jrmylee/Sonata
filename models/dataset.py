import torch
from torch.utils.data import Dataset, DataLoader
from models.chords import *
import numpy as np

import os
class ChordDataset(Dataset):
    def __init__(self, main_dir, paths):
        self.chord_class = Chords()
        self.main_dir = main_dir
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.paths[idx]
        file = torch.load(os.path.join(self.main_dir, filename))

        chords = self.chord_class.chords(file['chords'])
        chords = self.chord_class.reduce_to_triads(chords)
        chords = self.chord_class.assign_chord_id(chords)
        chords = list(chords['chord_id'])

        audio = torch.log(torch.abs(file['feature']) + 1e-6)
        sample = { 'audio': audio, 'chords': chords}
        return sample
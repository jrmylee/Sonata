# from comet_ml import Experiment
# experiment = Experiment(api_key="HUSGKDDKrFXWOdBLDY3GAhIXD",
#                         project_name="sonata", workspace="jrmyleecape")

import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import sys
import yaml
import torch

from models.preprocess import Preprocess
from models.dataset import ChordDataset
from models.dataloader import ChordDataloader
from models.augment import Augment
from models.chords import Chords

config = yaml.load(open("./config/config.yaml"))

sys.path.insert(1, config['model']['path'])
from btc_model import *
from torch.utils.data import DataLoader

sr = config['preprocess']['sample_rate']
hop_size = config['preprocess']['hop_size']
window_size = config['preprocess']['window_size']
song_hz = config['preprocess']['song_hz']

p = Preprocess(sr, hop_size, song_hz, window_size, Augment(Chords()))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_dataset_from_file(file_path):
    loaded = torch.load(file_path)
    features = loaded['features']
    chords = loaded['chords']
    if len(features[0]) > 108:
        features = [arr[0:108] for arr in features]
    return list(zip(features, chords))

model = BTC_model(config=config['model']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)

features_path = "/Users/jrmylee/Documents/Development/projects/mir/playground/checkpoints/checkpoint.pth"
full_dataset = get_dataset_from_file(features_path)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_set, test_set = ChordDataset(train_dataset), ChordDataset(test_dataset) 
train_dataloader = ChordDataloader(train_set, batch_size=128, shuffle=True, num_workers=4)
test_dataloader = ChordDataloader(test_set, batch_size=128, shuffle=True, num_workers=4)

for epoch in range(1):
    model.train()
    train_loss_list = []
    total = 0.
    correct = 0.
    second_correct = 0.
    print("epoch: " + str(epoch))
#     Training
    print("Training")
    for i_batch, data in enumerate(train_dataloader):
        print("Batch: " + str(i_batch))
        features, chords = data

        batch_size = features.shape[0]
        features = features.unsqueeze(1).expand(batch_size, 108, 108)
        features.requires_grad = True
        
        optimizer.zero_grad()
        features = features.to(device)
        chords = chords.to(device)
        # Train
        prediction, total_loss, weights, second = model(features, chords)

        total_loss.backward()
        optimizer.step()
# Validation
    print("validation")

    with torch.no_grad():
        model.eval()
        n = 0
        for i, data in enumerate(test_dataloader):
            features, chords = data['audio'], data['chord']
            features = torch.tensor(features)
            batch_size = features.shape[0]
            features = features.unsqueeze(1).expand(batch_size, 108, 108)
            features.requires_grad = True
            
            optimizer.zero_grad()
            features = features.to(device)
            chords = chords.to(device)
            # Train
            prediction, total_loss, weights, second = model(features, chords)



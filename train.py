import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import sys
import preprocess
from dataset import ChordDataset

config = {
    'model': {
        'path': '/Users/jrmylee/Documents/Development/projects/mir/repos/BTC-ISMIR19',
        'feature_size' : 108,
        'timestep' : 108,
        'num_chords' : 133,
        'input_dropout' : 0.2,
        'layer_dropout' : 0.2,
        'attention_dropout' : 0.2,
        'relu_dropout' : 0.2,
        'num_layers' : 8,
        'num_heads' : 4,
        'hidden_size' : 128,
        'total_key_depth' : 128,
        'total_value_depth' : 128,
        'filter_size' : 128,
        'loss' : 'ce',
        'probs_out' : False
    } 
}

sys.path.insert(1, config['model']['path'])
from btc_model import *

import torch
from torch.utils.data import DataLoader
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Get rid of this later, replace with internal chord system
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def get_data():
    datasets = {
        "isophonics-beetles" : {
            "mp3": "/Users/jrmylee/Documents/Development/projects/mir/datasets/isophonics/beetles_albums",
            "labels": "/Users/jrmylee/Documents/Development/projects/mir/datasets/isophonics/beetles_annotations"
        },
        "isophonics-king" : {
            "mp3": "/Users/jrmylee/Documents/Development/projects/mir/datasets/isophonics/carol_king_albums",
            "labels": "/Users/jrmylee/Documents/Development/projects/mir/datasets/isophonics/carol_king_annotations"
        }
    }

    sample_rate = 22050

    p = preprocess.Preprocess()

    king_albums = p.get_mp3(datasets['isophonics-king']['mp3'])
    king_labels = p.get_labels(datasets['isophonics-king']['labels'])
    beetles_albums = p.get_mp3(datasets['isophonics-beetles']['mp3'])
    beetles_labels = p.get_labels(datasets['isophonics-beetles']['labels'])

    data = [
        (king_albums, king_labels),
        (beetles_albums, beetles_labels)
    ]
    return data

def get_chords_and_features(data):
    features, chords = [], []
    for d in data:
        album_label_dict = {}
        albums_dict = d[0]
        l_dict = d[1]
        for label_path in l_dict:
            song_label_dict = generate_song_labels(label_path, l_dict)
            album_title = path_to_album(label_path)
            album_label_dict[album_title] = song_label_dict
        f, c = generate_features(albums_dict, album_label_dict)
        features.append(f)
        chords.append(c)

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
train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=4)


for epoch in range(3):
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
        features, chords = data['audio'], data['chord']
        chords = [le.fit_transform(arr) for arr in chords] #encode string labels
        features = torch.tensor(features)
        features = features.unsqueeze(1).expand(128, 108, 108)
        features.requires_grad = True
        
        optimizer.zero_grad()
        
        chords = torch.tensor(chords).reshape(128, 108) 
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
            chords = [le.fit_transform(arr) for arr in chords] #encode string labels
            features = torch.tensor(features)
            features = features.unsqueeze(1).expand(128, 108, 108)
            features.requires_grad = True
            
            optimizer.zero_grad()
            
            chords = torch.tensor(chords).reshape(128, 108) 
            features = features.to(device)
            chords = chords.to(device)
            # Train
            prediction, total_loss, weights, second = model(features, chords)



# from comet_ml import Experiment
# experiment = Experiment(api_key="HUSGKDDKrFXWOdBLDY3GAhIXD",
#                         project_name="sonata", workspace="jrmyleecape")

import numpy as np
import librosa
import librosa.display
import os
import sys
import yaml
import torch
import multiprocessing as mp
from nnAudio import Spectrogram

from models.preprocess import Preprocess
from models.dataset import ChordDataset
from models.dataloader import ChordDataloader
from models.augment import Augment
from models.chords import Chords

config = yaml.load(open("./config/config.yaml"))

sys.path.insert(1, config['model']['path'])
from btc_model import *

from torch.utils.data import DataLoader


if torch.cuda.is_available():
    print("Using GPU")
    device = "cuda:0"
else:
    print("Using CPU")
    device = "cpu"

aug = Augment(Chords())
config = yaml.load(open("./config/config.yaml"))
sr = config['preprocess']['sample_rate']
hop_size = config['preprocess']['hop_size']
window_size = config['preprocess']['window_size']
song_hz = config['preprocess']['song_hz']
save_dir = config['preprocess']['save_dir']
cqt_layer = Spectrogram.CQT(device=device, sr=sr, hop_length=hop_size, fmin=220, fmax=None, n_bins=108, bins_per_octave=24, norm=1, window='hann', center=True, pad_mode='reflect')
p = Preprocess(sr, hop_size, song_hz, window_size, save_dir, aug, cqt_layer)

num_epochs = config['model'].get('num_epochs')

def get_data():
    datasets = {
        "isophonics-beetles" : {
            "mp3": config['preprocess']['data_path'] + "/beetles_albums",
            "labels": config['preprocess']['data_path'] + "/beetles_annotations"
        },
        "isophonics-king" : {
            "mp3": config['preprocess']['data_path'] + "/carol_king_albums",
            "labels": config['preprocess']['data_path'] + "/carol_king_annotations"
        }
    }

    king_albums = p.get_mp3(datasets['isophonics-king']['mp3'])
    king_labels = p.get_labels(datasets['isophonics-king']['labels'])
    beetles_albums = p.get_mp3(datasets['isophonics-beetles']['mp3'])
    beetles_labels = p.get_labels(datasets['isophonics-beetles']['labels'])

    data = [
        (king_albums, king_labels),
        (beetles_albums, beetles_labels)
    ]
    return data

def generate_chords_and_features(data):
    augment_fns = [
        (lambda x, sr : x, lambda l : l, ""),
        (aug.get_stretched_audio, lambda l : l, "_stretched")
    ]
    for i in range(1, 13):
        augment_fns.append((aug.get_augment_pitch_audio(i), aug.get_augment_pitch_chords(i), str(i) + "_pitched"))
    for d in data:
        album_label_dict = {}
        albums_dict = d[0]
        l_dict = d[1]
        for label_path in l_dict:
            song_label_dict = p.generate_song_labels(label_path, l_dict)
            album_title = p.path_to_album(label_path)
            album_label_dict[album_title] = song_label_dict
        for fn1, fn2, extension in augment_fns:
            p.generate_features(albums_dict, album_label_dict, extension, fn1, fn2)

if not config['preprocess'].get("preprocessed"):
    d = get_data()
    generate_chords_and_features(d)

dataset = os.listdir(save_dir)

model = BTC_model(config=config['model']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print("Loading Data...")
train_set, test_set = ChordDataset(save_dir, train_dataset), ChordDataset(save_dir, test_dataset) 
train_dataloader = ChordDataloader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_dataloader = ChordDataloader(test_set, batch_size=128, shuffle=True, num_workers=0)
print("Data loaded!")

for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0.0
    print("epoch: " + str(epoch))
#     Training
    print(" Training...")
    remaining = train_size
    for i_batch, data in enumerate(train_dataloader): 
        if i_batch % 10 == 0:
            print(" Number of samples remaining: " + str(remaining))
        features, chords = data
        features.requires_grad = True
        
        optimizer.zero_grad()
        chords = chords.to(device)
        # Train
        prediction, total_loss, weights, second = model(features, chords)
        
        running_loss += total_loss.item()
        
        if i_batch % 100 == 99:
            print("  batch: " + str(i_batch))
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i_batch + 1, running_loss / 100))
            running_loss = 0.0
        
        total_loss.backward()
        optimizer.step()
        
        remaining -= 128
# Validation 

    print("Done training!  Validation:")

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for i, data in enumerate(test_dataloader):
            val_features, val_chords = data
            val_features.requires_grad = True
            
            optimizer.zero_grad()
            val_features = features.to(device)
            val_chords = chords.to(device)
            # Train
            val_prediction, val_loss, weights, val_second = model(val_features, val_chords)
            total += val_prediction.size(0)
            correct += (val_prediction.view(val_chords.size(0), 108) == val_chords).sum().item()
        result = (100 * correct / total)
        print("Validation result: %" + str(result) )
    file_name = config['model'].get("output_dir") + "model-epoch-" + str(epoch)
    model_obj = {"model": model.state_dict(), 'optimizer': optimizer.state_dict(), "epoch": epoch}
    torch.save(model_obj, file_name)
####
import IPython.display as ipd
import numpy as np
import pandas as pd
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import sys
import preprocess
import re
import yaml

from scipy.io import wavfile as wav
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.regularizers import l1, l2, l1_l2

config = yaml.load(open("./config/config.yaml"))

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

def extract_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    return mfccs_processed

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

def filename_to_title(filename):
    name = re.sub(r'\([^)]*\)', '', filename)
    new_name = ""
    for character in name:
        if character == '.':
            break
        if character.isalnum() and not character.isnumeric():
            new_name += character
    return new_name

def path_to_album(path):
    return os.path.basename(os.path.normpath(path))

def generate_song_labels(label_album_path):
    song_label_dict = {}
    file_labels = labels_dict[label_album_path]
    for file in file_labels:
        if not file['filename'].endswith('.lab'):
            continue
        song_label_dict[file['title']] = []
        with open(os.path.join(label_album_path, file['filename'])) as fp:
            line = fp.readline()
            while line:
                tokens = line.split(' ')
                if len(tokens) == 1: tokens = line.split('\t')
                onset = int(float(tokens[0]))
                offset = int(float(tokens[1]))
                chord = tokens[2][:len(tokens[2]) - 1]
                song_label_dict[file['title']].append((onset, offset, chord))
                line = fp.readline()
    return song_label_dict

def generate_features(albums_dict, album_label_dict):
    features = []
    counter = 0
    for album in albums_dict:
        album_title = path_to_album(album)
        for song in albums_dict[album]:
            counter += 1
            song_path = os.path.join(album, song["filename"])
            song_title = filename_to_title(song["filename"])
            print(str(counter) +"th song: " + song_title)
            data, sr = librosa.load(song_path)
            if album_title in album_label_dict:
                if song_title in album_label_dict[album_title]:
                    for intervals in album_label_dict[album_title][song_title]:
                        start, end, chord = intervals[0], intervals[1], intervals[2]
                        if end > start:
                            start_index = librosa.time_to_samples(start)
                            end_index = librosa.time_to_samples(end)
                            audio_slice = data[int(start_index):int(end_index)]
                            if len(audio_slice) == 0:
                                continue
                            mfccs = librosa.feature.mfcc(y=audio_slice, sr=sample_rate, n_mfcc=40)
                            mfccs_processed = np.mean(mfccs.T,axis=0)
                            features.append([mfccs_processed,  chord])
    return features

def get_all_features():
    features = []
    for d in data:
        album_label_dict = {}
        albums_dict = d[0]
        labels_dict = d[1]
        for label_path in labels_dict:
            song_label_dict = generate_song_labels(label_path)
            album_title = path_to_album(label_path)
            album_label_dict[album_title] = song_label_dict
        f = generate_features(albums_dict, album_label_dict)
        features += f
    return features
featuresdf = pd.DataFrame(features, columns=['feature','chord_label'])
featuresdf.head()
X = np.array(featuresdf.feature.tolist())
Y = np.array(featuresdf.chord_label.tolist())

le = LabelEncoder()
yy = to_categorical(le.fit_transform(Y))
# Split Dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.50, random_state = 127)
#Feedforward NN 
num_labels = yy.shape[1]
filter_size = 2
def build_model_graph(input_shape=(40,)):
    model = Sequential()
    model.add(Dense(256, input_shape=input_shape, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=l2(1e-4),
        activity_regularizer=l2(1e-5)))
    model.add(Activation('relu'))
    model.add(Dense(256, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=l2(1e-4),
        activity_regularizer=l2(1e-5)))
    model.add(Activation('relu'))
    model.add(Dense(num_labels, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=l2(1e-4),
        activity_regularizer=l2(1e-5)))
    model.add(Activation('softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model
model = build_model_graph()
# Display model architecture summary 
model.summary()
# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]
print("Pre-training accuracy: %.4f%%" % accuracy)
num_epochs = 100
num_batch_size = 32
model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)
# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))
score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: {0:.2%}".format(score[1]))
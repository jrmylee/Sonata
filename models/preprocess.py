import os
import re
import numpy as np
import librosa
import torch

if torch.cuda.is_available():
    print("Using GPU")
    device = "cuda:0"
else:
    print("Using CPU")
    device = "cpu"

class Preprocess():
    def __init__(self, sample_rate, hop_size, song_hz, window_size, save_dir, augmenter, cqt_layer):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.song_hz = song_hz
        self.window_size  = window_size
        self.hop_interval = hop_size / song_hz
        self.get_num_samples = lambda x : x / self.hop_interval
        self.augmenter = augmenter
        self.save_dir = save_dir
        self.cqt_layer = cqt_layer
        self.songs_memo = {}

    def get_files(self, directory):
        files = {}
        for dirpath, dirnames, filenames in os.walk(directory):
            if not dirnames:
                for filename in filenames:
                    new_name = self.filename_to_title(filename)
                    if not filename.endswith('.lab') and not filename.endswith('.mp3') and not filename.endswith('m4a'):
                        continue
                    song_obj = {
                        "filename": filename,
                        "title": new_name
                    }
                    if dirpath not in files:
                        files[dirpath] = [song_obj]
                    else:
                        files[dirpath].append(song_obj)
        return files
    
    def get_mp3(self, directory):
        return self.get_files(directory)
    
    def get_labels(self, directory):
        return self.get_files(directory)

    def filename_to_title(self, filename):
        name = re.sub(r'\([^)]*\)', '', filename)
        new_name = ""
        for character in name:
            if character == '.':
                break
            if character.isalnum() and not character.isnumeric():
                new_name += character
        return new_name

    def path_to_album(self, path):
        return os.path.basename(os.path.normpath(path))

    def generate_song_labels(self, label_album_path, labels_dict):
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

    def get_chords_in_interval(self, audio, chord_intervals, interval):
        start_index = librosa.time_to_samples(interval[0])
        end_index = librosa.time_to_samples(interval[1])
        audio_slice = audio[int(start_index):int(end_index)]
        ref_start, ref_end = interval[0], interval[1]
        
        chords = []
        curr_interval = chord_intervals[0]
        index = 0
        while curr_interval[0] < ref_end and index < len(chord_intervals):
            curr_interval = chord_intervals[index]
            if curr_interval[1] > ref_start:
                chords.append(curr_interval[2])
            index += 1
        return audio_slice, chords


    def get_chord_at_time(self, chord_intervals, time):        
        for interval in chord_intervals:
            start, end = interval[0], interval[1]
            if start <= time and end >= time:
                return interval[2]
        return "N"

    def get_mfcc(self, audio, sample_rate):
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=144)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
        
    def get_cqt(self, audio, sample_rate):
        return librosa.cqt(audio, sr=sample_rate, n_bins=144, bins_per_octave=24, hop_length=2048)

    def get_start_end_indices(self, start_time, end_time):
        start_index = librosa.time_to_samples(start_time)
        end_index = librosa.time_to_samples(end_time)
        return start_index, end_index

    # def generate_song_paths(self, albums_dict, album_label_dict, augment_audio_fn, augment_chords_fn):
    #     for album in albums_dict:
    #         album_title = self.path_to_album(album)
    #         for song in albums_dict[album]:
    #             song_path = os.path.join(album, song["filename"])
    #             song_title = self.filename_to_title(song["filename"])
    #             song_save_path = self.save_dir + song_title + file_extension 
    #             if album_title in album_label_dict:
    #                 if song_title in album_label_dict[album_title]:
    #                     self.song_paths.append(song_path)
    # def generate_song_features(self, song_path, song_title):
    #     if song_title in self.songs_memo:
    #         data, sr = self.songs_memo[song_title]
    #     else:
    #         data, sr = librosa.load(song_path)
    #         self.songs_memo[song_title] = (data, sr)
    #     data = augment_audio_fn(data, sr)

    #     curr_start_time = 0
    #     total_duration = librosa.get_duration(y=data, sr=sr)
    #     intervals = album_label_dict[album_title][song_title]
    #     i = 0
    #     while curr_start_time + self.window_size < total_duration:
    #         if not os.path.exists(song_save_path + str(i) + ".pth"):
    #             curr_sec = curr_start_time
    #             curr_chords = [] # chords in the time frame
    #             while curr_sec < curr_start_time + self.window_size:
    #                 chord = self.get_chord_at_time(intervals, curr_sec)
    #                 curr_sec += self.hop_interval
    #                 curr_chords.append(chord)
    #             start_index, end_index = self.get_start_end_indices(curr_start_time, curr_start_time+self.window_size)
    #             audio_slice = data[int(start_index):int(end_index)]
                
    #             curr_chords = augment_chords_fn(curr_chords)
    #             audio_slice = torch.tensor(audio_slice).to(device).float()

    #             if len(audio_slice) != 0:
    #                 features = self.cqt_layer(audio_slice)
    #                 save_obj = {
    #                     "feature": features,
    #                     "chords": curr_chords
    #                 }
    #                 torch.save(save_obj, song_save_path + str(i) + ".pth")
    #         curr_start_time += self.hop_interval
    #         i += 1
    def generate_features(self, albums_dict, album_label_dict, file_extension, augment_audio_fn, augment_chords_fn):
        counter = 0
        for album in albums_dict:
            album_title = self.path_to_album(album)
            for song in albums_dict[album]:
                counter += 1
                song_path = os.path.join(album, song["filename"])
                song_title = self.filename_to_title(song["filename"])
                song_save_path = self.save_dir + song_title + file_extension
                if album_title in album_label_dict:
                    if song_title in album_label_dict[album_title]:
                        if song_title in self.songs_memo:
                            data, sr = self.songs_memo[song_title]
                        else:
                            data, sr = librosa.load(song_path)
                            self.songs_memo[song_title] = (data, sr)
                        data = augment_audio_fn(data, sr)

                        curr_start_time = 0
                        total_duration = librosa.get_duration(y=data, sr=sr)
                        intervals = album_label_dict[album_title][song_title]
                        i = 0
                        while curr_start_time + self.window_size < total_duration:
                            if not os.path.exists(song_save_path + str(i) + ".pth"):
                                print("window for " + song_title + " " + str(i) + " does not exist!")
                                curr_sec = curr_start_time
                                curr_chords = [] # chords in the time frame
                                while curr_sec < curr_start_time + self.window_size:
                                    chord = self.get_chord_at_time(intervals, curr_sec)
                                    curr_sec += self.hop_interval
                                    curr_chords.append(chord)
                                start_index, end_index = self.get_start_end_indices(curr_start_time, curr_start_time+self.window_size)
                                audio_slice = data[int(start_index):int(end_index)]
                                
                                curr_chords = augment_chords_fn(curr_chords)
                                audio_slice = torch.tensor(audio_slice).to(device).float()

                                if len(audio_slice) != 0:
                                    features = self.cqt_layer(audio_slice)
                                    save_obj = {
                                        "feature": features,
                                        "chords": curr_chords
                                    }
                                    torch.save(save_obj, song_save_path + str(i) + ".pth")
                            curr_start_time += self.hop_interval
                            i += 1
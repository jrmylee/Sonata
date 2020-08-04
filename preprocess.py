import os
import re
import numpy as np
import librosa
from comet_ml import Experiment

class Preprocess():
    # Returns 
    def filename_to_title(self, filename):
        name = re.sub(r'\([^)]*\)', '', filename)
        new_name = ""
        for character in name:
            if character == '.':
                break
            if character.isalnum() and not character.isnumeric():
                new_name += character
        return new_name
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

    def generate_labels_features(self, mp3s):
        Fs = 22050
        for album_path in mp3s:
            for song_obj in mp3s[album_path]:
                original_wav, sr = librosa.load(os.path.join(album_path, song_obj["filename"]), sr=Fs)
                N = 4096
                H = 2205
                gamma = 100
                norm_p = '2'

                # Compute chroma features with elliptic filter bank
                P = librosa.iirt(y=original_wav, sr=Fs, win_length=N, hop_length=H, center=True, tuning=0.0)
                P_compressed = np.log(1.0 + gamma * P)
                C_nonorm = librosa.feature.chroma_cqt(C=P_compressed, bins_per_octave=12, n_octaves=7, fmin=librosa.midi_to_hz(24), norm=None)
                print(C_nonorm)
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

                                if chord != "N":
                                    pitched, pitched_label = augment_pitch(audio_slice, sample_rate, chord)
                                    features.append([pitched, pitched_label])

                                stretch_noised = augment_stretched_noise(audio_slice, sample_rate, chord)
                                features.append([stretch_noised, chord])
                                
                                noised = augment_stretched_noise(audio_slice, sample_rate, chord, True, False)
                                features.append([noised, chord])
                                
                                stretched = augment_stretched_noise(audio_slice, sample_rate, chord, False, True)
                                features.append([stretched, chord])
        return features



# if __name__ == "__main__":
#     p = Preprocess()
#     mp3_files = p.get_mp3()
#     p.generate_labels_features(mp3_files)

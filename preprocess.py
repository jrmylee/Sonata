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
    # def process(self):

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



# if __name__ == "__main__":
#     p = Preprocess()
#     mp3_files = p.get_mp3()
#     p.generate_labels_features(mp3_files)

import chords
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

class Transform():
    def __init__():
        this.c = chords.Chords()

    def augment_pitch(data, sr, label):
        semitone = random.randint(1, 12)
        aug_chord = c.shift(semitone, data, sr,label)
        mfccs = librosa.feature.mfcc(y=aug_chord[0], sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T,axis=0)
        
        return mfccs_processed, aug_chord[1]

    def augment_stretched_noise(data, sr, label, noise=True, stretch=True):
        composition = []
        if noise:
            composition.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5))
        if stretch:
            composition.append(TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5))
        augmenter = Compose(composition)
        
        aug_chord = augmenter(samples=data, sample_rate=sr)
        mfccs = librosa.feature.mfcc(y=aug_chord, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T,axis=0)
        
        return mfccs_processed
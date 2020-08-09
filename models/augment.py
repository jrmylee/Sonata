import random
import librosa
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

class Augment():
    def __init__(self, chords):
        self.chords = chords
    def augment_pitch(self, data, sr, labels):
        semitone = random.randint(1, 12)
        aug_audio = self.chords.shift_audio(semitone, data, sr)
        for i in range(len(labels)):
            if labels[i] == "N":
                return [], []
            labels[i] = self.chords.shift_label(semitone, labels[i])
        return aug_audio, labels

    def augment_stretched_noise(self, data, sr, noise=True, stretch=True):
        composition = []
        if noise:
            composition.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5))
        if stretch:
            composition.append(TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5))
        augmenter = Compose(composition)
        
        aug_chord = augmenter(samples=data, sample_rate=sr)
        
        return aug_chord
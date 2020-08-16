import random
import librosa
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

class Augment():
    def __init__(self, chords):
        self.chords = chords
    def get_augment_pitch_fn(self, semitone):
        def augment_pitch(data, sr, labels):
            aug_audio = self.chords.shift_audio(semitone, data, sr)
            for i in range(len(labels)):
                if labels[i] == "N":
                    return [], []
                labels[i] = self.chords.shift_label(semitone, labels[i])
            return aug_audio, labels
        return augment_pitch

    def augment_stretched_noise(self, data, sr, labels):
        composition = []
        composition.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5))
        augmenter = Compose(composition)
        
        aug_chord = augmenter(samples=data, sample_rate=sr)
        
        return aug_chord, labels
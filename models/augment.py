import random
import librosa
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

class Augment():
    def __init__(self, chords):
        self.chords = chords
    
    def get_augment_pitch_audio(self, semitone):
        return lambda x, sr : self.chords.shift_audio(semitone, x, sr)

    def get_augment_pitch_chords(self, semitone):
        def augment_chords(labels):
            for i in range(len(labels)):
                if labels[i] == "N":
                    return [], []
                labels[i] = self.chords.shift_label(semitone, labels[i])
            return labels
        return augment_chords

    def get_stretched_audio(self, x, sr):
        composition = []
        composition.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5))
        augmenter = Compose(composition)
        
        aug_chord = augmenter(samples=x, sample_rate=sr)
        
        return aug_chord
    
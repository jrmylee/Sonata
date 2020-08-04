from music21 import *
import librosa

class Chords:
    def __init__(self):
        self.interval_mapping = {
            1 : 'm2',
            2: 'M2',
            3: 'm3',
            4: 'M3',
            5: 'P4',
            6: 'd5',
            7: 'P5',
            8: 'm6',
            9: 'M6',
            10: 'm7',
            11: 'M7',
            12: 'P8'
        }
    def shift(self, semitones, data, sr, label):
        tonic = label[0:1]
        n = note.Note(tonic)
        new_data = librosa.effects.pitch_shift(data, sr, n_steps=semitones)
        return (new_data, n.transpose(self.interval_mapping[semitones]).name + label[1:])
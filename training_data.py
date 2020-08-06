import yaml
from models.preprocess import Preprocess
from models.augment import Augment
from models.chords import Chords

config = yaml.load(open("./config/config.yaml"))
sr = config['preprocess']['sample_rate']
hop_size = config['preprocess']['hop_size']
window_size = config['preprocess']['window_size']
song_hz = config['preprocess']['song_hz']

p = Preprocess(sr, hop_size, song_hz, window_size, Augment(Chords()))

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

def get_chords_and_features(data):
    features, chords = [], []
    for d in data:
        album_label_dict = {}
        albums_dict = d[0]
        l_dict = d[1]
        for label_path in l_dict:
            song_label_dict = p.generate_song_labels(label_path, l_dict)
            album_title = p.path_to_album(label_path)
            album_label_dict[album_title] = song_label_dict
        f, c = p.generate_features(albums_dict, album_label_dict)
        features.append(f)
        chords.append(c)
    return features, chords

d = get_data()
features, chords = get_chords_and_features(d)
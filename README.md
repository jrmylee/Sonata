# Sonata

Project for chord recognition using deep RNNs.  Designed to be used with the Isophonics Dataset(http://isophonics.net/),
but can be used with any dataset with labeled chords corresponding to a time interval.  

Note: Datasets do not come with corresponding audio due to copyright concerns.

Setup:
1. Clone Dataset
2. Clone BTC Model
3. add config/config.yaml with relevant paths
4. mkdir checkpoints
5. run train.py

Example config(config/config.yaml):
```
model: 
  path: #PATH TO MODEL
  feature_size : 108
  timestep : 108
  num_chords : 133
  input_dropout : 0.2
  layer_dropout : 0.2
  attention_dropout : 0.2
  relu_dropout : 0.2
  num_layers : 8
  num_heads : 4
  hidden_size : 128
  total_key_depth : 128
  total_value_depth : 128
  filter_size : 128
  loss : 'ce'
  probs_out : False
  sample_rate: 22050
preprocess:
  data_path: #PATH TO DATASET
  sample_rate: 22050
  hop_size: 2048
  window_size: 10
  song_hz: 22050
```
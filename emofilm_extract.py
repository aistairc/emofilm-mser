# filename: emofilm_extract.py
# script to extract acoustic embedding form EmoFilm dataset
# input_dir: /data/EmoFilm/wav_corpus_16k
# output_dir: ./cache/w2v2.pkl 
# coded by b-atmaja@aist.go.jp
# read the readme BEFORE running this script
# changelog:
# 2023-03-10: initial version


import os
import audinterface
import audonnx
import pandas as pd

model_root = "/data/models/w2v2-L-robust/"

model = audonnx.load(model_root)

# read annotation data
data = pd.read_csv('./EmoFilm_labels_v2.csv')
files = data['file'].values
emo = data['emo'].values

# for debugging
# print(files)
# print(emo)

# feature extraction
hidden_states = audinterface.Feature(
    model.outputs['hidden_states'].labels,
    process_func=model,
    process_func_args={
        'output_names': 'hidden_states',
    },
    sampling_rate=16000,    
    resample=True,    
    num_workers=5,
    verbose=True,
)

cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)


def cache_path(file):
    return os.path.join(cache_dir, file)


path = cache_path('w2v2.pkl')
if not os.path.exists(path):
    features_w2v2 = hidden_states.process_files(
        files=files,
        root= '/data/EmoFilm/wav_corpus_16k/',
    )
    features_w2v2.to_pickle(path)
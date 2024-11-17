import torch
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm

with open('rank_20.pkl', 'rb') as f:
    rank = pickle.load(f)

d = pd.read_csv('waterbird_complete95_forest2water2/metadata.csv')

indices = []
for i in tqdm(range(d.shape[0])):
    if d.loc[i, 'img_id'] in rank:
        indices.append(i)
print(len(indices))
d = d.drop(index = indices)

d.to_csv('new_metadata_95.csv')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
import tqdm
import pickle
from collections import defaultdict

with open('0el2n.pkl', 'rb') as f:
    scores = pickle.load(f)

scores.sort(key = lambda x:-x[1])

all_ind = []

all_0 = []
all_0o = []
all_1 = []
all_1o = []

all_other = []

counter_0 = 0
counter_1 = 0

for i, j, k, l in scores:
    if k == 0 and l == 0:
        all_0.append(i)
    if k == 0 and l == 1:
        all_0o.append(i)
    if k == 1 and l == 1:
        all_1.append(i)
    if k == 1 and l == 0:
        all_1o.append(i)

l0 = len(all_0)
l0o = len(all_0o)
l1 = len(all_1)
l1o = len(all_1o)

add_remove1 = (1 - (int(l1*0.75) + l1o)/(int(l0*0.75) + l0o)) * l0*0.75

all_ind = all_0[:int(l0*0.25) + int(add_remove1) + 128] + all_1[:int(l1*0.25)]

print(len(all_ind))

with open('rank_20.pkl', 'wb') as f:
    pickle.dump(all_ind, f)

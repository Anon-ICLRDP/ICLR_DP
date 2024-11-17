import pickle

import torch
import torchvision
from torchvision import transforms
from PIL import Image

import random
import numpy as np
import pandas as pd
import os

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset

with open('20el2n.pkl', 'rb') as f:
    x = pickle.load(f)


x.sort(key = lambda x:-x[1])

all_ = [250, 500, 750, 900, 1000]

for curr in all_:
    all_ranks = []
    counter = 0
    for i, j, k, l in x:
        if i == 4:
            if counter < curr:
                all_ranks.append(l)
            else:
                pass

            counter += 1

    print(sum(all_ranks)/len(all_ranks))

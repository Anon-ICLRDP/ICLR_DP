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

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

with open('2el2n.pkl', 'rb') as f:
    x = pickle.load(f)

train_loader = torch.load('train_loader.pth')

x.sort(key = lambda x:x[1])

all_ = []
all_ranks = []
for i, j, k, l in x[int(len(x)*10//100):]:
    all_.append(k)
    all_ranks.append(l)

new_images = []
new_labels = []
for batch_idx, (images, labels, index, rank) in enumerate(train_loader):
    for j in range(len(labels)):
        if index[j].item() in all_:
            new_images.append(images[j])
            new_labels.append(labels[j])


new_dataset = []
for j in range(len(new_images)):
    new_dataset.append([new_images[j], new_labels[j]])

train_loader = torch.utils.data.DataLoader(new_dataset, shuffle = True, batch_size = 128)

print(len(train_loader))
print(len(train_loader.dataset))

torch.save(train_loader, 'train_loader_smol_el2n.pth')

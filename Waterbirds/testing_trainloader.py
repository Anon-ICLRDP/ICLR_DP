import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
import tqdm
import pickle
from collections import defaultdict

t1 = torch.load('train_loader.pth')
model = torch.load('network.pt').cuda()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
with open('9el2n.pkl', 'rb') as f:
    scores = pickle.load(f)

d = defaultdict(float)
for ind, el2n, label in scores:
    d[ind] = el2n

model.layer4[2].conv3.register_forward_hook(get_activation('conv1'))

correct_0 = []
incorrect_0_repr = 0 
count_0 = 0
correct_1 = []
incorrect_1_repr = 0
count_1 = 0

for idx, batch in enumerate(t1):
    x, y, indi = batch[0].cuda(), batch[1].cuda(), batch[-1].cuda()
    logits = model(x)
    preds = torch.argmax(logits, axis=1)
    for j in range(len(preds)):
        if preds[j].item() == y[j].item():
            if y[j].item() == 1:
                correct_1.append([x[j], activation['conv1'][j], indi[j], d[indi[j]]])
            else:
                correct_0.append([x[j], activation['conv1'][j], indi[j], d[indi[j]]])

correct_0.sort(key = lambda x:-x[-1])
correct_1.sort(key = lambda x:-x[-1])

rank00 = []
for i, j, k, l in correct_0:
    rank00.append(k)

rank11 = []
for i, j, k, l in correct_1:
    rank11.append(k)

with open('rank00_2.pkl', 'wb') as f:
    print(len(rank00[:int(len(correct_0)*0.9)]))
    pickle.dump(rank00[:int(len(correct_0)*0.9)], f)
with open('rank11_2.pkl', 'wb') as f:
    pickle.dump(rank11[:int(len(correct_1)*0.9)], f)

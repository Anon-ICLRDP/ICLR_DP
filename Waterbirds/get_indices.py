import torch
import numpy as np
import pandas as pd
import os
import tqdm
import torch.nn.functional as F
import pickle

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = torch.load('network.pt')
train_loader = torch.load('train_loader.pth')
model.eval()

model.layer4[1].conv2.register_forward_hook(get_activation('conv1'))

g11 = []
g01 = []
g10 = []
g00 = []

for batch in tqdm.tqdm(train_loader):
    x, y, g, p, i = batch
    x, y, p = x.cuda(device=1), y.cuda(device=1), p.cuda(device=1)
    output = model(x)
    for j in range(len(p)):
        if y[j].item() == 0 and p[j].item() == 0:
            g00.append((activation['conv1'][j], i[j]))
        elif y[j].item() == 0 and p[j].item() == 1:
            g01.append((activation['conv1'][j], i[j]))
        elif y[j].item() == 1 and p[j].item() == 0:
            g10.append((activation['conv1'][j], i[j]))
        elif y[j].item() == 1 and p[j].item() == 1:
            g11.append((activation['conv1'][j], i[j]))

# rank based on cs

score = []

for activation1, ind in g00:
    sc1 = 0
    for activation2, ind2 in g10:
        sc1 += F.cosine_similarity(activation1.flatten(), activation2.flatten(), dim=-1).item()
    sc1 /= len(g10)
    sc2 = 0
    for activation2, ind2 in g01:
        sc2 += F.cosine_similarity(activation1.flatten(), activation2.flatten(), dim=-1).item()

    sc2 /= len(g01)
    score.append(((sc2 - sc1), ind))

score.sort(key = lambda x:x[0])
rank00 = []

for i, j in score:
    rank00.append(j)

score = []

for activation1, ind in g11:
    sc1 = 0
    for activation2, ind2 in g01:
        sc1 += F.cosine_similarity(activation1.flatten(), activation2.flatten(), dim=-1).item()
    sc1 /= len(g10)
    sc2 = 0
    for activation2, ind2 in g10:
        sc2 += F.cosine_similarity(activation1.flatten(), activation2.flatten(), dim=-1).item()

    sc2 /= len(g01)
    score.append(((sc2 - sc1), ind))

score.sort(key = lambda x:x[0])
rank11 = []

for i, j in score:
    rank11.append(j)

with open('rank00.pkl', 'wb') as f:
    pickle.dump(rank00[:1500], f)
with open('rank11.pkl', 'wb') as f:
    pickle.dump(rank11[:500], f)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import sys

from collections import defaultdict
from collections import Counter

t1 = torch.load('trainloader.pt')
model = torch.load('4_model.pt').cuda()
model.eval()
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

print('s1')
# All indices of minority and majority based on el2n sampling

d = defaultdict()
with open('multinli/data/rank_20_other1.pkl', 'rb') as f:
    majority_indices_0, majority_indices_1, majority_indices_2 = pickle.load(f)
with open('multinli/data/rank_20_other2.pkl', 'rb') as f:
    minority_indices_0, minority_indices_1, minority_indices_2 = pickle.load(f)

print(len(majority_indices_0), len(majority_indices_1), len(majority_indices_2), len(minority_indices_0), len(minority_indices_1), len(minority_indices_2))
for j in majority_indices_0:
    d[j] = 1
for j in majority_indices_1:
    d[j] = 1
for j in majority_indices_2:
    d[j] = 1
for j in minority_indices_0:
    d[j] = 0
for j in minority_indices_1:
    d[j] = 0
for j in minority_indices_2:
    d[j] = 0
print('s1')

counter = 0

#model.layer4[2].conv3.register_forward_hook(get_activation('conv1'))

model.bert.pooler.activation.register_forward_hook(get_activation('conv1'))

minority_0 = []
majority_0_repr = 0
count_0 = 0
minority_1 = []
majority_1_repr = 0
count_1 = 0
minority_2 = []
majority_2_repr = 0
count_2 = 0
counter = 0
counter_other = 0
other_counter = 0
other_other_counter = 0
other_other_other_counter = 0
for idx, batch in tqdm(enumerate(t1)):
    other_other_other_counter += 1
    x, y, indi = batch[0].cuda(), batch[1].cuda(), batch[-1].cuda()
    input_ids = x[:, :, 0]
    input_masks = x[:, :, 1]
    segment_ids = x[:, :, 2]
    outputs = model(
        input_ids=input_ids,
        attention_mask=input_masks,
        token_type_ids=segment_ids,
        labels=y
    )[1]
    #preds = torch.argmax(logits, axis=1)
    for j in range(len(y)):
        other_other_counter += 1
        if indi[j].item() in d:
            if not d[indi[j].item()]:
                counter += 1
                if y[j].item() == 2:
                    minority_2.append([x[j], activation['conv1'][j], indi[j]])
                elif y[j].item() == 1:
                    minority_1.append([x[j], activation['conv1'][j], indi[j]])
                else:
                    minority_0.append([x[j], activation['conv1'][j], indi[j]])
            else:
                counter_other += 1
                if y[j].item() == 2:
                    if count_1 == 0:
                        #print('Enters Once')
                        majority_2_repr = activation['conv1'][j]
                    else:
                        majority_2_repr += activation['conv1'][j]
                    count_2 += 1
                elif y[j].item() == 1:
                    if count_1 == 0:
                        #print('Enters Once')
                        majority_1_repr = activation['conv1'][j]
                    else:
                        majority_1_repr += activation['conv1'][j]
                    count_1 += 1
                else:
                    if count_0 == 0:
                        print('Enters Once')
                        majority_0_repr = activation['conv1'][j]
                    else:
                        majority_0_repr += activation['conv1'][j]
                    count_0 += 1
        else:
            other_counter += 1

print(counter)
print(counter_other)
print(other_counter)
print(other_other_counter)
print(other_other_other_counter)

print('s1')
majority_2_repr /= count_2
majority_1_repr /= count_1
majority_0_repr /= count_0

print(len(minority_1))
print(len(minority_0))
print(len(minority_2))

all_scores = []
for batch in tqdm(minority_1):
    act1, ind = batch[1], batch[2]
    sc1 = F.cosine_similarity(act1.flatten(), majority_1_repr.flatten(), dim=-1).item()
    sc2 = F.cosine_similarity(act1.flatten(), majority_0_repr.flatten(), dim=-1).item()
    sc3 = F.cosine_similarity(act1.flatten(), majority_2_repr.flatten(), dim=-1).item()
    all_scores.append(((sc3 + sc2) - sc1, ind))

all_scores.sort(key = lambda x:x[0])
print(len(all_scores))

#print(len(all_scores))
all_scores = all_scores[:10000]

all_majority = majority_indices_0 + majority_indices_1 + majority_indices_2
print('this', len(all_scores))
print(len(all_majority))
print('this', len(all_scores))
for i, j in all_scores:
    all_majority.append(j)
#print(len(all_majority))
print('s1')

all_scores = []
for batch in tqdm(minority_0):
    act1, ind = batch[1], batch[2]
    sc1 = F.cosine_similarity(act1.flatten(), majority_1_repr.flatten(), dim=-1).item()
    sc2 = F.cosine_similarity(act1.flatten(), majority_0_repr.flatten(), dim=-1).item()
    sc3 = F.cosine_similarity(act1.flatten(), majority_2_repr.flatten(), dim=-1).item()
    all_scores.append(((sc3 + sc1) - sc2, ind))

all_scores.sort(key = lambda x:x[0])

print('this', len(all_scores))
all_scores = all_scores[:10000]
print('this', len(all_scores))

for i, j in all_scores:
    all_majority.append(j)

all_scores = []
for batch in tqdm(minority_2):
    act1, ind = batch[1], batch[2]
    sc1 = F.cosine_similarity(act1.flatten(), majority_1_repr.flatten(), dim=-1).item()
    sc2 = F.cosine_similarity(act1.flatten(), majority_0_repr.flatten(), dim=-1).item()
    sc3 = F.cosine_similarity(act1.flatten(), majority_2_repr.flatten(), dim=-1).item()
    all_scores.append(((sc1 + sc2) - sc3, ind))

all_scores.sort(key = lambda x:x[0])

print('this', len(all_scores))
all_scores = all_scores[:10000]
print('this', len(all_scores))

for i, j in all_scores:
    all_majority.append(j)
print('finaltoprune', len(all_majority))

with open('majority_indices_final.pkl', 'wb') as f:
    pickle.dump(all_majority, f)

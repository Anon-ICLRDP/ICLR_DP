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

scores.sort(key = lambda x:x[1])

print(len(scores))

spurious_dict = defaultdict(int)
nonspurious_dict = defaultdict(int)

'''for idx, (i, j, k, l) in enumerate(scores):
    if idx < len(scores)//10:
        if k == l:
            spurious_dict[1] += 1
        else:
            nonspurious_dict[1] += 1
    elif idx < len(scores)*2//10:
        if k == l:
            spurious_dict[2] += 1
        else:
            nonspurious_dict[2] += 1
    elif idx < len(scores)*3//10:
        if k == l:
            spurious_dict[3] += 1
        else:
            nonspurious_dict[3] += 1
    elif idx < len(scores)*4//10:
        if k == l:
            spurious_dict[4] += 1
        else:
            nonspurious_dict[4] += 1
    elif idx < len(scores)*5//10:
        if k == l:
            spurious_dict[5] += 1
        else:
            nonspurious_dict[5] += 1
    elif idx < len(scores)*6//10:
        if k == l:
            spurious_dict[6] += 1
        else:
            nonspurious_dict[6] += 1
    elif idx < len(scores)*7//10:
        if k == l:
            spurious_dict[7] += 1
        else:
            nonspurious_dict[7] += 1
    elif idx < len(scores)*8//10:
        if k == l:
            spurious_dict[8] += 1
        else:
            nonspurious_dict[8] += 1
    elif idx < len(scores)*9//10:
        if k == l:
            spurious_dict[9] += 1
        else:
            nonspurious_dict[9] += 1
    else:
        if k == l:
            spurious_dict[10] += 1
        else:
            nonspurious_dict[10] += 1'''

for idx, (i, j, k, l) in enumerate(scores):
    if idx < len(scores)//4:
        if k == l:
            spurious_dict[1] += 1
        else:
            nonspurious_dict[1] += 1
    elif idx < len(scores)*2//4:
        if k == l:
            spurious_dict[2] += 1
        else:
            nonspurious_dict[2] += 1
    elif idx < len(scores)*3//4:
        if k == l:
            spurious_dict[3] += 1
        else:
            nonspurious_dict[3] += 1
    elif idx < len(scores)*4//4:
        if k == l:
            spurious_dict[4] += 1
        else:
            nonspurious_dict[4] += 1

for i in spurious_dict:
    print(i, spurious_dict[i]/sum(spurious_dict.values()))

for i in nonspurious_dict:
    print(i, nonspurious_dict[i]/sum(nonspurious_dict.values()))

print(spurious_dict)
print(nonspurious_dict)

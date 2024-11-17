import torch
import pandas as pd
import numpy as np
import os
import random

d1 = pd.read_csv('other3.csv')
d2 = pd.read_csv('other2.csv')

print(d1.shape, d2.shape)
required = True

to_include1 = []
to_include2 = []
to_include3 = []
count1 = 0
count2 = 0
count3 = 0

for i in range(d1.shape[0]):
    if d2.loc[i, 'split'] != 0:
        to_include1.append(i)
        to_include2.append(i)
        to_include3.append(i)
    elif d1.loc[i, 'Male'] == 1 and d1.loc[i, 'Eyeglasses'] == -1 and count1 < 2500:
        to_include1.append(i)
        to_include2.append(i)
        to_include3.append(i)
        count1 += 1
    elif d1.loc[i, 'Male'] == 1 and d1.loc[i, 'Eyeglasses'] == 1:
        if count2 < 0:
            to_include1.append(i)
        if count2 < 2500:
            to_include2.append(i)
        if count2 < 5000:
            to_include3.append(i)
        count2 += 1
    elif d1.loc[i, 'Male'] == -1 and d1.loc[i, 'Eyeglasses'] == -1 and count3 < 5000:
        to_include1.append(i)
        to_include2.append(i)
        to_include3.append(i)
        count3 += 1
    elif d1.loc[i, 'Male'] == -1 and d1.loc[i, 'Eyeglasses'] == 1:
        to_include1.append(i)
        to_include2.append(i)
        to_include3.append(i)
        if random.random() > 0.5:
            d2.loc[i, 'split'] = 1
        else:
            d2.loc[i, 'split'] = 2

print(len(to_include1))
print(len(to_include2))
print(len(to_include3))

d1_new1 = d1.loc[to_include1]
d2_new1 = d2.loc[to_include1]

d1_new2 = d1.loc[to_include2]
d2_new2 = d2.loc[to_include2]

d1_new3 = d1.loc[to_include3]
d2_new3 = d2.loc[to_include3]

print(d1_new1.shape)
print(d2_new1.shape)
print(d1_new2.shape)
print(d2_new2.shape)
print(d1_new3.shape)
print(d2_new3.shape)
'''
d2_new1.to_csv('list_eval_partition1.csv')
d1_new1.to_csv('metadata1.csv')
'''
d2_new2.to_csv('list_eval_partition.csv')
d1_new2.to_csv('metadata.csv')
'''
d2_new3.to_csv('list_eval_partition3.csv')
d1_new3.to_csv('metadata3.csv')'''

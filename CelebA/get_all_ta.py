import torch
import numpy as np
import os
import pandas as pd

# Vanilla ERM

# Number of training samples misclassified final epoch

d1 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/model_outputs/train.csv')

to_access = d1.shape[0] - 1
print(d1.loc[to_access, 'avg_acc_group:0'])
print(d1.loc[to_access, 'avg_acc_group:1'])
print(d1.loc[to_access, 'avg_acc_group:2'])
print(d1.loc[to_access, 'avg_acc_group:3'])
print()

d1 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/model_outputs/test.csv')
d2 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/model_outputs/val.csv')

max_val = 0
max_epoch = 0
for i in range(d2.shape[0]):
    if d2.loc[i, 'avg_acc_group:1'] > max_val:
        max_val = d2.loc[i, 'avg_acc_group:1']
        max_epoch = d2.loc[i, 'epoch']

for i in range(d1.shape[0]):
    if d1.loc[i, 'epoch'] == max_epoch:
        print(d1.loc[i, 'avg_acc_group:0'])
        print(d1.loc[i, 'avg_acc_group:1'])
        print(d1.loc[i, 'avg_acc_group:2'])
        print(d1.loc[i, 'avg_acc_group:3'])
        break
print()
# After JTT

d1 = pd.read_csv('results/CelebA/CelebA_sample_exp/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch20/JTT_upweight_50_epochs_50_lr_1e-05_weight_decay_0.1/model_outputs/test.csv')
d2 = pd.read_csv('results/CelebA/CelebA_sample_exp/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch20/JTT_upweight_50_epochs_50_lr_1e-05_weight_decay_0.1/model_outputs/val.csv')

#d1 = pd.read_csv('results/CelebA/CelebA_sample_exp/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1_oneclass/final_epoch20/JTT_upweight_50_epochs_50_lr_1e-05_weight_decay_0.1/model_outputs/test.csv')
#d2 = pd.read_csv('results/CelebA/CelebA_sample_exp/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1_oneclass/final_epoch20/JTT_upweight_50_epochs_50_lr_1e-05_weight_decay_0.1/model_outputs/val.csv')

max_val = 0
max_epoch = 0
for i in range(d2.shape[0]):
    if d2.loc[i, 'avg_acc_group:1'] > max_val:
        max_val = d2.loc[i, 'avg_acc_group:1']
        max_epoch = d2.loc[i, 'epoch']

for i in range(d1.shape[0]):
    if d1.loc[i, 'epoch'] == max_epoch:
        print(d1.loc[i, 'avg_acc_group:0'])
        print(d1.loc[i, 'avg_acc_group:1'])
        print(d1.loc[i, 'avg_acc_group:2'])
        print(d1.loc[i, 'avg_acc_group:3'])
        break
print()

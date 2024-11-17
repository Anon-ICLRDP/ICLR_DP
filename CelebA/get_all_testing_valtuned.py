import torch
import numpy as np
import os
import pandas as pd

# Vanilla ERM

# Number of training samples misclassified final epoch

#d1 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_noprune/model_outputs/test.csv')
#d2 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_noprune/model_outputs/val.csv')
#d1 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_50prune/model_outputs/test.csv')
#d2 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_50prune/model_outputs/val.csv')
#d1 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_75prune/model_outputs/test.csv')
#d2 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_75prune/model_outputs/val.csv')
#d1 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_90prune/model_outputs/test.csv')
#d2 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_90prune/model_outputs/val.csv')
#d1 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_50prunen/model_outputs/test.csv')
#d2 = pd.read_csv('results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_50prunen/model_outputs/val.csv')
#d1 = pd.read_csv('results/CelebA/CelebA_sample_exp/rel2n/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_50prune/model_outputs/test.csv')
#d2 = pd.read_csv('results/CelebA/CelebA_sample_exp/rel2n/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_50prune/model_outputs/val.csv')
#main_path = 'results/CelebA/CelebA_sample_exp/el2n_oracle/ERM_upweight_0_epochs_10_lr_0.001_weight_decay_0.0001_'
#main_path2 = 'prune/model_outputs/'
main_path = 'results/CelebA/CelebA_sample_exp/el2n/ERM_upweight_0_epochs_25_lr_0.001_weight_decay_0.0001_'
main_path2 = 'prune/model_outputs/'

numbers = ['10', '25', '40', '50', '75', '90', '95', '97']
numbers = ['no']

all_1 = []
all_2 = []
for num in numbers:
    print(num)
    try:
        d1 = pd.read_csv(main_path + num + main_path2 + 'test.csv')
        d2 = pd.read_csv(main_path + num + main_path2 + 'val.csv')
    except:
        continue
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

    print('No Tuning')

    print(d1.loc[d1.shape[0] - 1, 'avg_acc_group:0'])
    print(d1.loc[d1.shape[0] - 1, 'avg_acc_group:1'])
    all_1.append(d1.loc[d1.shape[0] - 1, 'avg_acc_group:1'])
    print(d1.loc[d1.shape[0] - 1, 'avg_acc_group:2'])
    print(d1.loc[d1.shape[0] - 1, 'avg_acc_group:3'])
    all_2.append((d1.loc[d1.shape[0] - 1, 'avg_acc_group:2']*(2500/5000) + d1.loc[d1.shape[0] - 1, 'avg_acc_group:3']*(2500/5000)))
    print()
print(all_1)
print(all_2)

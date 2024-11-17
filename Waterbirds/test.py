import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import tqdm
import argparse
import sys
from collections import defaultdict
import json
from functools import partial
import pickle

from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data

from utils import MultiTaskHead
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p
from utils import update_dict, get_results, write_dict_to_tb

parser = argparse.ArgumentParser(description="Train model on waterbirds data")
# Data
with open('trainset.pkl', 'rb') as f:
    trainset = pickle.load(f)

get_yp_func = partial(get_y_p, n_places=trainset.n_places)
test_loader = torch.load('test_loader.pth')
train_loader = torch.load('train_loader.pth')

#models = ['outputs/ResNet18_best.pt', 'pruned_10_net.pt', 'pruned_20_net.pt', 'pruned_30_net.pt', 'pruned_40_net.pt', 'pruned_50_net.pt', 'pruned_60_net.pt', 'pruned_70_net.pt', 'pruned_80_net.pt', 'pruned_90_net.pt', 'pruned_95_net.pt', 'pruned_97_net.pt', 'pruned_99_net.pt']

models = ['network_rel2n2.pt']
for model in models:
    print(model)

    pruned_net = torch.load(model)
    pruned_net.cuda(device=1)

    counter = 0

    results = evaluate(pruned_net, test_loader, get_yp_func, False, False)
    print(results)
    print((3498/4795)*results['accuracy_0_0'] + (184/4795)*results['accuracy_0_1'] + (56/4795)*results['accuracy_1_0'] + (1057/4795)*results['accuracy_1_1'])

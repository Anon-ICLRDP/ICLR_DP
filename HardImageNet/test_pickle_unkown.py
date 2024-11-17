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

random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)

target_resolution = (224, 224)
scale = 256.0 / 224.0

trans_test = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

trans_train = transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, labels, indexs, ranks, transform=None):
        """
        Args:
            image_list (list): List of PIL images.
            labels (list): List of labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_list = image_list
        self.labels = labels
        self.ranks = ranks
        self.indexs = indexs
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.labels[idx]
        rank = self.ranks[idx]
        index = self.indexs[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, index, rank

class CustomImageDatasetTest(torch.utils.data.Dataset):
    def __init__(self, image_list, labels, transform=None):
        """
        Args:
            image_list (list): List of PIL images.
            labels (list): List of labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_list = image_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

with open('1el2n.pkl', 'rb') as f:
    x = pickle.load(f)

with open('pairs.pkl', 'rb') as f:
    all_pairs = pickle.load(f)

d_ = {}

for i, j in all_pairs:
    d_[j] = i

train_loader = torch.load('train_loader.pth')
print('og', len(train_loader.dataset))

x.sort(key = lambda x:-x[1])

all_ = []
all_ranks = []
counter = 0
class_counter = 0
class_counter_other = 0
d = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}
d_other = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}

for i, j, k, l in x:
    d_other[i] += 1

for i, j, k, l in x:
    if False:#d[i] < d_other[i]/40:
        d[i] += 1
    else:
        all_.append(k)
        all_ranks.append(l)
        if i == 4:
            class_counter += 1
        if i == 11:
            class_counter_other += 1

    counter += 1
        
print(counter)
print('cc', class_counter)
print('cco', class_counter_other)

new_images = []
new_labels = []
for batch_idx, (images, labels, index, rank) in enumerate(train_loader):
    for j in range(len(labels)):
        if index[j].item() in all_:
            new_images.append(d_[index[j].item()])
            new_labels.append(labels[j])

#new_dataset = []
#for j in range(len(new_images)):
#    new_dataset.append([new_images[j], new_labels[j]])

new_dataset = CustomImageDatasetTest(image_list=new_images, labels=new_labels, transform=trans_train)
train_loader_new = torch.utils.data.DataLoader(new_dataset, shuffle = True, batch_size = 128)

print(len(train_loader_new))
print('new', len(train_loader_new.dataset))

torch.save(train_loader_new, 'train_loader_smol_rel2n_nonoraclenone.pth')

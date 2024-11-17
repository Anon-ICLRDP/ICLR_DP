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

x.sort(key = lambda x:-x[1])

all_ = []
all_ranks = []
counter = 0
for i, j, k, l in x:
    if i == 11:
        if counter < 250:
            pass
        else:
            all_.append(k)
            all_ranks.append(l)

        counter += 1
    else:
        all_.append(k)
        all_ranks.append(l)

print(counter)
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
train_loader = torch.utils.data.DataLoader(new_dataset, shuffle = True, batch_size = 128)

print(len(train_loader))
print(len(train_loader.dataset))

torch.save(train_loader, 'train_loader_smol_rel2n_oracle.pth')

import torch
import torchvision
from torchvision import transforms
from PIL import Image

import random
import numpy as np
import pandas as pd
import os

from collections import defaultdict
import pickle

path_list_train = os.listdir('HardImageNet_Images/train')
path_list_val = os.listdir('HardImageNet_Images/val')

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

t1 = transforms.PILToTensor()
t2 = transforms.ToTensor()
t3 = transforms.ToPILImage()
t4 = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
t5 = transforms.Resize((224, 224))

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


with open('hardImageNet/meta/paths_by_rank.pkl', 'rb') as f:
    x = pickle.load(f)

with open('1el2n.pkl', 'rb') as f:
    y = pickle.load(f)
y.sort(key = lambda x:-x[1])

all_indices = defaultdict(list)

for i, j, k, l in y:
    all_indices[i].append(k)

all_to_skip = set()

for i in all_indices:
    for j in all_indices[i][:int(len(all_indices[i])//3)]:
        all_to_skip.add(j)

map_ = defaultdict(int)
for i in x:
    #print(i, x[i][0])
    map_[x[i][0].split('/')[1]] = i

#print(map_)
#print(type(x))
#print(1/0)

train_dataset = []
val_dataset = []
index_counter = 0
wtf_counter = 0

train_images = []
test_images = []

train_labels = []
test_labels = []

all_rank = []
index = []

image_index_pairs = []
all_list = [25473, 25440, 24698, 24493, 23919, 23675, 23027, 23029, 22686, 22351, 22159, 21039, 19583, 16165, 15489, 10906, 10515, 10471, 10400, 10065, 8396, 8387, 8179, 8079, 7993, 7973, 7918, 7693, 230, 263, 736, 745, 945, 991, 1074, 1244, 1348, 1583, 2088, 2299, 3169, 3264, 3444, 3541, 3937, 4186, 4201, 4603, 5428, 5773, 5963, 6119, 6459, 6663, 7033, 7038, 7188, 7212, 7321]
all_list = [str(i) for i in all_list]
other_counter = 0
other_other_counter = 0

for idx, c in enumerate(path_list_train):
    if c == 'n03218198':
        print('sled', idx)
    elif c == 'n04228054':
        print('ski', idx)
    rank = x[map_[c]]
    curr_images = os.listdir('HardImageNet_Images/train/' + c)
    for im in curr_images:
        image = Image.open('HardImageNet_Images/train/' + c + '/' + im)
        try:
            curr_rank = rank.index('train/' + c + '/' + im)
        except:
            continue
        if (c == 'n03218198' and im.split('_')[-1][:-5] not in all_list) or (c =='n04228054' and other_counter >= 100):
            continue
        else:
            if c == 'n04228054':
                other_counter += 1
        other = t1(image)
        if other.size()[0] == 1:
            other = other.expand(3, *other.shape[1:])
        try:
            #train_dataset.append((t4(t2(t5(t3(other)))), int(idx), int(index_counter), int(curr_rank)))
            if index_counter not in all_to_skip:
                #if True:
                train_images.append(t3(other))
                train_labels.append(int(idx))
                all_rank.append(int(curr_rank))
                index.append(int(index_counter))
                image_index_pairs.append([t3(other), index_counter])
            index_counter += 1
        except:
            print(other.size())
    curr_images = os.listdir('HardImageNet_Images/val/' + c)
    for im in curr_images:
        image = Image.open('HardImageNet_Images/val/' + c + '/' + im)
        other = t1(image)
        if other.size()[0] == 1:
            other = other.expand(3, *other.shape[1:])
        try:
            test_images.append(t3(other))
            test_labels.append(int(idx))
            #val_dataset.append((t4(t2(t5(t3(other)))), int(idx)))
        except:
            print(other.size())
            counter += 1


print(len(train_images))
print(len(train_labels))
print(len(index))
print(len(all_rank))

#train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, batch_size = 128)
#val_loader = torch.utils.data.DataLoader(val_dataset, shuffle = True, batch_size = 128)

train_dataset = CustomImageDataset(image_list=train_images, labels=train_labels, indexs=index, ranks=all_rank, transform = trans_train)
test_dataset = CustomImageDatasetTest(image_list=test_images, labels=test_labels, transform = trans_test)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, batch_size = 128)
val_loader = torch.utils.data.DataLoader(test_dataset, shuffle = True, batch_size = 128)

print(len(train_loader))
print(len(train_loader.dataset))
print(len(val_loader))
print(len(val_loader.dataset))

torch.save(train_loader, 'train_loader_pruned.pth')
torch.save(val_loader, 'val_loader_pruned.pth')

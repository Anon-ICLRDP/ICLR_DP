import argparse
import numpy as np
import pandas as pd
import os
import shutil
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.manual_seed(1)
torch.cuda.manual_seed(1)

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

# CIFAR 10 load

train_loader = torch.load('train_loader_20.pth')
val_loader = torch.load('val_loader_20.pth')

training_required = True

if training_required:

  unpruned_net = torchvision.models.resnet18(pretrained=True)
  d = unpruned_net.fc.in_features
  unpruned_net.fc = torch.nn.Linear(d, 15)
  #unpruned_net.cuda()
  unpruned_net = nn.DataParallel(unpruned_net.cuda())
  optimizer = optim.SGD(unpruned_net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
  criterion_CE = nn.CrossEntropyLoss()

  epochs = 50

  checkpoint_every = 1150

  best_val = 0.

  for epoch in range(1, epochs + 1, 1):
      unpruned_net.train()
      avg_loss = 0.
      training_acc = 0.
      val_acc = 0.
      testing_acc = 0.

      for batch_idx, (images, labels, _, rank) in tqdm(enumerate(train_loader)):
          images, labels = images.to(device), labels.to(device)
          optimizer.zero_grad()

          outputs = unpruned_net(images)
          _, predictions = torch.max(outputs.data, 1)
          training_acc += (predictions == labels).sum().item()

          loss = criterion_CE(outputs, labels)
          avg_loss += loss.item()
          loss.backward()

          optimizer.step()

      print(f'Epoch {epoch} -> Loss: {avg_loss/len(train_loader)}, Training Accuracy: {training_acc/len(train_loader.dataset)}')

      if epoch % checkpoint_every == 0 and epoch != 0:
          torch.save(unpruned_net, 'ResNet_epoch' + str(epoch) + '.pt')
      
      unpruned_net.eval()
      with torch.no_grad():
          for batch_idx, (test_images, test_labels) in enumerate(val_loader):
              test_images, test_labels = test_images.to(device), test_labels.to(device)

              test_outputs = unpruned_net(test_images)
              _, test_predictions = torch.max(test_outputs.data, 1)
              val_acc += (test_predictions == test_labels).sum().item()

          print('Val Accuracy: ', val_acc/len(val_loader.dataset))
          print()
          '''if epoch in {1, 10, 20}:
              counter = 0
              all_ = []
              for batch_idx, (images, labels, index, rank) in enumerate(train_loader):
                  images, labels = images.to(device), labels.to(device)
                  for j in range(len(labels)):
                      image = images[j].unsqueeze(0)
                      output = unpruned_net(image)[0]
                      probs = F.softmax(output).detach()
                      label = labels[j].item()
                      curr_index = index[j].item()
                      curr_rank = rank[j].item()
                      vector = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
                      vector[label] = 1
                      el2n = torch.norm(vector - probs)
                      all_.append([label, el2n, curr_index, curr_rank])
              with open(str(epoch) + 'el2n.pkl', 'wb') as f:
                  pickle.dump(all_, f)'''

torch.save(unpruned_net, 'ResNet20_last_20.pt')

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
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import pickle

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.manual_seed(2)
torch.cuda.manual_seed(2)

# CIFAR 10 load

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


train_loader = torch.load('train_loader_pruned.pth')
val_loader = torch.load('val_loader.pth')

training_required = True

if training_required:

  unpruned_net = torchvision.models.resnet18(pretrained=True)
  d = unpruned_net.fc.in_features
  unpruned_net.fc = torch.nn.Linear(d, 15)
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

      for batch_idx, (images, labels) in tqdm(enumerate(train_loader)):
          images, labels = images.to(device), labels.to(device)
          optimizer.zero_grad()

          outputs = unpruned_net(images)
          _, predictions = torch.max(outputs.data, 1)
          training_acc += (predictions == labels).sum().item()

          loss = criterion_CE(outputs, labels)
          avg_loss += loss.item()
          loss.backward()

          optimizer.step()
      torch.cuda.empty_cache()

      print(f'Epoch {epoch} -> Loss: {avg_loss/len(train_loader)}, Training Accuracy: {training_acc/len(train_loader.dataset)}')

      if epoch % checkpoint_every == 0 and epoch != 0:
          torch.save(unpruned_net, 'ResNet_epoch' + str(epoch) + '.pt')

      
      unpruned_net.eval()
      for batch_idx, (test_images, test_labels) in enumerate(val_loader):
          test_images, test_labels = test_images.to(device), test_labels.to(device)

          test_outputs = unpruned_net(test_images)
          _, test_predictions = torch.max(test_outputs.data, 1)
          val_acc += (test_predictions == test_labels).sum().item()

      torch.cuda.empty_cache()

      print('Val Accuracy: ', val_acc/len(val_loader.dataset))
      print()

torch.save(unpruned_net, 'ResNet20_pruned.pt')

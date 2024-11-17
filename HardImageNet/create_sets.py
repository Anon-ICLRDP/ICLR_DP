import torch
import torchvision
from torchvision import transforms
from PIL import Image

import random
import numpy as np
import pandas as pd
import os

from collections import defaultdict

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


train_dataset = torchvision.datasets.ImageFolder('./HardImageNet_Images/train', transform=transform)
val_dataset = torchvision.datasets.ImageFolder('./HardImageNet_Images/val', transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle = True, batch_size = 128)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle = True, batch_size = 128)

print(len(train_loader))
print(len(train_loader.dataset))

torch.save(train_loader, 'train_loader.pth')
torch.save(val_loader, 'val_loader.pth')

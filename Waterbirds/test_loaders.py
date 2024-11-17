import torch

t1 = torch.load('train_loader.pth')
print(len(t1.dataset))
t2 = torch.load('train_loader2.pth')
print(len(t2.dataset))

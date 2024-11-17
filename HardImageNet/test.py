import torch
import pickle
import os
from collections import defaultdict

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


models = ['ResNet20_last.pt', 'ResNet20_last_3.pt', 'ResNet20_last_5.pt', 'ResNet20_last_10.pt', 'ResNet20_last_15.pt', 'ResNet20_last_25.pt', 'ResNet20_last_50.pt', 'ResNet20_50prune.pt', 'ResNet20_70prune.pt', 'ResNet20_90prune.pt', 'ResNet20_95prune.pt', 'ResNet20_97prune.pt', 'ResNet20_99prune.pt', 'ResNet20_10prunereverse.pt', 'ResNet20_30prunereverse.pt', 'ResNet20_50prunereverse.pt', 'ResNet20_70prunereverse.pt', 'ResNet20_90prunereverse.pt', 'ResNet20_95prunereverse.pt', 'ResNet20_97prunereverse.pt', 'ResNet20_99prunereverse.pt', 'ResNet20_99_5prunereverse.pt', 'ResNet20_50prune_lth.pt', 'ResNet20_70prune_lth.pt', 'ResNet20_90prune_lth.pt']
#models = ['nonoracle/' + i for i in models]
#models = ['ResNet20_last.pt', 'ResNet20_last_3.pt', 'ResNet20_last_5.pt', 'ResNet20_last_10.pt', 'ResNet20_last_15.pt', 'ResNet20_last_25e.pt', 'ResNet20_last_50e.pt']
#models = ['ResNet20_last_25e.pt', 'ResNet20_last_25.pt', 'ResNet20_last_50e.pt', 'ResNet20_last_50.pt']
#models = ['ResNet20_last.pt', 'ResNet20_last_smol_rel2n_oraclesemi.pt', 'ResNet20_last_smol_el2n_oraclesemi.pt', 'ResNet20_last_smol_rel2n_oraclesemi2.pt', 'ResNet20_last_smol_el2n_oraclesemi2.pt', 'ResNet20_last_smol_rel2n_oraclesemi3.pt', 'ResNet20_last_smol_el2n_oraclesemi3.pt', 'ResNet20_last_smol_rel2n_oraclesemi4.pt', 'ResNet20_last_smol_el2n_oraclesemi4.pt', 'ResNet20_last_smol_rel2n_oraclesemi5.pt', 'ResNet20_last_smol_el2n_oraclesemi5.pt', 'ResNet20_last_smol_rel2n_oraclesemi6.pt', 'ResNet20_last_smol_el2n_oraclesemi6.pt']

#models = ['ResNet50_last.pt', 'ResNet50_last_smol_rel2n_fulloracle.pt', 'ResNet50_last_smol_el2n_fulloracle.pt', 'ResNet50_last_smol_rel2n_fulloracle2.pt', 'ResNet50_last_smol_el2n_fulloracle2.pt', 'ResNet50_last_smol_rel2n_fulloracle3.pt', 'ResNet50_last_smol_el2n_fulloracle3.pt']
val_loader = torch.load('val_loader.pth')

for m in models:
    try:
        print(m)
        net = torch.load(m).cuda()
    except:
        continue
    val_acc = 0
    d0 = defaultdict(int)
    d1 = defaultdict(int)
    d2 = defaultdict(int)
    d3 = defaultdict(int)
    d4 = defaultdict(int)
    d5 = defaultdict(int)
    d6 = defaultdict(int)
    d7 = defaultdict(int)
    d8 = defaultdict(int)
    d9 = defaultdict(int)
    d10 = defaultdict(int)
    d11 = defaultdict(int)
    d12 = defaultdict(int)
    d13 = defaultdict(int)
    d14 = defaultdict(int)
    for batch_idx, (test_images, test_labels) in enumerate(val_loader):
        test_images, test_labels = test_images.cuda(), test_labels.cuda()
        test_outputs = net(test_images)
        _, test_predictions = torch.max(test_outputs.data, 1)
        val_acc += (test_predictions == test_labels).sum().item()
        for j in range(len(test_predictions)):
            if test_labels[j].item() == 0:
                d0[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 1:
                d1[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 2:
                d2[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 3:
                d3[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 4:
                d4[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 5:
                d5[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 6:
                d6[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 7:
                d7[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 8:
                d8[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 9:
                d9[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 10:
                d10[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 11:
                d11[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 12:
                d12[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 13:
                d13[test_predictions[j].item()] += 1
            elif test_labels[j].item() == 14:
                d14[test_predictions[j].item()] += 1

    print(val_acc/len(val_loader.dataset))
    print(d1)
    print(d6)

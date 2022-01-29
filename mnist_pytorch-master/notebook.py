# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import sys



from torchvision.transforms import ToTensor

import torch
import torch.nn as nn
import torch.optim as optim



from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet50

# +
from sklearn.datasets import fetch_openml

class MNISTDataset(Dataset):
    
    def __init__(self, transform=None):
        self.mnist = fetch_openml('mnist_784', version=1,)
        self.data = self.mnist.data.reshape(-1, 28, 28, 1).astype('uint8')
        self.label = self.mnist.target.astype(int)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = torch.from_numpy(np.array(self.label[idx]))
        
        if self.transform:
            data = self.transform(data)
                
        sample = (data, label)
        return sample


# +
def train(net, train_loader):
    net.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad() 
        outputs = net(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predict = torch.max(outputs.data, 1)
        correct += (predict == labels).sum().item()
        total += labels.size(0)
        
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    return net, train_loss, train_acc

def valid(net, valid_loader):
    net.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        
        for batch_idx, (images, labels) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predict = torch.max(outputs.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
            
    val_loss = running_loss / len(valid_loader)
    val_acc = correct / total
    
    return net, val_loss, val_acc


# -

#parameter 
num_epochs = 50
batch_size = 128
learning_rate = 0.01

device = 'cuda:0'
criterion = nn.CrossEntropyLoss()

dataset = MNISTDataset(transform=ToTensor())

fold = KFold(n_splits=3, shuffle=True, random_state=0)

cv = 0.0

for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(dataset.data, dataset.label)):
    
    print('fold {}'.format(fold_idx))
    net = resnet50(num_classes=10)
    net.conv1  = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(Subset(dataset, valid_idx), shuffle=False, batch_size=batch_size)
    
    for epoch_idx in range(num_epochs):
        
        net, train_loss, train_acc = train(net, train_loader)
        net, valid_loss, valid_acc = valid(net, valid_loader)
        
        print('train_loss {:.3f} valid loss {:.3f} train_acc {:.3f} valid_acc {:.3f}'.format(train_loss, valid_loss, train_acc, valid_acc))
        
    cv += valid_acc / fold.n_splits



print('cv {}'.format(cv))



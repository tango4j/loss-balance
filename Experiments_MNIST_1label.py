#!/usr/bin/env python
# coding: utf-8

from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

# Set up data loaders
from datasets import SiameseMNIST

from trainer import fit, fit_siam, fit_org 
from losses import ContrastiveLoss_mod
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt

# Set up the network and training parameters
from networks import EmbeddingNet, SiameseNet, ClassificationNet
from losses import ContrastiveLoss, CrossEntropy
from metrics import *

mean, std = 0.1307, 0.3081

train_dataset = MNIST('../data/MNIST', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
test_dataset = MNIST('../data/MNIST', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))
n_classes = 10

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']



seed = 20
torch.manual_seed(seed)
np.random.seed(seed)

batch_size = 128


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
ipdb.set_trace()

margin = 1.
embedding_net = EmbeddingNet()
# model = SiameseNet(embedding_net)
model = ClassificationNet(embedding_net, n_classes=n_classes)
if cuda:
    model.cuda()

# loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100


margin = 1.
# loss_fn = ContrastiveLoss(margin)
# loss_fn = ContrastiveLoss_mod(margin)
loss_fn = CrossEntropy()
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 500

# metrics=[SimpleSiamDistAcc]
metrics=[AccumulatedAccuracyMetric_mod]
fit_org(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metric_classes=metrics)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)




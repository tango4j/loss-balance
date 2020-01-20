#!/usr/bin/env python
# coding: utf-8

from utils import *

from datasets import DatasetTorchvision, SiameseMNIST, SiameseMNIST_MT, TripletMNIST_MT
from torchvision.datasets import ImageFolder
from networks import EmbeddingNet, EmbeddingNetVGG, EmbeddingNetRGB, SiameseNet, SiameseNet_ClassNet, Triplet_ClassNet
        
from utils import RandomNegativeTripletSelector, HardestNegativeTripletSelector, SemihardNegativeTripletSelector 
        
from datasets import BalancedBatchSampler

from losses import ContrastiveLoss, OnlineTripletLoss
from losses import ContrastiveLoss_mod, CrossEntropy

import torch
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

from metrics import *
from trainer import fit, fit_siam, fit_org
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

import os
from utils import *
import utils
import sys

cuda = torch.cuda.is_available()
"""
######################################################################

1. add metric "siameseEucDist" - Done
2. Do data processing: siamese and classification data togther for multitasking - under procsess
3. Modify the trainer function

######################################################################
"""

            
seed_offset, ATLW, loss_key = int(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])
loss_type_dict = {"co": "Ctrst", "tr": "Trplt"}
loss_type = loss_type_dict[loss_key]

dataset_name = 'MNIST'
# dataset_name = 'FashionMNIST'
# dataset_name = 'CIFAR10'
# dataset_name = 'CIFAR100'
# dataset_name = 'CALTECH101'

if dataset_name == 'CALTECH101':
    dataC = ImageFolder
else:
    dataC = DatasetTorchvision(dataset_name)

if 'MNIST' in dataset_name:
    EmbeddingNet = EmbeddingNet
    embd_size=2
    n_epochs=25

elif 'CIFAR' in dataset_name:
    # EmbeddingNet = EmbeddingNetRGB
    EmbeddingNet = EmbeddingNetVGG
    embd_size=512
    if dataset_name == 'CIFAR10':
        n_epochs=50
    elif dataset_name == 'CIFAR100':
        n_epochs=100
    # n_epochs=2
 
# ipdb.set_trace()
train_dataset, test_dataset, n_classes = dataC.getDataset()


# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
        
batch_size = 2**11
# batch_size = 2**10

log_interval = 500

loss_fn_ce = CrossEntropy()
        
# Step 4
margin=1.0
loss_type = 'Ctrst'
loss_type = 'Trplt'

if loss_type == 'Ctrst':
    siamese_MT_train_dataset = SiameseMNIST_MT(train_dataset, seed=0, noisy_label=False) 
    siamese_MT_test_dataset = SiameseMNIST_MT(test_dataset, seed=0, noisy_label=False)

    _ClassNet = SiameseNet_ClassNet
    loss_fn = ContrastiveLoss_mod(margin)
    loss_fn_tup = (loss_fn, loss_fn_ce)

elif loss_type == 'Trplt':
    siamese_MT_train_dataset = SiameseMNIST_MT(train_dataset, seed=0, noisy_label=False) 
    siamese_MT_test_dataset = SiameseMNIST_MT(test_dataset, seed=0, noisy_label=False)

    # batch_size=2**7
    batch_size=2**15
    batch_size=311250
    triplet_seed_samples = 250
    train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=n_classes, n_samples=triplet_seed_samples)
    test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=n_classes, n_samples=triplet_seed_samples)
    
    # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    # _ClassNet = SiameseNet_ClassNet
    _ClassNet = Triplet_ClassNet
    # loss_fn_triplet= OnlineTripletLoss(margin, HardestNegativeTripletSelector(batch_size, margin))
    loss_fn_triplet= OnlineTripletLoss(margin, RandomNegativeTripletSelector(batch_size, margin))
    
    loss_fn = ContrastiveLoss_mod(margin)
    loss_fn_tup = (loss_fn, loss_fn_triplet, loss_fn_ce)


interval = 0.2
# interval = 0.1
write_list, mw_list = [], []
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# For reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

metric_classes=[SimpleSiamDistAcc, AccumulatedAccuracyMetric_mod]

soff=0
if ATLW   == 'na': # No automatic loss weight tuning: Grid search
    start = 0
    # start = 0.5
    # interval = interval
    end = 1.0+ (interval*0.5)
    seedmax = 10
    log_tag = "grid_search"

elif 'kl' in ATLW: # KLMW and MVMW
    init_mix_weight = 0.5
    start, interval, end = 0, 1, 1
    seedmax = 50
    log_tag = "auto"

elif ATLW == 'gn': # GradNorm method
    init_mix_weight = 0.5
    start, interval, end = 0, 1, 1
    seedmax = 50
    log_tag = "gradnorm"

# ipdb.set_trace()
seed_range = (soff+1, soff+seedmax)
for k in range(seed_range[0], seed_range[1]+1):
    for mwk, mix_weight in enumerate(np.arange(start, end, interval)):
        if ATLW in ['gn', 'kl', 'klan' ]:
            mix_weight = init_mix_weight
            print("Mix weight count:", mwk, mix_weight)

        # Setting a seed and an initial weight.
        mix_weight = round(mix_weight, 3)
        seed = k + seed_offset
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Data Loader Initialization
        if loss_type == "Ctrst":
            siamese_train_MT_loader = torch.utils.data.DataLoader(siamese_MT_train_dataset, batch_size=batch_size, shuffle=False, **kwargs)
            siamese_test_MT_loader = torch.utils.data.DataLoader(siamese_MT_test_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        elif loss_type == "Trplt":
            siamese_train_MT_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
            siamese_test_MT_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)
        
        # Model Initialization
        embedding_net = EmbeddingNet(embd_size)
        model_mt = _ClassNet(embedding_net, n_classes=n_classes, embd_size=embd_size)
        
        # Step 3
        if cuda:
            model_mt.cuda()
            
        optimizer_mt = optim.Adam
        scheduler_mt = None
        
        mix_weight = torch.tensor(mix_weight).cuda()
        variable_dict = {"embd_size": embd_size, "mix_weight": mix_weight, "ATLW": ATLW, "seed": seed, "margin": margin, "seed_range":seed_range, "loss_type": loss_type,
                "interval": interval, "log_tag": log_tag, "dataset_name": dataset_name, "model_name": embedding_net.model_name, "write_list": write_list, "batch_size": batch_size}
        write_var, write_list, mix_weight, mix_weight_list = fit_siam(siamese_train_MT_loader, siamese_test_MT_loader, model_mt, loss_fn_tup, optimizer_mt, scheduler_mt, n_epochs, cuda, log_interval, metric_classes, variable_dict)
    
        # write_list, mw_list = writeAndSave(write_var, mix_weight_list, margin, seed_offset, n_epochs, interval, log_tag, write_list)


import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

import ipdb
import copy
import random
import os
import torch


from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100

def get_noisylabel(label, n_classes):
    lin = np.arange(0, n_classes) 
    noisy_label  = random.choice(np.delete(lin, label))
    return noisy_label


class DatasetTorchvision:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']
        
        self.stat_dict = {'MNIST': {'mean':0.1307, 'std':0.3081},
                   'FashoinMNIST': {'mean':0.28604059698879553, 'std':0.35302424451492237},
                        'CIFAR10': {'mean':0.25, 'std':0.35},
                       'CIFAR100': {'mean':0.25, 'std':0.35} 
                    }
        self.num_classes_dict = {'MNIST':10,
                                 'FashionMNIST':10, 
                                 'CIFAR10':10,
                                 'CIFAR100':100
                                 } 
        self.data_classes = [ str(x) for x in range(self.num_classes_dict[self.dataset_name]) ]
        
        self.MNIST = MNIST
        self.FashionMNIST = FashionMNIST
        self.CIFAR10 = CIFAR10
        self.CIFAR100 = CIFAR100

        
    def getDataset(self):
           
        # self.mean, self.std = self.stat_dict[self.dataset_name]['mean'], self.stat_dict[self.dataset_name]['std']
        self.mean, self.std = 0, 1
        self.n_classes = self.num_classes_dict[self.dataset_name]
        self.data_folder = '../data/{}'.format(self.dataset_name)

        self.train_dataset = getattr(self, self.dataset_name)(self.data_folder,
                                     train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((self.mean,), (self.std,))
                                     ]))
        self.test_dataset = getattr(self, self.dataset_name)(self.data_folder,
                                    train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((self.mean,), (self.std,))
                                    ]))

        if 'CIFAR' in self.dataset_name:
            self.train_dataset.train_labels = torch.tensor(self.train_dataset.targets)
            # self.train_dataset.train_data = torch.tensor(self.train_dataset.data).permute(0, 3, 1, 2) 
            self.train_dataset.train_data = torch.tensor(self.train_dataset.data)
            # self.train_dataset.train_data = torch.tensor(self.train_dataset.data)
            
            self.test_dataset.test_labels = torch.tensor(self.test_dataset.targets)
            # self.test_dataset.test_data = torch.tensor(self.test_dataset.data).permute(0, 3, 1, 2)
            self.test_dataset.test_data = torch.tensor(self.test_dataset.data)
            # self.test_dataset.test_data = torch.tensor(self.test_dataset.data)
            # ipdb.set_trace()
        return self.train_dataset, self.test_dataset, self.n_classes

class SiameseMNIST_MT(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_style_dataset, seed=0, noisy_label=False):
        np.random.seed(seed)
        self.mnist_style_dataset = mnist_style_dataset
    
        self.train = self.mnist_style_dataset.train
        self.transform = self.mnist_style_dataset.transform
        
        if self.train:
            self.train_labels = self.mnist_style_dataset.train_labels
            self.train_data = self.mnist_style_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
            self.data_tensor = self.train_data
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_style_dataset.test_labels
            if noisy_label:
                random_idx = torch.randperm(self.test_labels.size()[0]) 
                self.test_labels = self.test_labels[random_idx]
            self.test_data = self.mnist_style_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            self.data_tensor = self.test_data 

            positive_pairs = []
            print("creating pairs...")
            for i in range(0, len(self.test_data), 2):
                rand_idx = np.random.choice(self.label_to_indices[self.test_labels[i].item()]),
                label1, label2 = self.test_labels[i].item() , self.test_labels[rand_idx].item()
                positive_pairs.append([i, rand_idx[0], 1, label1, label2])
                assert label1 == label2, "label1 and label2 should be the same"

            negative_pairs = []
            for i in range(1, len(self.test_data), 2):
                rand_idx = np.random.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                label1, label2 = self.test_labels[i].item() , self.test_labels[rand_idx].item()
                assert label1 != label2, "label1 and label2 should be different"
                negative_pairs.append([i, rand_idx[0], 0, label1, label2])
            self.test_pairs = positive_pairs + negative_pairs

        # Setup the image mode:
        if len(self.data_tensor.shape) == 3:
            self.image_mode = 'L'
        if len(self.data_tensor.shape) == 4:
            self.image_mode = 'RGB'
    
    def __getitem__(self, index):
        # np.random.seed(0)
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
                label2 = copy.deepcopy(label1)
                assert label1 == label2, "label1 and label2 should be the same"
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
                label2 = self.train_labels[siamese_index].item()
                assert label1 != label2, "label1 and label2 should be different"
            img2 = self.train_data[siamese_index]
            # ipdb.set_trace()
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
            label1 = self.test_pairs[index][3]
            label2 = self.test_pairs[index][4]

        # print("img1 raw", img1.shape, self.image_mode)
        img1 = Image.fromarray(img1.numpy(), mode=self.image_mode)
        img2 = Image.fromarray(img2.numpy(), mode=self.image_mode)
        
        # print("img1 before transform", img1)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # print("img1", img1.shape)
        return (img1, img2), target, label1, label2

    def __len__(self):
        return len(self.mnist_style_dataset)

class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        if len(self.mnist_dataset.train_data.shape) == 3:
            self.image_mode = 'L'
        elif len(self.mnist_dataset.train_data.shape) == 4:
            self.image_mode = 'RGB'
    
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            # random_state = np.random.RandomState(29)
            random_state = np.random.RandomState()

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
            assert target < 2, "target should be either 0 or 1"

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]
          
        img1 = Image.fromarray(img1.numpy(), mode=self.image_mode)
        img2 = Image.fromarray(img2.numpy(), mode=self.image_mode)
        img3 = Image.fromarray(img3.numpy(), mode=self.image_mode)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

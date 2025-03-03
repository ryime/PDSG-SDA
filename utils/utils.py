import os
import time
import torch
import torch.distributed
import torch.utils.data
import errno
import datetime
from math import nan

import sys

sys.path.append('..')

class DatasetSplitter(torch.utils.data.Dataset):
    '''To split CIFAR10DVS into training dataset and test dataset'''
    def __init__(self, parent_dataset, rate=0.1, train=True):

        self.parent_dataset = parent_dataset
        self.rate = rate
        self.train = train
        self.it_of_original = len(parent_dataset) // 10
        self.it_of_split = int(self.it_of_original * rate)

    def __len__(self):
        return int(len(self.parent_dataset) * self.rate)

    def __getitem__(self, index):
        base = (index // self.it_of_split) * self.it_of_original
        off = index % self.it_of_split
        if not self.train:
            off = self.it_of_original - off - 1
        item = self.parent_dataset[base + off]

        return item

class DatasetWarpper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


class DVStransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        shape = [img.shape[0], img.shape[1]]
        img = img.flatten(0, 1)
        img = self.transform(img)
        shape.extend(img.shape[1:])
        img = img.view(shape)
        return img

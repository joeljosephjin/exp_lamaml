
import numpy as np
from PIL import Image
import torch

# what is a transform??
from torchvision import datasets, transforms
import os

# ??
# from dataloaders import cifar_info

# creating a custom dataset by subclassing and overriding the Dataset class in torch.utils.data
# has funcns for length and get_item
class DummyDataset(torch.utils.data.Dataset):

    # x- image, y- label, trsf= transform, pretrsf= pre-transform, super_y= ??
    def __init__(self, x, y, trsf, pretrsf = None, imgnet_like = False, super_y = None):
        self.x, self.y = x, y
        self.super_y = super_y

        # transforms to be applied before and after conversion to imgarray
        self.trsf = trsf
        self.pretrsf = pretrsf

        # imgnet comes as array
        # if not from imgnet, needs to be converted to imgarray first
        self.imgnet_like = imgnet_like

    # since images are square, it only needs the width i guess
    def __len__(self):
        return self.x.shape[0]

    # return x[idx], y[idx], super_y[idx] after converting to array and applying transforms
    def __getitem__(self, idx):
        # does the only step thats required i guess
        x, y = self.x[idx], self.y[idx]
        # if super_y has something, return its idx-th element too
        if self.super_y is not None: super_y = self.super_y[idx]

        # apply a pre-transform if necessary
        if(self.pretrsf is not None): x = self.pretrsf(x)    
        
        # convert to array if necessary (i.e. if its not imgnet-like)
        if(not self.imgnet_like): x = Image.fromarray(x)
        
        # apply a post-transform i guess
        x = self.trsf(x)

        # return {x, y, super_y(if it exists)} 
        if self.super_y is not None: return x, y, super_y
        else: return x, y

# creating a custom dataset AGAIN, but this time the most simple
# has funcns for length and get_item - extremely simplistic and straightforward
class DummyArrayDataset(torch.utils.data.Dataset):

    # no transforms or supers or array-izers - just simple
    def __init__(self, x, y):
        self.x, self.y = x, y

    # usual stuff, getting the width
    def __len__(self):
        return self.x.shape[0]

    # simple return the idx-th item. [no transforms, no array-izing or supers]
    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        return x, y

# return a collection of _get_datasets from the list
def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]

# the real deal of previous function
# returns dataset(object)s of {cifar10,cifar100,tinyimgnet}
def _get_dataset(dataset_name):
    # lower_case dataset_name and strip it of any spaces at the end or beginning
    dataset_name = dataset_name.lower().strip()

    # straightforward returning the appropriate dataset(object)
    if dataset_name == "cifar10": return iCIFAR10
    elif dataset_name == "cifar100": return iCIFAR100
    elif dataset_name == "tinyimagenet": return iImgnet
    else: raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


import torch
from torchvision import datasets, transforms, utils
from utils.utils_data import cross_merge_dfs, logspace, allcombinations, add_standard_attributes, repeat, soft_astype_df
from time import time
import pandas as pd
from typing import Any, Tuple, Optional, Union
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import pickle
import os
from sparse_recov.train_model import ConvGenerator, Generator, ConvVAE, VAEWrapper
from sparse_recov import convvae

train_transforms = transforms.Compose([
    transforms.Resize((128, 112)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder("../datasets/celeba", transform=train_transforms)

root = "../datasets/celeba"
train_dataset = datasets.CelebA(root=root, split="train", download=False)

# Sample one random image from the dataset
random_idx = np.random.randint(0, len(train_data))
true_signal, _ = train_data[random_idx]

true_signal = true_signal.to(device)

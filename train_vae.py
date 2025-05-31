import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sparse_recov.train_model import ConvVAE, vanilla_train_vae
from sparse_recov.toy_Fourier import heavy_tail_MNIST_loaders
import pandas as pd
import seaborn as sns
from pathlib import Path
from sparse_recov.train_model import train_with_heavy_tail_loss_co

vanilla_train_vae()

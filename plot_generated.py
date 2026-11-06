import torch
from sparse_recov.train_model import ConvVAE
from sparse_recov.generative import get_fft_loc_coherences
import matplotlib.pyplot as plt
from seaborn import heatmap, histplot
from utils.utils_plot import LogNorm
import numpy as np
import seaborn as sns

tail_vae = ConvVAE(k=16)
tail_state=torch.load("./models/fat_tail.pth", map_location=torch.device('cpu'))
tail_vae.load_state_dict(tail_state)
Gtail= tail_vae.decoder
tco = get_fft_loc_coherences(Gtail)
sns.heatmap(Gtail(torch.randn(Gtail.k))[0,0,:,:].detach().numpy())

from diffusers.models import AutoencoderKL

autoencoder = AutoencoderKL.from_pretrained("microsoft/mri-autoencoder-v0.1")

decoder = autoencoder.decoder.requires_grad_(False)

from fastmri.data import SliceDataset
import fastmri.data
fastmri.data.mri_data


SliceDataset(challenge = "singlecoil", transform = To)
help(SliceDataset)

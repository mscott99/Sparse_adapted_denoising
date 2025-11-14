import torch
from sparse_recov.convvae.model import VAE
import os


def get_model(label="vae-200"):
    # load model
    proj_path = os.getenv("PROJECT")
    model_dir = os.getenv("MODEL_DIR")
    assert model_dir is not None, "MODEL_DIR is not set."
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dump = torch.load(os.path.join(model_dir,
                      f"{label}.pth"), map_location=device, weights_only=False)
    vae = VAE(dump['input_shape'], dump['z_dim']).to(device)
    vae.load_state_dict(dump['state_dict'])
    vae.requires_grad_(False)
    return vae

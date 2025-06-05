import torch
from . import model
import os


def get_model(label="vae-200"):
    # load model
    proj_path = os.getenv("PROJECT")
    assert proj_path is not None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dump = torch.load(os.path.join(proj_path, "models",
                      f"{label}.dat"), map_location=device)
    vae = model.VAE(dump['input_shape'], dump['z_dim']).to(device)
    vae.load_state_dict(dump['state_dict'])
    vae.requires_grad_(False)
    return vae

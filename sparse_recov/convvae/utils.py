import torch
from . import model

def get_model(label="vae-200"):
    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dump = torch.load(f"./models/{label}.dat", map_location=device)
    vae = model.VAE(dump['input_shape'], dump['z_dim']).to(device)
    vae.load_state_dict(dump['state_dict'])
    vae.requires_grad_(False)
    return vae

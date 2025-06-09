from . import convvae
import numpy as np
import torchvision
import os
from torchvision import transforms
from .train_model import VAEWrapper
import torch


def exterior_redo_recover_signal(b, num_tries=3, tol_stop=1e-4, max_iter=4):
    best_obj, best_img = recover_signal(b, max_iter)
    for _ in range(1, num_tries):
        if best_obj < tol_stop:
            print(best_obj)
            return best_obj, best_img

        last_obj, last_img = recover_signal(b, max_iter)
        if last_obj < best_obj:
            print(last_obj)
            best_obj = last_obj
            best_img = last_img
    # print("Accuracy ", tol_stop, " not achieved, only managed ", best_obj[3])
    return best_obj, best_img


def recover_signal(b, max_iter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(1, 32, requires_grad=True)
    full_model = convvae.get_model("vae-32").eval()
    generator = VAEWrapper(full_model).to(device)

    optimizer = torch.optim.LBFGS([z])

    def closure():
        optimizer.zero_grad()
        loss = torch.norm(generator(z) - b)**2
        loss.backward()
        return loss

    for _ in max_iter:
        optimizer.step(closure)

    return (torch.norm(generator(z) - b)/torch.norm(b)).item(), generator(z)

if __name__ == "__main__":
    n = 128*112*3
    train_transforms = transforms.Compose([
        transforms.Resize((128, 112)),
        transforms.ToTensor(),
    ])
    data_dir = os.getenv("DATA_DIR")
    assert data_dir is not None
    train_data = torchvision.datasets.CelebA(data_dir, target_type = "identity", split="train", transform = train_transforms, download = False)
    random_idx = np.random.randint(0, int(len(train_data)))
    true_signal, _ = train_data[random_idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_signal = true_signal.to(device)
    result = exterior_redo_recover_signal(true_signal, 60, 1e-3, 4)    

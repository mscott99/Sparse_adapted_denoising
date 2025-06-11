from sparse_recov import convvae
import numpy as np
import torchvision
import os
from torchvision import transforms
from sparse_recov.train_model import VAEWrapper
import torch

def random_signal():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model = convvae.get_model("vae-32").eval().to(device)
    generator = VAEWrapper(full_model)
    generator.requires_grad_(False)
    return generator(torch.randn(1,32, device = device, requires_grad=False))

def exterior_redo_recover_signal(b, num_tries=3, tol_stop=1e-4, max_iter=4, printstuff=False):
    best_obj, best_img = recover_signal(b, max_iter)
    for _ in range(1, num_tries):
        if best_obj < tol_stop:
            if printstuff:
                print(best_obj)
            return best_obj, best_img

        last_obj, last_img = recover_signal(b, max_iter)
        if last_obj < best_obj:
            if printstuff:
                print(last_obj)
            best_obj = last_obj
            best_img = last_img
    # print("Accuracy ", tol_stop, " not achieved, only managed ", best_obj[3])
    return best_obj, best_img


def recover_signal(b, max_iter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model = convvae.get_model("vae-32").eval().to(device)
    b.to(device)
    generator = VAEWrapper(full_model)
    generator.requires_grad_(False)
    # mu_p, log_var_p = full_model.forward_encoder(b)
    # z = full_model.sampling(mu_p, log_var_p)
    # z.requires_grad = True
    z = 0.3 * torch.randn(1, 32, requires_grad=False, device = device)
    z.requires_grad = True
    optimizer = torch.optim.LBFGS([z])

    def closure():
        optimizer.zero_grad()
        loss = torch.norm(generator(z) - b)**2
        loss.backward()
        return loss

    for _ in range(max_iter):
        optimizer.step(closure)

    return (torch.norm(generator(z) - b)/torch.norm(b)).item(), generator(z).detach()

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
    true_signal = true_signal.to(device).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # full_model = convvae.get_model("vae-32").eval().to(device)
    # true_signal = full_model.forward(true_signal)[2]
    # true_signal.requires_grad=False
    # best, true_signal = exterior_redo_recover_signal(true_signal, 2, 1e-3, 20, False)    
    true_signal = random_signal()
    best, true_signal = exterior_redo_recover_signal(true_signal, 100, 1e-3, 5, True)    

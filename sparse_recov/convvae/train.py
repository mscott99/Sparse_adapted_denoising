#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets.celeba import CelebA
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from sparse_recov.train_model import VAEWrapper
from tensorboardX import SummaryWriter

from sparse_recov.train_model import get_fft_co_with_grad

from typing import Callable
import argparse

from model import VAE

def kl_loss(mu, log_var):
    # TODO: dividir entre el numero de batches? 
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - torch.exp(log_var))

def r_loss(y_train, y_pred):
    r_loss = torch.mean((y_train - y_pred) ** 2)
    return r_loss

def coherence_loss(vae:VAE):
    return -torch.sum(get_fft_co_with_grad(VAEWrapper(vae), 15))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--data_dir", help="Folder where the data is located")
    parser.add_argument("--epochs", default=1, type=int, help='Number of times to iterate the whole dataset')
    parser.add_argument("--visual_every", default=200, type=int, help='Display faces every n batches')
    parser.add_argument("--z_dim", type=int, default=200, help='Dimensions of latent space')
    parser.add_argument("--r_loss_factor", type=float, default=10000.0, help='r_loss factor')
    parser.add_argument("--lr", type=float, default=0.002, help='Learning rate')
    parser.add_argument("--batch_size", default=32, type=int, help='Batch size')
    parser.add_argument("--load", type=str, default='', help='Load pretrained weights')
    parser.add_argument("--co_loss_coeff", type=float, default=0.0, help="Fourier coherence loss coefficient")
    parser.add_argument("--starting_state", type=str, help="File name of checkpoint, located in $MODEL_DIR.")
    parser.add_argument("--prefix", type=str, default="vae-", help="checkpoint file name.")

    model_dir = os.getenv("MODEL_DIR")
    assert  model_dir != None, "MODEL_DIR not set."
    assert model_dir != None, "MODEL_DIR not set."
    model_dir = Path(model_dir)

    args = parser.parse_args()
    save_dir = tempfile.mkdtemp(prefix=args.prefix, dir=model_dir)
    # data where the images are located
    data_dir = os.getenv("DATA_DIR")
    assert  data_dir != None, "DATA_DIR not set."
    data_dir = Path(data_dir)
    device = torch.device("cuda" if args.cuda else "cpu")
    train_transforms = transforms.Compose([
        # transforms.Resize((176, 144)),
        transforms.Resize((128, 112)),
        transforms.ToTensor(),
    ])
    # train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    train_data = CelebA(data_dir, target_type = "identity", split="train", transform = train_transforms, download = False)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    images, labels = next(iter(trainloader))

    # create model
    input_shape = next(iter(trainloader))[0].shape
    vae = VAE(input_shape[-3:], args.z_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    if args.starting_state:
        saved_state = torch.load(os.path.join(model_dir, args.starting_state), weights_only=False)
        print(os.path.join(model_dir, args.starting_state))
        if "state_dict" in saved_state.keys() and saved_state['state_dict']:
            vae.load_state_dict(saved_state['state_dict'])
            print("Weights loaded")
        if "opt_state" in saved_state.keys() and saved_state['opt_state']:
            optimizer.load_state_dict(saved_state['opt_state'])
            print("Optimizer loaded")

    # create tensorboard writer
    writer = SummaryWriter(comment= args.prefix + str(args.z_dim))

    # generate random points in latent space so we can see how the network is training
    latent_space_test_points = np.random.normal(scale=1.0, size=(16, args.z_dim))
    latent_space_test_points_v = torch.Tensor(latent_space_test_points).to(device)

    batch_iterations = 0
    training_losses = []
    vae.train()
    for e in range(args.epochs):
        epoch_loss = []
        epoch_co_loss = []
        for images, _ in trainloader:
            images_v = images.to(device)

            optimizer.zero_grad()

            mu_v, log_var_v, images_out_v = vae(images_v)
            r_loss_v = r_loss(images_out_v, images_v)
            kl_loss_v = kl_loss(mu_v, log_var_v)
            loss = kl_loss_v + r_loss_v * args.r_loss_factor

            if args.co_loss_coeff != 0.0:
                co_loss = args.co_loss_coeff * coherence_loss(vae)
                loss +=  co_loss
                epoch_co_loss.append(co_loss.item())

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

            if batch_iterations % args.visual_every == 0:
                # print loss
                print("Batch: {}\tLoss: {}".format(batch_iterations + e * len(trainloader) / args.batch_size, loss.item()))
                writer.add_scalar('loss', np.mean(epoch_loss[-args.visual_every:]), batch_iterations)
                if args.co_loss_coeff != 0.0:
                    writer.add_scalar('co_loss', np.mean(epoch_co_loss[-args.visual_every:]), batch_iterations)
                    writer.add_scalar('remainder', np.mean(epoch_loss[-args.visual_every:]) - np.mean(epoch_co_loss[-args.visual_every:]), batch_iterations)

            batch_iterations = batch_iterations + 1

        else:
            training_losses.append(np.mean(epoch_loss))
            if min(training_losses) == training_losses[-1]:
                vae.save(os.path.join(save_dir, 'vae_epoch_' + str(e) + '.pth'), opt_state=optimizer.state_dict(), train_settings=args)

            vae.eval()

            generated_imgs_v = vae.forward_decoder(latent_space_test_points_v).detach()
            imgs_grid = utils.make_grid(generated_imgs_v)

            writer.add_image('preview-1', imgs_grid.cpu().numpy(), batch_iterations)

            vae.train()


if __name__ == "__main__":
    main()

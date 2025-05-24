from sparse_recov.conv_vae import get_model, VAE
import torch
import matplotlib.pyplot as plt

model = get_model()
model.decoder_conv
model.z_dim

model.decoder_conv()

# Generate a random sample from the latent space
z = torch.randn(1, model.z_dim)
# If model is on GPU, move z to GPU
if next(model.parameters()).is_cuda:
    z = z.cuda()

# Generate reconstruction from the latent sample
with torch.no_grad():
    reconstructed = model.forward_decoder(z)
    
img = reconstructed[0].permute(1, 2, 0).cpu().numpy()

# Plot the reconstructed image
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.title('Reconstructed Image from VAE')
plt.axis('off')
plt.show()

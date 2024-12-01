import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
import os
import torchvision.utils as vutils

# Mostra imagens geradas
def show_generated_images(generator, latent_dim, num_images=16):
    z = torch.randn(num_images, latent_dim)
    fake_images = generator(z).detach()
    grid = torch.cat([img.squeeze() for img in fake_images], dim=-1).cpu().numpy()
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.show()

def save_generated_image(fake_images, epoch, output_dir='generated_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, f'generated_epoch_{epoch}.png')

    vutils.save_image(fake_images.data, filename, normalize=True, nrow=8)
    print(f"Images saved to {filename}")
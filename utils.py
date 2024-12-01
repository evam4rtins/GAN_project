import matplotlib.pyplot as plt
import torch

# Mostra imagens geradas
def show_generated_images(generator, latent_dim, num_images=16):
    z = torch.randn(num_images, latent_dim)
    fake_images = generator(z).detach()
    grid = torch.cat([img.squeeze() for img in fake_images], dim=-1).cpu().numpy()
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.show()

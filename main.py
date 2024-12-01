import torch
import torch.nn as nn
import torch.optim as optim
from models import Generator, Discriminator
from dataset import get_dataloader
from utils import show_generated_images

# Configurações
latent_dim = 100
batch_size = 64
lr = 0.0002
epochs = 50

# Inicializa modelos, otimizadores e função de perda
generator = Generator(latent_dim)
discriminator = Discriminator()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

# Carrega dados
dataloader = get_dataloader(batch_size)

# Treinamento
for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        real = torch.ones(real_imgs.size(0), 1)  # Labels reais
        fake = torch.zeros(real_imgs.size(0), 1)  # Labels falsos

        # Treinando o Discriminador
        z = torch.randn(real_imgs.size(0), latent_dim)
        fake_imgs = generator(z).detach()
        loss_D_real = criterion(discriminator(real_imgs), real)
        loss_D_fake = criterion(discriminator(fake_imgs), fake)
        loss_D = loss_D_real + loss_D_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Treinando o Gerador
        z = torch.randn(real_imgs.size(0), latent_dim)
        fake_imgs = generator(z)
        loss_G = criterion(discriminator(fake_imgs), real)  # Queremos enganar o discriminador
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    # Exibe progresso
    print(f"Epoch {epoch+1}/{epochs} - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
    
    # Mostra imagens geradas a cada 10 epochs
    if (epoch + 1) % 10 == 0:
        show_generated_images(generator, latent_dim)

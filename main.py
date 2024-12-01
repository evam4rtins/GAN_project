import torch
import torch.nn as nn
import torch.optim as optim
from models import Generator, Discriminator
from dataset import get_dataloader
from utils import show_generated_images, save_generated_image  # Supondo que a função save_generated_image esteja em utils.py

# Configurações
latent_dim = 100
batch_size = 64
lr = 0.0002
epochs = 150
output_dir = 'generated_images'  # Diretório onde as imagens serão salvas

# Verifica se há uma GPU disponível, se não, usa a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move os modelos para o dispositivo (GPU ou CPU)
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

# Carrega dados
dataloader = get_dataloader(batch_size)

# Função para calcular a precisão do discriminador
def calculate_accuracy(discriminator, real_images, fake_images):
    real_preds = discriminator(real_images).view(-1)  # Previsões para imagens reais
    fake_preds = discriminator(fake_images).view(-1)  # Previsões para imagens falsas

    real_labels = torch.ones(real_images.size(0)).to(device)
    fake_labels = torch.zeros(fake_images.size(0)).to(device)

    # Calcular a precisão para imagens reais
    real_accuracy = (real_preds > 0.5).float().mean().item()  # Predições > 0.5 são "reais"
    fake_accuracy = (fake_preds < 0.5).float().mean().item()  # Predições < 0.5 são "falsas"

    # Calcular a precisão geral (média das duas precisões)
    accuracy = (real_accuracy + fake_accuracy) / 2
    return accuracy, real_accuracy, fake_accuracy

# Função para salvar as imagens geradas
def save_images(generator, epoch, latent_dim, output_dir):
    z = torch.randn(64, latent_dim).to(device)  # Vetor de ruído para gerar imagens
    fake_imgs = generator(z)
    save_generated_image(fake_imgs, epoch, output_dir=output_dir)

# Treinamento
for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)  # Certifique-se de que as imagens estão no dispositivo correto (CPU/GPU)
        real = torch.ones(real_imgs.size(0), 1).to(device)  # Labels reais
        fake = torch.zeros(real_imgs.size(0), 1).to(device)  # Labels falsos

        # Treinando o Discriminador
        z = torch.randn(real_imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z).detach()  # Desconectar do gráfico de cálculo (não treina o gerador aqui)
        loss_D_real = criterion(discriminator(real_imgs), real)
        loss_D_fake = criterion(discriminator(fake_imgs), fake)
        loss_D = loss_D_real + loss_D_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Treinando o Gerador
        z = torch.randn(real_imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z)
        loss_G = criterion(discriminator(fake_imgs), real)  # Queremos enganar o discriminador
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    # Calculando a precisão do discriminador
    accuracy, real_accuracy, fake_accuracy = calculate_accuracy(discriminator, real_imgs, fake_imgs)

    # Exibe progresso
    print(f"Epoch {epoch+1}/{epochs} - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
    print(f"Discriminator Accuracy: {accuracy*100:.2f}% (Real: {real_accuracy*100:.2f}%, Fake: {fake_accuracy*100:.2f}%)")

    # Salva imagens geradas a cada 10 épocas
    if (epoch + 1) % 10 == 0:
        save_generated_image(fake_imgs, epoch + 1, output_dir)

  

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normaliza para o intervalo [-1, 1]
    ])
    dataset = datasets.MNIST('.', download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

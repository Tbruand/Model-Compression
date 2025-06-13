# src/dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DataManager:
    def __init__(self):
        # Définition des transformations à appliquer aux images :
        # - Convertir en tenseur (tensor)
        # - Normaliser les pixels (moyenne 0.5, écart-type 0.5)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Téléchargement et préparation du jeu d'entraînement MNIST
        self.train_dataset = datasets.MNIST(
            "data", train=True, download=True, transform=transform
        )

        # Téléchargement et préparation du jeu de test MNIST
        self.test_dataset = datasets.MNIST(
            "data", train=False, download=True, transform=transform
        )

    def get_loaders(self, batch_size):
        """
        Retourne deux DataLoaders :
        - un pour l'entraînement (avec mélange des données),
        - un pour le test (sans mélange).
        """
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
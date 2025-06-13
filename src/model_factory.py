# src/model_factory.py
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Première couche de convolution : prend une image en niveaux de gris (1 canal), produit 32 cartes de caractéristiques
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Deuxième couche de convolution : prend 32 cartes, en produit 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Couche fully connected (dense) : connecte l'image aplatie à 128 neurones
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Dernière couche fully connected : produit 10 sorties pour 10 classes (par ex. MNIST)
        self.fc2 = nn.Linear(128, 10)
        # Opération de max pooling : réduit la taille spatiale de moitié
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Applique la première convolution, l'activation ReLU et le pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Applique la deuxième convolution, ReLU et pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Aplatissement des cartes de caractéristiques pour les couches fully connected
        x = x.view(-1, 64 * 7 * 7)
        # Couche dense avec activation ReLU
        x = F.relu(self.fc1(x))
        # Sortie finale sans activation (softmax appliqué plus tard si besoin)
        return self.fc2(x)
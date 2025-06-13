# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

from src.dataset import DataManager
from src.model_factory import SimpleCNN
from src.utils import evaluate_model
from src import config

class ModelTrainer:
    def __init__(self):
        # Détermine si l'on peut utiliser un GPU (cuda) ou le CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Instancie le modèle CNN défini dans model_factory.py
        self.model = SimpleCNN().to(self.device)
        # Prépare les gestionnaires de données d'entraînement et de test
        self.data = DataManager()

    def train_model(self):
        # Charge les DataLoaders pour les jeux d'entraînement et de test
        train_loader, test_loader = self.data.get_loaders(config.BATCH_SIZE)
        # Initialise l'optimiseur Adam et la fonction de perte (cross-entropy)
        optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        self.model.train()  # Passe le modèle en mode entraînement
        for epoch in range(config.EPOCHS):
            epoch_loss = 0.0  # Stocke la perte cumulée pour l'epoch
            start_time = time.time()  # Démarre le chronomètre pour l'epoch

            # Boucle sur chaque batch du jeu d'entraînement
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}"):
                # Envoie les données sur le device approprié (GPU ou CPU)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()              # Réinitialise les gradients
                outputs = self.model(inputs)       # Passe avant du modèle (prédictions)
                loss = criterion(outputs, labels)  # Calcul de la perte
                loss.backward()                    # Rétropropagation du gradient
                optimizer.step()                   # Mise à jour des poids
                epoch_loss += loss.item()          # Accumule la loss pour l'epoch

            elapsed = time.time() - start_time     # Temps écoulé pour cette epoch
            avg_loss = epoch_loss / len(train_loader)  # Moyenne de la perte sur l'epoch

            # Évaluation du modèle sur le jeu de test (calcul de l'accuracy)
            self.model.eval()  # Passe le modèle en mode évaluation
            accuracy = evaluate_model(self.model, test_loader, self.device)
            self.model.train()  # Repasse en mode entraînement pour la prochaine epoch

            # Affiche les résultats de l'epoch (loss moyenne, accuracy et temps écoulé)
            print(f"✅ Epoch {epoch+1} terminé — Loss moyenne: {avg_loss:.4f} — Accuracy: {accuracy:.2f}% — Temps: {elapsed:.2f}s")

        # Sauvegarde les poids du modèle entraîné dans un fichier
        torch.save(self.model.state_dict(), config.MODEL_PATH)
        return self.model
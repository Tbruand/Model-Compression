# src/distillation.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

from src.model_factory import SimpleCNN
from src.dataset import DataManager
from src import config

class KnowledgeDistiller:
    # La classe KnowledgeDistiller gère le processus de distillation des connaissances
    # entre un modèle enseignant (teacher) et un modèle étudiant (student).
    # L'objectif est d'entraîner le modèle étudiant à imiter les prédictions du modèle enseignant,
    # souvent plus grand ou plus performant, afin d'obtenir un modèle plus compact et efficace.
    def __init__(self, teacher_model):
        # Le modèle enseignant est un modèle préalablement entraîné (plus gros ou précis)
        self.teacher_model = teacher_model
        # Le modèle étudiant est une copie plus compacte à entraîner par imitation
        self.student_model = SimpleCNN()
        # On travaille sur GPU si disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Chargement des données d'entraînement
        self.data = DataManager()

    def apply(self):
        # Passage des modèles sur le bon device
        self.teacher_model.to(self.device).eval()      # Le modèle enseignant ne sera jamais mis à jour
        self.student_model.to(self.device).train()     # Le modèle étudiant sera entraîné

        # Optimiseur et fonction de perte adaptée à la distillation (KL Divergence)
        optimizer = optim.Adam(self.student_model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.KLDivLoss(reduction="batchmean")
        # Fonctions pour transformer les logits en probabilités
        softmax = nn.Softmax(dim=1)
        log_softmax = nn.LogSoftmax(dim=1)

        train_loader, _ = self.data.get_loaders(config.BATCH_SIZE)

        for epoch in range(config.EPOCHS):
            epoch_loss = 0.0
            start_time = time.time()

            # Boucle d'entraînement sur chaque batch
            # tqdm est utilisé pour afficher une barre de progression pendant l'entraînement,
            # ce qui permet de suivre facilement l'avancement.
            for inputs, _ in tqdm(train_loader, desc=f"Distillation Epoch {epoch+1}/{config.EPOCHS}"):
                inputs = inputs.to(self.device)

                with torch.no_grad():
                    teacher_output = self.teacher_model(inputs)  # Pas de gradient pour le teacher

                student_output = self.student_model(inputs)

                # Calcul de la divergence entre les distributions de sortie
                # Le loss KL mesure la différence entre la distribution de sortie du student
                # (après log_softmax) et celle du teacher (après softmax).
                loss = criterion(log_softmax(student_output), softmax(teacher_output))

                # Étapes classiques d'entraînement : remise à zéro des gradients,
                # rétropropagation du loss, mise à jour des poids.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            elapsed = time.time() - start_time
            avg_loss = epoch_loss / len(train_loader)

            # Affichage du résumé de l'époque : perte moyenne et temps écoulé
            print(f"✅ Distillation Epoch {epoch+1} terminé — Loss moyenne: {avg_loss:.4f} — Temps: {elapsed:.2f}s")

        # Sauvegarde du modèle étudiant après distillation
        torch.save(self.student_model.state_dict(), config.STUDENT_MODEL_PATH)
        return self.student_model
# src/pruning.py
import torch.nn.utils.prune as prune
import torch
from src import config

class PruningCompressor:
    """
    Classe responsable de la compression d'un modèle via le pruning (élagage).

    Le pruning consiste à supprimer des poids inutiles dans les couches du réseau
    afin de réduire sa complexité sans dégrader significativement ses performances.
    """

    def __init__(self, model):
        # Le modèle PyTorch à compresser est passé à l'initialisation
        self.model = model

    def apply(self):
        # Parcourt tous les sous-modules du modèle (ex: Conv2d, Linear)
        for name, module in self.model.named_modules():
            # Ne prune que les couches linéaires et convolutives
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                # Applique un élagage L1 non structuré sur les poids (40% des plus petits)
                prune.l1_unstructured(module, name='weight', amount=0.4)
                # Supprime les attributs de masquage temporaires et rend le modèle "propre"
                prune.remove(module, 'weight')

        # Sauvegarde le modèle compressé dans un fichier
        torch.save(self.model.state_dict(), config.PRUNED_MODEL_PATH)

        # Retourne le modèle élagué
        return self.model
# src/utils.py
import os
import torch

def count_parameters(model):
    """Retourne le nombre total de paramètres entraînables du modèle."""
    # On parcourt tous les paramètres du modèle et on additionne ceux qui sont entraînables
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(filepath):
    """Retourne la taille du fichier du modèle sauvegardé, en mégaoctets (Mo)."""
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)  # Taille brute en octets
        return round(size_bytes / (1024 * 1024), 2)  # Conversion en Mo avec arrondi à 2 décimales
    else:
        return 0.0  # Retourne 0 si le fichier n'existe pas

def evaluate_model(model, dataloader, device):
    """
    Évalue un modèle en mesurant son taux de bonnes prédictions (accuracy) sur un jeu de test.

    Args:
        model: Le modèle PyTorch à évaluer.
        dataloader: DataLoader contenant les données de test.
        device: 'cuda' ou 'cpu' selon le matériel disponible.

    Returns:
        L'accuracy en pourcentage, arrondie à 2 décimales.
    """
    model.eval()  # Mode évaluation : désactive le dropout et batchnorm
    correct = 0   # Compteur de bonnes prédictions
    total = 0     # Nombre total d'exemples

    with torch.no_grad():  # On désactive la rétropropagation pour accélérer l'inférence
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Prédictions brutes
            _, predicted = torch.max(outputs.data, 1)  # Indice de la classe prédite
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return round(100 * correct / total, 2)  # Calcul de l'accuracy en pourcentage
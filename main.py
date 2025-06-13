# main.py

# Imports principaux : PyTorch, JSON et modules du projet
import torch
import json
from src.trainer import ModelTrainer                 # Classe d'entraînement du modèle de base
from src.pruning import PruningCompressor           # Classe de compression par pruning
from src.quantization import QuantizationCompressor # Classe de compression par quantization dynamique
from src.distillation import KnowledgeDistiller     # Classe de distillation de connaissances
from src.utils import count_parameters, get_model_size, evaluate_model  # Fonctions utilitaires
from src.dataset import DataManager                 # Chargement des données
from src import config                              # Paramètres globaux du projet

# Fonction pour sauvegarder les résultats du pipeline dans un fichier JSON
def save_report(results):
    with open("models/report.json", "w") as f:
        json.dump(results, f, indent=2)

# Lancement du pipeline principal
if __name__ == "__main__":
    data = DataManager()
    _, test_loader = data.get_loaders(config.BATCH_SIZE)  # Récupère uniquement le jeu de test
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}

    # Étape 1 : Entraînement du modèle de base (non compressé)
    trainer = ModelTrainer()
    model = trainer.train_model()
    results["Baseline"] = {
        "Params": count_parameters(model),
        "Size (MB)": get_model_size(config.MODEL_PATH),
        "Accuracy": evaluate_model(model, test_loader, device)
    }

    # Étape 2 : Compression par pruning (élagage des poids inutiles)
    pruner = PruningCompressor(model)
    pruned_model = pruner.apply()
    results["Pruned"] = {
        "Params": count_parameters(pruned_model),
        "Size (MB)": get_model_size(config.PRUNED_MODEL_PATH),
        "Accuracy": evaluate_model(pruned_model, test_loader, device)
    }

    # Étape 3 : Compression par quantization dynamique
    quantizer = QuantizationCompressor(pruned_model)
    quantized_model = quantizer.apply()
    results["Quantized"] = {
        "Params": count_parameters(quantized_model),
        "Size (MB)": get_model_size(config.QUANTIZED_MODEL_PATH),
        "Accuracy": evaluate_model(quantized_model, test_loader, device)
    }

    # Étape 4 : Compression via distillation de connaissances (modèle étudiant)
    distiller = KnowledgeDistiller(teacher_model=quantized_model)
    student_model = distiller.apply()
    results["Student"] = {
        "Params": count_parameters(student_model),
        "Size (MB)": get_model_size(config.STUDENT_MODEL_PATH),
        "Accuracy": evaluate_model(student_model, test_loader, device)
    }

    # Sauvegarde du rapport final dans un fichier JSON
    save_report(results)

    print("\n✅ Pipeline terminé. Rapport enregistré dans models/report.json")
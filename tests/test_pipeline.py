# tests/test_pipeline.py

import sys
import os

# Ajout du chemin parent au PYTHONPATH pour permettre les imports relatifs à src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importation des modules nécessaires pour les tests
import torch
from src.model_factory import SimpleCNN
from src.utils import count_parameters, get_model_size, evaluate_model
from src.dataset import DataManager
from src.pruning import PruningCompressor
from src.quantization import QuantizationCompressor
from src.distillation import KnowledgeDistiller
import os

# Vérifie que le modèle SimpleCNN produit une sortie de forme correcte
def test_model_forward_shape():
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)  # batch de 1 image MNIST
    out = model(x)
    assert out.shape == (1, 10)  # 10 classes attendues

# Vérifie que le nombre de paramètres retourné est un entier positif
def test_count_parameters():
    model = SimpleCNN()
    total = count_parameters(model)
    assert isinstance(total, int)
    assert total > 0

# Vérifie que la taille du fichier modèle est mesurable et non nulle
def test_get_model_size():
    dummy_model_path = "models/test_model.pt"
    model = SimpleCNN()
    torch.save(model.state_dict(), dummy_model_path)
    size = get_model_size(dummy_model_path)
    assert size > 0.0
    os.remove(dummy_model_path)  # Nettoyage du fichier temporaire

# Vérifie que evaluate_model retourne une accuracy dans [0, 100]
def test_evaluate_model():
    model = SimpleCNN()
    data = DataManager()
    _, test_loader = data.get_loaders(32)
    acc = evaluate_model(model, test_loader, "cpu")
    assert 0 <= acc <= 100

# Vérifie que le processus de pruning ne casse pas le modèle
def test_pruning_runs():
    model = SimpleCNN()
    pruner = PruningCompressor(model)
    pruned_model = pruner.apply()
    assert isinstance(pruned_model, SimpleCNN)

# Vérifie que le processus de quantization retourne un module valide
def test_quantization_runs():
    model = SimpleCNN()
    quantizer = QuantizationCompressor(model)
    quantized = quantizer.apply()
    assert isinstance(quantized, torch.nn.Module)

# Vérifie que le modèle étudiant peut être généré via distillation
def test_distillation_runs():
    teacher = SimpleCNN()
    distiller = KnowledgeDistiller(teacher)
    student = distiller.apply()
    assert isinstance(student, SimpleCNN)
# src/config.py

# Taille des lots (batch size) utilisée lors de l'entraînement et l'évaluation
BATCH_SIZE = 64

# Nombre total d'époques (passes complètes sur le jeu de données) pour l'entraînement
EPOCHS = 5

# Taux d'apprentissage utilisé par l'optimiseur (Adam)
LEARNING_RATE = 0.001

# Chemin de sauvegarde du modèle de base (avant compression)
MODEL_PATH = "models/baseline_model.pt"

# Chemin de sauvegarde du modèle après application du pruning
PRUNED_MODEL_PATH = "models/pruned_model.pt"

# Chemin de sauvegarde du modèle après quantization dynamique
QUANTIZED_MODEL_PATH = "models/quantized_model.pt"

# Chemin de sauvegarde du modèle étudiant après distillation de connaissances
STUDENT_MODEL_PATH = "models/student_model.pt"
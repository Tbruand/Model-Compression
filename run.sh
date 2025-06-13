#!/bin/bash

echo "🚀 Initialisation de l'entraîneur ResNet50 pour MNIST..."
echo "📦 Installation des dépendances..."
pip install -r requirements.txt

echo "🎯 Lancement du pipeline complet..."
python main.py

echo "✅ Terminé ! Les modèles sont disponibles dans le dossier models/"
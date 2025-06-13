# README.md
# 🧠 Model Compression Project

Ce projet montr comment appliquer différentes techniques de compression de modèles de deep learning en PyTorch :
- Pruning (élagage)
- Quantization (quantification)
- Knowledge Distillation (distillation de connaissances)

## 📁 Structure
```
model-compression/
├── data/               # Données (ex : MNIST)
├── models/             # Modèles enregistrés
├── src/                # Code source principal
├── tests/              # Tests unitaires (à venir)
├── notebooks/          # Analyses et visualisations
├── main.py             # Lancement du pipeline complet
├── run.sh              # Script d'automatisation
├── requirements.txt    # Dépendances
└── README.md           # Ce fichier
```

## 🚀 Lancement rapide

Vous avez deux options pour lancer le projet :

### ✅ Option 1 : avec le script automatique `run.sh` (recommandé)

```bash
chmod +x run.sh
./run.sh
```

Ce script effectue automatiquement :
1. L'installation des dépendances via `pip install -r requirements.txt`
2. L'exécution du pipeline complet (entraînement + compression)
3. La sauvegarde des modèles compressés dans le dossier `models/`
4. L'enregistrement des métriques dans `models/report.json`

### 🛠️ Option 2 : sans script bash (manuel)

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer le pipeline manuellement
python main.py
```

Cette méthode produit les mêmes résultats que le script, utile si vous ne pouvez pas exécuter de `.sh` sur votre système.

## 🔧 Dépendances
```bash
pip install -r requirements.txt
```

## 🔍 Références
- https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
- https://pytorch.org/docs/stable/quantization.html
- https://pytorch.org/tutorials/intermediate/distiller.html

## 📊 Résultats

À la fin de l'exécution du pipeline (`main.py`), un fichier `models/report.json` est généré contenant les métriques suivantes pour chaque modèle :

```json
{
  "Baseline": {"Params": 421642, "Size (MB)": 1.61, "Accuracy": 99.02},
  "Pruned":   {"Params": 421642, "Size (MB)": 1.61, "Accuracy": 98.88},
  "Quantized":{"Params": 18816,  "Size (MB)": 0.46, "Accuracy": 98.87},
  "Student":  {"Params": 421642, "Size (MB)": 1.61, "Accuracy": 98.82}
}
```

## 📓 Notebook

Un notebook `notebooks/visualisation.ipynb` permet de charger et visualiser les performances des modèles (taille, nombre de paramètres, accuracy) à partir du fichier `report.json`.

Lancer dans Jupyter :

```bash
jupyter notebook notebooks/visualisation.ipynb
```
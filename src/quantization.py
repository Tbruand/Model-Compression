# src/quantization.py

# Import des dépendances nécessaires
import torch
from src import config

class QuantizationCompressor:
    """
    Cette classe applique une quantization dynamique à un modèle PyTorch.

    La quantization permet de réduire la taille et les calculs d'un modèle
    en convertissant certains poids de float32 vers des entiers (ex: int8).
    """

    def __init__(self, model):
        # Le modèle à compresser est passé au constructeur
        self.model = model

    def apply(self):
        # Place le modèle en mode évaluation (obligatoire avant quantization)
        self.model.eval()

        # Déclare le moteur de quantization à utiliser (ici qnnpack pour CPU ARM)
        torch.backends.quantized.engine = "qnnpack"

        # Applique la quantization dynamique : seuls les modules Linear sont affectés ici
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,                          # modèle source
            {torch.nn.Linear},                   # types de modules à quantifier
            dtype=torch.qint8                    # type de quantization
        )

        # Sauvegarde du modèle quantifié
        torch.save(quantized_model.state_dict(), config.QUANTIZED_MODEL_PATH)

        # Retourne le modèle quantifié
        return quantized_model
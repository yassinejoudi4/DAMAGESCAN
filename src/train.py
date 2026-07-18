"""
train.py

Entraîne le modèle (VGG16 gelé + couches de classification personnalisées)
sur le dataset d'entraînement, valide à chaque epoch, puis sauvegarde
le modèle final dans models/ et l'historique d'entraînement dans outputs/.
"""

import os
import json
from preprocess import train_ds, valid_ds
from model import model

EPOCHS = 10

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS
)

os.makedirs("models", exist_ok=True)
model.save("models/vgg16_damage_classifier.keras")
print("Modèle sauvegardé dans models/vgg16_damage_classifier.keras")

os.makedirs("outputs", exist_ok=True)
with open("outputs/training_history.json", "w") as f:
    json.dump(history.history, f, indent=2)
print("Historique sauvegardé dans outputs/training_history.json")
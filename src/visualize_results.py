"""
visualize_results.py

Charge le modèle entraîné, fait des prédictions sur le dataset test,
affiche et sauvegarde une matrice de confusion dans outputs/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from preprocess import test_ds

model = tf.keras.models.load_model("models/vgg16_damage_classifier.keras")

class_names = ["crack", "dent"]

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    y_true.extend(labels.numpy().flatten().astype(int))
    y_pred.extend(predicted_classes)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix.png")
print("Matrice de confusion sauvegardée dans outputs/confusion_matrix.png")

plt.show()
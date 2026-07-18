"""
evaluate.py

Charge le modèle entraîné sauvegardé, l'évalue sur le dataset test/
(jamais vu pendant l'entraînement ni la validation), et affiche
les métriques finales de performance.
"""

import tensorflow as tf
from preprocess import test_ds

model = tf.keras.models.load_model("models/vgg16_damage_classifier.keras")

loss, accuracy = model.evaluate(test_ds)

print(f"Test loss : {loss:.4f}")
print(f"Test accuracy : {accuracy:.4f}")
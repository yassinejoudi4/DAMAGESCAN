"""
model.py

Charge VGG16 pré-entraîné (sur ImageNet) comme extracteur de features,
gèle ses poids, et ajoute des couches de classification personnalisées
pour distinguer "crack" et "dent".
"""

import tensorflow as tf
from keras.applications import VGG16
from keras import layers, models

IMG_SIZE = (224, 224)

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
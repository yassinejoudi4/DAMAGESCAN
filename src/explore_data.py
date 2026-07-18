"""
preprocess.py

Charge les images du dataset (train/valid/test) depuis le dossier data/,
les redimensionne en 224x224 (format attendu par VGG16), les regroupe en
batchs, et normalise les pixels (valeurs entre 0 et 1).
"""
import os
import matplotlib.pyplot as plt
from PIL import Image

data_dir = os.path.join("data", "aircraft_damage_dataset_v1", "train")

categories = ["crack", "dent"]


fig, axes = plt.subplots(2, 10, figsize=(12, 8))

for row, category in enumerate(categories):
    category_path = os.path.join(data_dir, category)
    image_files = os.listdir(category_path)[:10]  # on prend les 3 premières images

    for col, image_file in enumerate(image_files):
        image_path = os.path.join(category_path, image_file)
        img = Image.open(image_path)

        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{category}")
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()
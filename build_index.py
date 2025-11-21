# build_index.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf

DATASET_DIR = "dataset"   # your images root folder
IMG_SIZE = 224

# create MobileNetV2 feature extractor
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])

# preprocessing helper
def load_and_preprocess(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # correct preprocessing
    return x

emb_list = []
labels = []
image_paths = []

# Walk folder
for class_name in sorted(os.listdir(DATASET_DIR)):
    class_folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_folder):
        continue
    files = [f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for fname in files:
        p = os.path.join(class_folder, fname)
        try:
            x = load_and_preprocess(p)
            emb = model.predict(np.expand_dims(x, axis=0))  # shape (1, dim)
            emb = emb.flatten()
            # L2-normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            emb_list.append(emb)
            labels.append(class_name)
            image_paths.append(p)
        except Exception as e:
            print("Failed on", p, e)

# Convert and save
embeddings = np.vstack(emb_list).astype("float32")  # shape (N, D)

np.savez_compressed("embeddings.npz",
                    embeddings=embeddings,
                    labels=np.array(labels),
                    image_paths=np.array(image_paths))

print("Saved embeddings.npz")
print("Num images:", embeddings.shape[0], "Dimension:", embeddings.shape[1])

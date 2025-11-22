# query_image.py
import numpy as np
from PIL import Image
import tensorflow as tf
import os

EMBED_FILE = "embeddings.npz"
IMG_SIZE = 224
TOP_K = 3

# load index
data = np.load(EMBED_FILE, allow_pickle=True)
db_embeddings = data["embeddings"]     # shape (N, D)
db_labels = data["labels"]             # shape (N,)
db_image_paths = data["image_paths"]   # shape (N,)

# load model same as before
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])

def preprocess_image(path_or_pil):
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert("RGB")
    else:
        img = path_or_pil.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x

def get_embedding_from_array(x):
    emb = model.predict(np.expand_dims(x, axis=0))[0]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def find_similar(image_path, top_k=TOP_K):
    x = preprocess_image(image_path)
    q_emb = get_embedding_from_array(x)  # shape (D,)

    # cosine similarity = dot product because both are unit-norm
    sims = np.dot(db_embeddings, q_emb)  # shape (N,)
    # get top K indices
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for i in idxs:
        results.append({
            "label": str(db_labels[i]),
            "image_path": str(db_image_paths[i]),
            "score": float(sims[i])
        })
    return results

# Quick test if you run this file directly:
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python query_image.py <path_to_query_image>")
        sys.exit(1)
    q = sys.argv[1]
    res = find_similar(q, top_k=5)
    for r in res:
        print(r)

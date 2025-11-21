import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

DATASET = "dataset"
IMG_SIZE = 224
BATCH = 16

# Data generator
datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation"
)

# Transfer Learning Model
base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=8
)

# Save the model
model.save("medicine_model.h5")

print("Model training completed.")
print("Number of classes:", train_data.num_classes)

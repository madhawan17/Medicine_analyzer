import os

DATASET = "dataset"

print("Checking dataset...")

if not os.path.exists(DATASET):
    print("❌ Dataset folder NOT found")
    quit()

subfolders = os.listdir(DATASET)

print("\nClasses found:")
for folder in subfolders:
    print(" -", folder)

print("\nImage count per class:")
for folder in subfolders:
    path = os.path.join(DATASET, folder)
    images = [f for f in os.listdir(path) if f.lower().endswith((".jpg",".png",".jpeg"))]
    print(folder, "→", len(images), "images")


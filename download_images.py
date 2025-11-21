import pandas as pd
import requests
import os
import time

CSV_FILE = "Medicine_Details.csv"  # change if needed
OUTPUT_FOLDER = "dataset"

df = pd.read_csv(CSV_FILE)

for index, row in df.iterrows():
    medicine = str(row["Medicine Name"]).strip()
    img_url = row["Image URL"]

    # Create a folder for each medicine
    folder_path = os.path.join(OUTPUT_FOLDER, medicine)
    os.makedirs(folder_path, exist_ok=True)

    # File name
    img_path = os.path.join(folder_path, f"{medicine}_{index}.jpg")

    try:
        print(f"Downloading {img_url} → {img_path}")
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()

        with open(img_path, "wb") as file:
            file.write(response.content)

        time.sleep(0.2)  # small delay to avoid rate limiting

    except Exception as e:
        print(f"Failed to download {img_url}: {e}")

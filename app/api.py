from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
import pandas as pd

from query_image import find_similar

app = FastAPI()

templates = Jinja2Templates(directory="templates")

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------------------
# 🔹 LOAD CSV DATA ONCE
# ---------------------------
CSV_FILE = "medicine_data.csv"   # rename your uploaded CSV to this

df = pd.read_csv(CSV_FILE)

# Create dictionary: {Medicine Name → Row Details}
DETAILS_DB = {}

for i, row in df.iterrows():
    name = str(row["Medicine Name"]).strip()
    DETAILS_DB[name] = row.to_dict()


# ---------------------------
# 🔹 FRONTEND
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------
# 🔹 AI PREDICTION API
# ---------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    temp_path = os.path.join(TEMP_DIR, file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # run similarity search
    results = find_similar(temp_path, top_k=3)
    top = results[0] if results else None

    # decision logic
    if top and top["score"] >= 0.82:
        verdict = "match"
    elif top and top["score"] >= 0.65:
        verdict = "low_confidence"
    else:
        verdict = "no_match"

    # ---------------------------
    # 🔹 Get medicine details
    # ---------------------------
    details = {}
    if top:
        med_name = top["label"]
        if med_name in DETAILS_DB:
            details = DETAILS_DB[med_name]
        else:
            details = {"error": "No details found for this medicine in CSV"}

    return {
        "verdict": verdict,
        "results": results,
        "details": details        # return ALL medicine details here
    }

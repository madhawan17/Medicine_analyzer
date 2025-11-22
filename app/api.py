from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
import pandas as pd
from dotenv import load_dotenv

from app.query_image import find_similar

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------------------
# LOAD CSV DATA
# ---------------------------
CSV_FILE = "medicine_data.csv"
df = pd.read_csv(CSV_FILE)
DETAILS_DB = { str(row["Medicine Name"]).strip(): row.to_dict() for _, row in df.iterrows() }

# ---------------------------
# FRONTEND ROUTE
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
             "groq_key": os.getenv("GROQ_API_KEY") # secure injection
        }
    )

# ---------------------------
# ANALYZE ROUTE
# ---------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    temp_path = os.path.join(TEMP_DIR, file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = find_similar(temp_path, top_k=3)
    top = results[0] if results else None

    if top and top["score"] >= 0.82:
        verdict = "match"
    elif top and top["score"] >= 0.65:
        verdict = "low_confidence"
    else:
        verdict = "no_match"

    details = DETAILS_DB.get(top["label"], {}) if top else {}

    return {
        "verdict": verdict,
        "results": results,
        "details": details
    }

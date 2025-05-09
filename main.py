import uvicorn
from fastapi import FastAPI
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import pandas as pd

from pipelines.inference import *

UPLOAD_DIR = "uploads"
RESULT_FILE = f"{UPLOAD_DIR}/predicted_output.csv"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



PATH_CONFIG = "config_fastapi_inf.yaml"

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "file_ready": os.path.exists(RESULT_FILE)})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    uploaded_path = f"{UPLOAD_DIR}/{file.filename}"
    
    with open(uploaded_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and predict
    df=run_inference(PATH_CONFIG, uploaded_path)
    # Save result
    df.to_csv(RESULT_FILE, index=False)
    

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": "Inference completed. Download is now available.",
        "file_ready": True
    })

@app.get("/download")
async def download_file():
    if os.path.exists(RESULT_FILE):
        return FileResponse(path=RESULT_FILE, filename="predicted_output.csv", media_type="text/csv")
    return {"error": "No file available"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
import os
import tempfile
from fastapi import FastAPI, File, UploadFile
import shutil
from src.model import Model

app = FastAPI(title="SAR-Buildings-Detection")

MODEL_PATH = "path/to/weights.pt"

model = Model(MODEL_PATH)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        tmp_path = tmp.name

    model.predict(image_path=tmp_path, chunk_size=1024)

    return 'Succesfull!'
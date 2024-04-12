from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import tempfile
import shutil
import cv2
import uuid
import shortuuid
import os
from src.model import Model
import sqlite3
from datetime import datetime

app = FastAPI(title="SAR-Buildings-Detection")

conn = sqlite3.connect('/app/data/SQLite/prediction.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS predictions
                  (id INTEGER PRIMARY KEY,
                  original TEXT,
                  prediction TEXT,
                  request_time TIMESTAMP)''')
conn.commit()

MODEL_PATH = "/app/data/train64_base/weights/last.pt"
OUTPUT_FOLDER = "/app/data/SQLite/predictions"
ORIGINAL_FOLDER = "/app/data/SQLite/originals"

model = Model(MODEL_PATH)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    original_path = os.path.join(ORIGINAL_FOLDER, file.filename)
    with open(original_path, "wb") as original_file:
        shutil.copyfileobj(file.file, original_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfile(original_path, tmp.name)
        tmp_path = tmp.name

    image_uuid = shortuuid.uuid()[:6]

    filename, file_extension = os.path.splitext(file.filename)
    new_filename = f"{filename}_{image_uuid}{file_extension}"
    output_file_path = os.path.join(OUTPUT_FOLDER, new_filename)

    pred = model.predict(image_path=tmp_path, chunk_size=512)

    image_pred_url = f"/images/predictions/{new_filename}"
    image_orig_url = f"/images/originals/{file.filename}"

    cv2.imwrite(output_file_path, pred, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    request_time = datetime.now()
    cursor.execute("INSERT INTO predictions (original, prediction, request_time) VALUES (?, ?, ?)",
                   (image_orig_url, image_pred_url, request_time))
    conn.commit()

    return {
        "image_pred_url": image_pred_url,
        "image_orig_url": image_orig_url
    }

@app.get("/images/predictions/{filename}")
async def get_prediction_image(filename: str):
    image_path = os.path.join(OUTPUT_FOLDER, filename)
    return FileResponse(image_path)

@app.get("/images/originals/{filename}")
async def get_original_image(filename: str):
    image_path = os.path.join(ORIGINAL_FOLDER, filename)
    return FileResponse(image_path)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

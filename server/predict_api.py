import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from server.model_manager import ModelManager

router = APIRouter()

# One model manager for the whole server process
mm = ModelManager()

ALLOWED = ("image/jpeg", "image/png", "image/bmp", "image/webp")

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    os.makedirs("server/storage/incoming", exist_ok=True)
    ext = os.path.splitext(file.filename)[1].lower() or ".img"
    temp_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join("server", "storage", "incoming", temp_name)

    data = await file.read()
    with open(temp_path, "wb") as f:
        f.write(data)

    return mm.predict_path(temp_path)


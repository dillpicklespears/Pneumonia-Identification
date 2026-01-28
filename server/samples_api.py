import os
import uuid
import shutil
import threading
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from server.train_job import train_and_publish
from server.build_combined_train import main as build_combined_train

router = APIRouter()

ALLOWED = ("image/jpeg", "image/png", "image/bmp", "image/webp")

BASE_DIR = "server/storage"
COUNTER_FILE = os.path.join(BASE_DIR, "labeled_since_retrain.txt")
AUTO_RETRAIN_THRESHOLD = 20

# Prevent multiple retrains at the same time
_retrain_lock = threading.Lock()


def _read_counter() -> int:
    try:
        with open(COUNTER_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip() or "0")
    except FileNotFoundError:
        return 0
    except Exception:
        return 0


def _write_counter(value: int) -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(COUNTER_FILE, "w", encoding="utf-8") as f:
        f.write(str(value))


def _background_retrain():
    # Only one retrain at a time
    if not _retrain_lock.acquire(blocking=False):
        return

    try:
        # reset counter immediately so new uploads start counting fresh
        _write_counter(0)

        # rebuild combined train + train + publish (updates active.pth)
        build_combined_train()
        info = train_and_publish(num_epochs=2)  # increase later if you want

        # Optional: write a small log file
        log_path = os.path.join("server", "storage", "last_retrain.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"timestamp_utc={datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}\n")
            for k, v in info.items():
                f.write(f"{k}={v}\n")

    except Exception as e:
        # If retrain fails, don't lose the count forever:
        # set counter back to threshold-1 so next labeled upload retriggers quickly.
        _write_counter(AUTO_RETRAIN_THRESHOLD - 1)

        err_path = os.path.join("server", "storage", "last_retrain_error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(str(e))

    finally:
        _retrain_lock.release()


@router.post("/samples")
async def upload_sample(
    file: UploadFile = File(...),
    label: str | None = Form(None),      # "NORMAL" or "PNEUMONIA" (optional)
    source_id: str | None = Form(None),  # optional identifier
):
    if file.content_type not in ALLOWED:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    # Normalize/validate label if provided
    if label is not None:
        label = label.strip().upper()
        if label not in ("NORMAL", "PNEUMONIA"):
            raise HTTPException(status_code=400, detail="label must be NORMAL or PNEUMONIA")

    # Storage paths
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "incoming"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "labeled"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "unlabeled"), exist_ok=True)

    ext = os.path.splitext(file.filename)[1].lower() or ".img"
    sample_id = uuid.uuid4().hex
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Decide folder based on label
    if label:
        out_dir = os.path.join(BASE_DIR, "labeled", label)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.join(BASE_DIR, "unlabeled")

    out_path = os.path.join(out_dir, f"{timestamp}_{sample_id}{ext}")

    data = await file.read()
    with open(out_path, "wb") as f:
        f.write(data)

    # Metadata sidecar
    meta_path = out_path + ".meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"sample_id={sample_id}\n")
        f.write(f"timestamp_utc={timestamp}\n")
        f.write(f"original_filename={file.filename}\n")
        f.write(f"content_type={file.content_type}\n")
        f.write(f"label={label or ''}\n")
        f.write(f"source_id={source_id or ''}\n")

    # Auto-retrain logic: only count LABELED samples
    triggered = False
    new_count = None
    if label:
        current = _read_counter()
        new_count = current + 1
        _write_counter(new_count)

        if new_count >= AUTO_RETRAIN_THRESHOLD:
            triggered = True
            t = threading.Thread(target=_background_retrain, daemon=True)
            t.start()

    return {
        "ok": True,
        "sample_id": sample_id,
        "saved_path": out_path,
        "label": label,
        "labeled_since_retrain": new_count,
        "auto_retrain_triggered": triggered,
        "auto_retrain_threshold": AUTO_RETRAIN_THRESHOLD,
    }


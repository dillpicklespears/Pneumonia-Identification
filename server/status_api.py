import os
from datetime import datetime
from fastapi import APIRouter

router = APIRouter()

BASE_DIR = "server/storage"
COUNTER_FILE = os.path.join(BASE_DIR, "labeled_since_retrain.txt")
LAST_RETRAIN_FILE = os.path.join(BASE_DIR, "last_retrain.txt")
LAST_RETRAIN_ERR = os.path.join(BASE_DIR, "last_retrain_error.txt")
ACTIVE_MODEL = os.path.join("server", "models", "active.pth")

def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def _read_int(path: str) -> int:
    try:
        txt = _read_text(path)
        return int(txt) if txt else 0
    except Exception:
        return 0

def _mtime(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")

@router.get("/status")
def status():
    labeled_since = _read_int(COUNTER_FILE)

    return {
        "ok": True,
        "labeled_since_retrain": labeled_since,
        "active_model_path": ACTIVE_MODEL,
        "active_model_last_modified": _mtime(ACTIVE_MODEL),
        "last_retrain_log": _read_text(LAST_RETRAIN_FILE),
        "last_retrain_error": _read_text(LAST_RETRAIN_ERR),
    }

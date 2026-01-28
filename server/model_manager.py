import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from loaddata import LoadData

# NOTE: your model is saved as a full model object (.pth)
# We'll support loading it safely with map_location
ACTIVE_MODEL_PATH = "server/models/active.pth"

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = LoadData()
        self.model = None
        self._last_mtime = None
        self.load_active()

    def load_active(self):
        if not os.path.exists(ACTIVE_MODEL_PATH):
            raise FileNotFoundError(f"Missing active model: {ACTIVE_MODEL_PATH}")

        mtime = os.path.getmtime(ACTIVE_MODEL_PATH)
        self._last_mtime = mtime

        self.model = torch.load(ACTIVE_MODEL_PATH, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()

    def maybe_reload(self):
        # Reload if file changed (hot swap)
        if not os.path.exists(ACTIVE_MODEL_PATH):
            return
        mtime = os.path.getmtime(ACTIVE_MODEL_PATH)
        if self._last_mtime is None or mtime != self._last_mtime:
            self.load_active()

    def predict_path(self, image_path: str):
        self.maybe_reload()

        img = Image.open(image_path).convert("RGB")
        x = self.data.val_test_transforms(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)

        result = {
            "class_id": int(idx.item()),
            "class_name": self.data.class_names[int(idx.item())],
            "confidence": float(conf.item()),
            "probabilities": probs.detach().cpu().numpy(),
        }

        # JSON serialize
        if isinstance(result["probabilities"], np.ndarray):
            result["probabilities"] = result["probabilities"].tolist()

        return result


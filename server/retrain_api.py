from fastapi import APIRouter, HTTPException

from server.train_job import train_and_publish
from server.build_combined_train import main as build_combined_train

router = APIRouter()

@router.post("/retrain")
def retrain():
    try:
        # 1) Rebuild combined training dataset (base + newly labeled)
        build_combined_train()

        # 2) Train and publish a new model (writes version + updates active.pth)
        info = train_and_publish(num_epochs=2)

        return {"ok": True, **info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

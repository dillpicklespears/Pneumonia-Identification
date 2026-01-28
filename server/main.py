from fastapi import FastAPI
from server.predict_api import router as predict_router
from server.samples_api import router as samples_router
from server.retrain_api import router as retrain_router
from server.status_api import router as status_router

app = FastAPI(title="PneuVision Server")

@app.get("/health")
def health():
    return {"ok": True}

app.include_router(predict_router)
app.include_router(samples_router)
app.include_router(retrain_router)
app.include_router(status_router)

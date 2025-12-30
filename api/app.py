from fastapi import FastAPI, Request
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager
import pickle
import threading

ROOT = Path(__file__).resolve().parents[1]  # mlops/
MODEL_PATH = ROOT / "models" / "model.pkl"

# A lock to avoid concurrent reload/read issues
model_lock = threading.Lock()

def load_model_from_disk():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty model file: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model_from_disk()
    yield

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    user_id: int

@app.get("/health")
def health(request: Request):
    with model_lock:
        m = request.app.state.model
    return {
        "status": "ok",
        "model_type": m.get("type", "unknown"),
        "model_path": str(MODEL_PATH),
    }

@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    with model_lock:
        model = request.app.state.model

    user_recs = model["user_top_products"].get(req.user_id)
    if user_recs:
        return {"user_id": req.user_id, "recommendations": user_recs[:5], "reason": "personalized"}
    return {"user_id": req.user_id, "recommendations": model["global_top_products"][:5], "reason": "popular_fallback"}

@app.post("/reload-model")
def reload_model(request: Request):
    new_model = load_model_from_disk()
    with model_lock:
        request.app.state.model = new_model
    return {
        "status": "reloaded",
        "model_type": new_model.get("type", "unknown"),
        "model_path": str(MODEL_PATH),
        "model_bytes": MODEL_PATH.stat().st_size,
    }

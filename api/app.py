from fastapi import FastAPI, Request
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager
import threading
from urllib.parse import urlparse
import boto3
import os
import pickle
from urllib.parse import urlparse

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



MODEL = None

def download_from_s3(s3_uri: str, local_path: str) -> str:
    # s3://bucket/key
    u = urlparse(s3_uri)
    bucket = u.netloc
    key = u.path.lstrip("/")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    s3.download_file(bucket, key, local_path)
    return local_path

def load_model() -> None:
    global MODEL
    s3_uri = os.getenv("MODEL_S3_URI")
    local_path = os.getenv("MODEL_LOCAL_PATH", "/tmp/model.pkl")

    if s3_uri:
        download_from_s3(s3_uri, local_path)

    with open(local_path, "rb") as f:
        MODEL = pickle.load(f)


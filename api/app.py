from __future__ import annotations

import hashlib
import os
import pickle
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import boto3
from fastapi import FastAPI, Request
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "model.pkl"

model_lock = threading.Lock()

# ---------------------------
# File metadata helpers
# ---------------------------
def file_size_bytes(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except Exception:
        return None


def file_mtime_utc_iso(path: Path) -> Optional[str]:
    try:
        st = path.stat()
        return datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def file_sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def build_model_meta(
    model: Dict[str, Any],
    model_path: Path,
    loaded_ms: int,
) -> Dict[str, Any]:
    """
    Compute metadata ONCE at load time (or reload time), not per-request.
    """
    sha = file_sha256(model_path)
    size = file_size_bytes(model_path)
    mtime_utc = file_mtime_utc_iso(model_path)

    return {
        "status": "ok",
        "model_type": model.get("type", "unknown"),
        "model_path": str(model_path),

        # File identity (this is what you compare to S3)
        "model_sha256": sha,
        "model_sha256_short": sha[:8] if sha else None,

        # File facts
        "model_size_bytes": size,
        "model_bytes": size,  # alias for convenience
        "model_mtime_utc": mtime_utc,

        # Load timing
        "loaded_at_unix": int(time.time()),
        "load_ms": loaded_ms,
    }


# ---------------------------
# Optional S3 download support
# ---------------------------
def download_from_s3(s3_uri: str, local_path: Path) -> Path:
    u = urlparse(s3_uri)
    if u.scheme != "s3" or not u.netloc or not u.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    bucket = u.netloc
    key = u.path.lstrip("/")

    local_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    s3.download_file(bucket, key, str(local_path))
    return local_path


def resolve_model_path() -> Path:
    env_model_path = os.getenv("MODEL_PATH")
    model_path = Path(env_model_path) if env_model_path else DEFAULT_MODEL_PATH

    s3_uri = os.getenv("MODEL_S3_URI")
    if s3_uri:
        local_override = os.getenv("MODEL_LOCAL_PATH")
        download_target = Path(local_override) if local_override else model_path
        download_from_s3(s3_uri, download_target)
        model_path = download_target

    return model_path


def load_model_from_disk(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty model file: {model_path}")

    with model_path.open("rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise RuntimeError(f"Loaded model is not a dict. Got: {type(obj)}")

    return obj


def load_model_bundle() -> tuple[Dict[str, Any], Path, Dict[str, Any]]:
    """
    Loads model + computes metadata once.
    Returns: (model, model_path, model_meta)
    """
    t0 = time.perf_counter()
    model_path = resolve_model_path()
    model = load_model_from_disk(model_path)
    loaded_ms = int((time.perf_counter() - t0) * 1000)
    meta = build_model_meta(model, model_path, loaded_ms)
    return model, model_path, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, model_path, meta = load_model_bundle()
    with model_lock:
        app.state.model = model
        app.state.model_path = model_path
        app.state.model_meta = meta
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    user_id: int


# ---------------------------
# Health & readiness
# ---------------------------
@app.get("/health")
def health(request: Request):
    """
    Liveness: FAST.
    Returns cached metadata computed at load time.
    """
    with model_lock:
        meta = getattr(request.app.state, "model_meta", None)

    return meta or {"status": "ok", "service": "inference"}


@app.get("/ready")
def ready(request: Request):
    """
    Readiness: only OK when model and metadata are loaded.
    """
    with model_lock:
        model = getattr(request.app.state, "model", None)
        meta = getattr(request.app.state, "model_meta", None)

    if model is None:
        return {"status": "not_ready", "reason": "model_not_loaded"}
    if not meta or not meta.get("model_sha256") or not meta.get("model_size_bytes"):
        return {"status": "not_ready", "reason": "model_metadata_missing"}

    return {"status": "ready"}


@app.get("/api/inference/health")
def health_alias(request: Request):
    return health(request)


@app.get("/api/inference/ready")
def ready_alias(request: Request):
    return ready(request)


# ---------------------------
# Inference
# ---------------------------
@app.post("/recommendations")
def recommendations(req: PredictRequest, request: Request):
    with model_lock:
        model = request.app.state.model

    # Support a couple common shapes
    rec_map = model.get("recommendations") or {}
    recs = rec_map.get(req.user_id) or model.get("popular") or [1, 2, 3, 101, 102]

    return {
        "user_id": req.user_id,
        "recommendations": recs,
        "source": "ml",
    }

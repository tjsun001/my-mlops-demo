from __future__ import annotations

import hashlib
import json
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

# Pointer defaults (Path A)
DEFAULT_POINTER_S3_URI = "s3://thurmans-demo-models/mlops/models/recommender/production.json"
DEFAULT_POINTER_LOCAL_PATH = Path("/tmp/production.json")

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
    *,
    pointer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute metadata ONCE at load time (or reload time), not per-request.
    """
    sha = file_sha256(model_path)
    size = file_size_bytes(model_path)
    mtime_utc = file_mtime_utc_iso(model_path)

    meta: Dict[str, Any] = {
        "status": "ok",
        "model_type": model.get("type", "unknown"),
        "model_path": str(model_path),
        # File identity
        "model_sha256": sha,
        "model_sha256_short": sha[:8] if sha else None,
        # File facts
        "model_size_bytes": size,
        "model_bytes": size,  # alias
        "model_mtime_utc": mtime_utc,
        # Load timing
        "loaded_at_unix": int(time.time()),
        "load_ms": loaded_ms,
    }

    if pointer:
        meta["model_pointer_uri"] = pointer.get("pointer_uri")
        meta["model_pointer_local_path"] = pointer.get("pointer_local_path")
        meta["s3_bucket"] = pointer.get("bucket")
        meta["s3_key"] = pointer.get("key")
        meta["s3_version_id"] = pointer.get("version_id")
        meta["semantic_version"] = pointer.get("semantic_version")
        meta["promoted_at"] = pointer.get("promoted_at")

    return meta


# ---------------------------
# S3 helpers (supports VersionId + pointer file)
# ---------------------------
def _s3_client():
    # boto3 uses instance profile automatically if present
    region = os.getenv("AWS_REGION", "us-east-1")
    return boto3.client("s3", region_name=region)


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    u = urlparse(s3_uri)
    if u.scheme != "s3" or not u.netloc or not u.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return u.netloc, u.path.lstrip("/")


def download_from_s3(
    s3_uri: str,
    local_path: Path,
    *,
    version_id: Optional[str] = None,
) -> Path:
    """
    Download an S3 object to a local path. Supports VersionId if provided.
    """
    bucket, key = _parse_s3_uri(s3_uri)

    local_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = _s3_client()
    if version_id:
        # download_file doesn't support VersionId; use get_object
        obj = s3.get_object(Bucket=bucket, Key=key, VersionId=version_id)
        local_path.write_bytes(obj["Body"].read())
    else:
        s3.download_file(bucket, key, str(local_path))

    return local_path


def ensure_model_present_from_pointer(pointer_s3_uri: str, model_path: Path) -> Dict[str, Any]:
    """
    Path A: Download production.json, then download model.pkl BY VersionId to model_path.
    Returns pointer metadata for /health and /model/info.
    """
    pointer_local_path = Path(os.getenv("MODEL_POINTER_LOCAL_PATH", str(DEFAULT_POINTER_LOCAL_PATH)))
    download_from_s3(pointer_s3_uri, pointer_local_path)

    pointer = json.loads(pointer_local_path.read_text())
    art = pointer["model"]["artifact"]

    bucket = art["bucket"]
    key = art["key"]
    version_id = (art.get("version_id") or "").strip()
    if not version_id:
        raise RuntimeError("production.json missing model.artifact.version_id")

    # Download model by VersionId to the path FastAPI will load from
    model_uri = f"s3://{bucket}/{key}"
    download_from_s3(model_uri, model_path, version_id=version_id)

    # Optional integrity check if sha256 is provided
    expected_sha = (art.get("sha256") or "").strip()
    if expected_sha:
        actual_sha = file_sha256(model_path)
        if actual_sha and actual_sha != expected_sha:
            raise RuntimeError(f"SHA mismatch: expected={expected_sha}, actual={actual_sha}")

    # Sanity check
    if (not model_path.exists()) or model_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded model is missing/empty: {model_path}")

    return {
        "pointer_uri": pointer_s3_uri,
        "pointer_local_path": str(pointer_local_path),
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "semantic_version": pointer.get("release", {}).get("semantic_version", ""),
        "promoted_at": pointer.get("release", {}).get("promoted_at"),
    }


def resolve_model_path() -> Path:
    """
    Resolve the local path we should load the model from.

    Path A (preferred):
      - MODEL_POINTER_S3_URI is set
      - We fetch production.json and then fetch model by VersionId into MODEL_PATH (default /tmp/model.pkl in container)

    Legacy direct download mode (optional):
      - MODEL_S3_URI points directly to an object (non-versioned)
    """
    env_model_path = os.getenv("MODEL_PATH")
    model_path = Path(env_model_path) if env_model_path else DEFAULT_MODEL_PATH

    # In Path A, we still load from model_path, but we ensure it's downloaded first in load_model_bundle().
    return model_path


# ---------------------------
# Model loading
# ---------------------------
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

    Behavior:
      - If MODEL_POINTER_S3_URI is set (Path A), download production.json and versioned model.pkl into model_path first.
      - Else if MODEL_S3_URI is set (legacy direct object), download it to model_path first.
      - Else load from local disk path.
    """
    t0 = time.perf_counter()

    model_path = resolve_model_path()

    pointer_meta: Optional[Dict[str, Any]] = None
    pointer_uri = os.getenv("MODEL_POINTER_S3_URI")

    if pointer_uri:
        # Always ensure the correct version is present at startup/reload.
        pointer_meta = ensure_model_present_from_pointer(pointer_uri, model_path)
    else:
        # Legacy direct S3 URI mode (optional)
        s3_uri = os.getenv("MODEL_S3_URI")
        if s3_uri:
            local_override = os.getenv("MODEL_LOCAL_PATH")
            download_target = Path(local_override) if local_override else model_path
            download_from_s3(s3_uri, download_target)
            model_path = download_target

    model = load_model_from_disk(model_path)
    loaded_ms = int((time.perf_counter() - t0) * 1000)

    meta = build_model_meta(model, model_path, loaded_ms, pointer=pointer_meta)
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
# Model control endpoints (Path A)
# ---------------------------
@app.get("/model/info")
def model_info(request: Request):
    with model_lock:
        meta = getattr(request.app.state, "model_meta", None)
    return meta or {"status": "not_ready"}


@app.post("/model/reload")
def model_reload(request: Request):
    """
    Pull latest production pointer + versioned model from S3 and reload in-memory model.
    Keep this internal-only (not public).
    """
    t0 = time.perf_counter()
    model, model_path, meta = load_model_bundle()
    meta["reload_ms"] = int((time.perf_counter() - t0) * 1000)

    with model_lock:
        request.app.state.model = model
        request.app.state.model_path = model_path
        request.app.state.model_meta = meta

    # Keep response small and useful
    return {"status": "reloaded", "reload_ms": meta["reload_ms"], "model_path": str(model_path)}


# ---------------------------
# Inference
# ---------------------------
@app.post("/recommendations")
def recommendations(req: PredictRequest, request: Request):
    with model_lock:
        model = request.app.state.model

    rec_map = model.get("recommendations") or {}
    recs = rec_map.get(req.user_id) or model.get("popular") or [1, 2, 3, 101, 102]

    return {
        "user_id": req.user_id,
        "recommendations": recs,
        "source": "ml",
    }
@app.get("/recommendations/{user_id}")
def recommendations_get(user_id: int, request: Request):
    # Compatibility route for callers that use GET /recommendations/{id}
    return recommendations(PredictRequest(user_id=user_id), request)


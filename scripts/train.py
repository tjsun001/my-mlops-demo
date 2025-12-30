import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from utils.db import get_conn

ROOT = Path(__file__).resolve().parents[1]  # repo root (mlops/)
MODEL_PATH = ROOT / "models" / "model.pkl"
LAST_TRAIN_FILE = ROOT / "models" / "last_train.txt"
TABLE_NAME = os.getenv("WORKFLOW_TABLE", "user_events")


def get_new_data():
    conn = get_conn()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY created_at", conn)
    conn.close()
    return df


def load_last_train_time():
    if LAST_TRAIN_FILE.exists():
        return datetime.fromisoformat(LAST_TRAIN_FILE.read_text().strip())
    return None


def save_last_train_time():
    LAST_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_TRAIN_FILE.write_text(datetime.now().isoformat())


def build_recommender_artifact(df):
    print("COLUMNS:", df.columns.tolist())
    print("HEAD user_id/product_id:\n", df[["user_id", "product_id"]].head())
    print("N unique users:", df["user_id"].nunique(), "N unique products:", df["product_id"].nunique())

    user_counts = defaultdict(Counter)
    global_counts = Counter()

    # Assumes df has columns: user_id, product_id
    for uid, pid in zip(df["user_id"], df["product_id"]):
        uid = int(uid)
        pid = int(pid)
        user_counts[uid][pid] += 1
        global_counts[pid] += 1

    return {
        "type": "popularity_recommender_v1",
        "user_top_products": {
            uid: [pid for pid, _ in ctr.most_common(50)]
            for uid, ctr in user_counts.items()
        },
        "global_top_products": [pid for pid, _ in global_counts.most_common(100)],
    }


def train_model(df):
    print(f"Training model on {len(df)} rows...")
    model = build_recommender_artifact(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    save_last_train_time()
    size = MODEL_PATH.stat().st_size
    print(f"Training complete. Model saved at {MODEL_PATH}. Size={size} bytes")


def retrain_model(new_rows_df):
    print(f"Retraining model on {len(new_rows_df)} new rows...")

    # simplest baseline: rebuild from full dataset
    full_df = get_new_data()
    model = build_recommender_artifact(full_df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    save_last_train_time()
    size = MODEL_PATH.stat().st_size
    print(f"Retraining complete. Model updated at {MODEL_PATH}. Size={size} bytes")


if __name__ == "__main__":
    df = get_new_data()
    last_train_time = load_last_train_time()

    if MODEL_PATH.exists() and last_train_time:
        new_data = df[df["created_at"] > last_train_time]
        print(f"New rows since last training: {len(new_data)}")
        if not new_data.empty:
            retrain_model(new_data)
        else:
            print("No new data. Skipping retraining.")
    else:
        print(f"No existing model found. Training from scratch on {len(df)} rows.")
        train_model(df)

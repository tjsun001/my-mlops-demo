import os
import pandas as pd
from utils.db import get_conn
from pathlib import Path
import pickle
from datetime import datetime

MODEL_PATH = Path("models/model.pkl")
LAST_TRAIN_FILE = Path("models/last_train.txt")
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
    LAST_TRAIN_FILE.write_text(datetime.now().isoformat())

def train_model(df):
    print(f"Training model on {len(df)} rows...")
    model = {"dummy_model": True}  # Replace with real training logic
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    save_last_train_time()
    print(f"Training complete. Model saved at {MODEL_PATH}.")

def retrain_model(df):
    print(f"Retraining model on {len(df)} new rows...")
    model = {"dummy_model": True}  # Replace with real retraining logic
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    save_last_train_time()
    print(f"Retraining complete. Model updated at {MODEL_PATH}.")

if __name__ == "__main__":
    df = get_new_data()
    last_train_time = load_last_train_time()
    
    if MODEL_PATH.exists() and last_train_time:
        new_data = df[df["created_at"] > last_train_time]
        print(f"New rows since last training: {len(new_data)}")  # <-- This is the report
        if not new_data.empty:
            retrain_model(new_data)
        else:
            print("No new data. Skipping retraining.")
    else:
        print(f"No existing model found. Training from scratch on {len(df)} rows.")
        train_model(df)

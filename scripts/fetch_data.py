from utils.db import get_conn
import pandas as pd
import os

TABLE_NAME = os.getenv("WORKFLOW_TABLE", "user_events")

def fetch_user_events():
    conn = get_conn()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY created_at", conn)
    conn.close()
    return df

if __name__ == "__main__":
    df = fetch_user_events()
    print(f"Fetched {len(df)} rows from {TABLE_NAME}")

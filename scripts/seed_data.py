from utils.db import get_conn
import random
from datetime import datetime, timedelta
import os

NUM_USERS = 10
NUM_PRODUCTS = 10
NUM_EVENTS = 50
EVENT_TYPES = ["view", "purchase", "add_to_cart"]

# Choose table based on environment
TABLE_NAME = os.getenv("WORKFLOW_TABLE", "user_events")

def generate_events():
    events = []
    for _ in range(NUM_EVENTS):
        user_id = random.randint(1, NUM_USERS)
        product_id = random.randint(1, NUM_PRODUCTS)
        event_type = random.choice(EVENT_TYPES)
        created_at = datetime.now() - timedelta(days=random.randint(0, 30))
        events.append((user_id, product_id, event_type, created_at))
    return events

def seed_data():
    conn = get_conn()
    cur = conn.cursor()

    # Create the table if it doesn't exist
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        user_id INT NOT NULL,
        product_id INT NOT NULL,
        event_type VARCHAR(50) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Insert synthetic events
    events = generate_events()
    for user_id, product_id, event_type, created_at in events:
        cur.execute(f"""
        INSERT INTO {TABLE_NAME} (user_id, product_id, event_type, created_at)
        VALUES (%s, %s, %s, %s)
        """, (user_id, product_id, event_type, created_at))

    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {len(events)} synthetic events into {TABLE_NAME} table.")

if __name__ == "__main__":
    seed_data()

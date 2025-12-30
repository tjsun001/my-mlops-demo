import os
import psycopg2
from psycopg2 import OperationalError

DB_NAME = os.getenv("DB_NAME", "demo")
DB_USER = os.getenv("DB_USER", "demo")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)

def get_conn():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except OperationalError as e:
        print("Error connecting to Postgres:", e)
        raise

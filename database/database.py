import sqlite3
import os
from datetime import datetime

# Explicit absolute path to ensure DB is created in the same folder as database.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "crowd_monitoring.db")

def get_connection():
    """Returns a new thread-safe SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    return conn

def init_db():
    """Initializes SQLite schema for telemetry and incident tracking."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS CAMERA (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT UNIQUE NOT NULL,
            location_tag TEXT DEFAULT 'General Surveillance',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS CROWD_RECORD (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            source_name TEXT NOT NULL,
            head_count INTEGER NOT NULL,
            density_status TEXT NOT NULL,
            fps REAL NOT NULL,
            latency_ms REAL NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ALERT (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            source_name TEXT NOT NULL,
            alert_level TEXT NOT NULL,
            head_count INTEGER NOT NULL,
            message TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f"[INFO] SQLite database initialized at: {DB_PATH}")

def register_camera_source(source_name):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO CAMERA (source_name) VALUES (?)", (source_name,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] register_camera_source: {e}")

def log_crowd_record(source_name, head_count, status, fps, latency_ms):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO CROWD_RECORD (timestamp, source_name, head_count, density_status, fps, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (now, source_name, head_count, status, fps, latency_ms))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] log_crowd_record: {e}")

def log_alert_event(source_name, alert_level, head_count, message):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO ALERT (timestamp, source_name, alert_level, head_count, message)
            VALUES (?, ?, ?, ?, ?)
        """, (now, source_name, alert_level, head_count, message))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] log_alert_event: {e}")

def fetch_recent_logs(limit=500):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, source_name, head_count, density_status, fps, latency_ms 
            FROM CROWD_RECORD ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"[ERROR] fetch_recent_logs: {e}")
        return []
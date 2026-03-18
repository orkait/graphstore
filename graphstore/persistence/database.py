"""SQLite wrapper for graphstore persistence.

Manages the database schema (blobs, wal, query_log, metadata tables),
connection setup with WAL mode, and metadata helpers.
"""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1


def open_database(path: str | Path) -> sqlite3.Connection:
    """Open or create the graphstore database."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("PRAGMA busy_timeout=5000")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blobs (
            key TEXT PRIMARY KEY,
            data BLOB,
            dtype TEXT
        );
        CREATE TABLE IF NOT EXISTS wal (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            statement TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            query TEXT NOT NULL,
            elapsed_us INTEGER NOT NULL,
            result_count INTEGER NOT NULL,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()


def get_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def set_metadata(conn: sqlite3.Connection, key: str, value: str):
    conn.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (key, value))
    conn.commit()

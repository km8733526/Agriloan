"""
database/init_db.py
-------------------
SQLite database initialisation and connection helper.

Tables created:
    farmers          – personal details of each applicant
    land_details     – land parcel data linked to a farmer
    loan_applications – ML evaluation results for each application

Design notes:
    - Uses WAL journal mode for better concurrent read performance.
    - Foreign keys enforced via PRAGMA.
    - All functions are idempotent (safe to call at every app startup).
    - Uses sqlite3.Row so rows can be accessed like dicts.

Usage:
    from database.init_db import init_db, get_connection
    init_db()                      # call once at startup
    conn = get_connection()        # get a connection for a request
"""

import os
import sqlite3

# Database file lives inside the database/ package directory
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")


# ─────────────────────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """
    Return a configured SQLite connection.

    Configuration:
        - row_factory = sqlite3.Row (dict-like row access)
        - foreign_keys = ON
        - journal_mode = WAL (better concurrency)

    The caller is responsible for closing the connection.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ─────────────────────────────────────────────────────────────────────────────

_DDL_STATEMENTS = [

    # ── 1. Farmers master table ────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS farmers (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name  TEXT    NOT NULL,
        aadhaar    TEXT    NOT NULL,
        phone      TEXT    NOT NULL,
        district   TEXT    NOT NULL,
        village    TEXT    NOT NULL,
        created_at TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ── 2. Land details (linked to a farmer) ──────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS land_details (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_id     INTEGER NOT NULL
                              REFERENCES farmers(id) ON DELETE CASCADE,
        survey_number TEXT    NOT NULL,
        land_size     REAL    NOT NULL,
        irrigation    TEXT    NOT NULL,
        ownership     TEXT    NOT NULL,
        latitude      REAL    NOT NULL,
        longitude     REAL    NOT NULL,
        created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ── 3. Loan applications – ML results + crop data ─────────────────────
    """
    CREATE TABLE IF NOT EXISTS loan_applications (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_id             INTEGER NOT NULL
                                      REFERENCES farmers(id) ON DELETE CASCADE,
        primary_crop          TEXT    NOT NULL,
        crop_diversity        INTEGER NOT NULL,
        yield_1               REAL,
        yield_2               REAL,
        yield_3               REAL,
        trust_score           INTEGER NOT NULL,
        risk_level            TEXT    NOT NULL,
        repayment_probability REAL    NOT NULL,
        status                TEXT    NOT NULL,
        soil_type             TEXT,
        soil_ph               REAL,
        annual_rainfall       INTEGER,
        drought_risk          TEXT,
        created_at            TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ── Performance indexes for dashboard queries ──────────────────────────
    "CREATE INDEX IF NOT EXISTS idx_apps_status      ON loan_applications(status)",
    "CREATE INDEX IF NOT EXISTS idx_apps_created_at  ON loan_applications(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_apps_farmer_id   ON loan_applications(farmer_id)",
    "CREATE INDEX IF NOT EXISTS idx_apps_trust_score ON loan_applications(trust_score)",
    "CREATE INDEX IF NOT EXISTS idx_land_farmer_id   ON land_details(farmer_id)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation entry point
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all required tables and indexes if they do not already exist.
    Idempotent – safe to call every time the Flask app starts.

    Raises:
        sqlite3.Error: If the database file cannot be created or any DDL fails.
    """
    # Ensure the database directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with get_connection() as conn:
        for stmt in _DDL_STATEMENTS:
            conn.execute(stmt)
        conn.commit()

    print(f"[DB] SQLite database initialised → {DB_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers used by app.py routes
# ─────────────────────────────────────────────────────────────────────────────

def row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain Python dict."""
    return dict(row)


def rows_to_list(rows: list[sqlite3.Row]) -> list[dict]:
    """Convert a list of sqlite3.Row objects to a list of dicts."""
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Run standalone: python database/init_db.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("[DB] All tables created successfully.")
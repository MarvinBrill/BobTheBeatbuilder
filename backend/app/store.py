import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from .config import DATA, inside


def portable(value):
    if isinstance(value, dict):
        return {key: portable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [portable(child) for child in value]
    if isinstance(value, str) and value.startswith(str(DATA)):
        path = Path(value)
        if path.is_relative_to(DATA):
            return "data://" + path.relative_to(DATA).as_posix()
    return value


def local(value):
    if isinstance(value, dict):
        return {key: local(child) for key, child in value.items()}
    if isinstance(value, list):
        return [local(child) for child in value]
    if isinstance(value, str) and value.startswith("data://"):
        return str(inside(DATA, value[7:]))
    return value


@contextmanager
def connect():
    connection = sqlite3.connect(DATA / "bob.sqlite3", timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    # WAL keeps the database consistent with NORMAL; FULL only adds an fsync per commit.
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, body TEXT, PRIMARY KEY(kind,id))"
    )
    connection.execute("CREATE TABLE IF NOT EXISTS state_revision (value INTEGER NOT NULL)")
    if connection.execute("SELECT value FROM state_revision").fetchone() is None:
        connection.execute("INSERT INTO state_revision VALUES (0)")
    for action in ("INSERT", "UPDATE", "DELETE"):
        connection.execute(
            f"CREATE TRIGGER IF NOT EXISTS records_revision_{action.lower()} "
            f"AFTER {action} ON records BEGIN UPDATE state_revision SET value=value+1; END"
        )
    connection.commit()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(kind: str, item: dict):
    with connect() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO records VALUES (?,?,?)",
            (kind, item["id"], json.dumps(portable(item))),
        )
    return item


def save_many(kind: str, items):
    """Insert or replace many records in one transaction, keeping the given order."""
    items = list(items)
    with connect() as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO records VALUES (?,?,?)",
            ((kind, item["id"], json.dumps(portable(item))) for item in items),
        )
    return items


def get_many(kind: str, identifiers):
    """Existing records by id; unknown ids are simply absent."""
    identifiers = list(dict.fromkeys(identifiers))
    found = {}
    with connect() as connection:
        for start in range(0, len(identifiers), 500):
            chunk = identifiers[start : start + 500]
            marks = ",".join("?" * len(chunk))
            rows = connection.execute(
                f"SELECT id, body FROM records WHERE kind=? AND id IN ({marks})", (kind, *chunk)
            ).fetchall()
            found.update((row[0], local(json.loads(row[1]))) for row in rows)
    return found


def get(kind: str, identifier: str):
    with connect() as connection:
        row = connection.execute(
            "SELECT body FROM records WHERE kind=? AND id=?", (kind, identifier)
        ).fetchone()
    if not row:
        raise ValueError(f"Unknown {kind}: {identifier}")
    return local(json.loads(row[0]))


def all_records(kind: str):
    with connect() as connection:
        rows = connection.execute(
            "SELECT body FROM records WHERE kind=? ORDER BY rowid DESC", (kind,)
        ).fetchall()
    return [local(json.loads(row[0])) for row in rows]


def count(kind: str):
    with connect() as connection:
        return connection.execute("SELECT COUNT(*) FROM records WHERE kind=?", (kind,)).fetchone()[
            0
        ]


def revision():
    with connect() as connection:
        return connection.execute("SELECT value FROM state_revision").fetchone()[0]

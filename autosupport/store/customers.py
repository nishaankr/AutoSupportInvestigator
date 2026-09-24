"""Repository for the `customers` table (memory-design.md). Read by `load_memory`, written
only by `update_memory`."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from autosupport.graph.state import CustomerMemory


def upsert(conn: sqlite3.Connection, memory: CustomerMemory) -> None:
    conn.execute(
        "INSERT INTO customers (customer_id, profile, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(customer_id) DO UPDATE SET profile = excluded.profile, updated_at = excluded.updated_at",
        (memory.customer_id, memory.model_dump_json(), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def get(conn: sqlite3.Connection, customer_id: str) -> CustomerMemory | None:
    row = conn.execute("SELECT profile FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    if row is None:
        return None
    return CustomerMemory.model_validate_json(row["profile"])

"""Repository for the `customers` table (memory-design.md). CP3 only reads — nothing
writes until `update_memory` (CP6)."""

from __future__ import annotations

import sqlite3

from autosupport.graph.state import CustomerMemory


def get(conn: sqlite3.Connection, customer_id: str) -> CustomerMemory | None:
    row = conn.execute("SELECT profile FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    if row is None:
        return None
    return CustomerMemory.model_validate_json(row["profile"])

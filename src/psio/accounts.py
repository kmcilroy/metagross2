# src/psio/accounts.py
from __future__ import annotations
import uuid


def fresh_username(prefix: str, wid: int, ep: int) -> str:
    return (f"{prefix}{wid}{ep}{uuid.uuid4().hex[:6]}")[:18]

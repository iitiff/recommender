"""ID helpers."""

from __future__ import annotations

import os
import time
import uuid


def uuid7() -> uuid.UUID:
    """A UUIDv7 (RFC 9562): 48-bit millisecond timestamp + random bits.

    Event IDs are UUIDv7 rather than v4 because they sort by creation time,
    which keeps the events index append-friendly and makes IDs readable in
    order when debugging. Python's stdlib gains `uuid.uuid7` in 3.14.
    """
    ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76  # version
    value |= ((rand >> 62) & 0xFFF) << 64  # rand_a, 12 bits
    value |= 0b10 << 62  # RFC 4122 variant
    value |= rand & ((1 << 62) - 1)  # rand_b, 62 bits
    return uuid.UUID(int=value)


def new_id() -> str:
    return str(uuid7())
